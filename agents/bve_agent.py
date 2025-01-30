import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
import wandb

from agents.common import load_transitions
from agents.evaluate.bve import evaluate
from grid_world import GridWorld
from per import PrioritizedReplayBuffer


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    runs_dir: str = "runs"
    """directory into which run data will be stored"""
    wandb_project_name: str = "offline_bve"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model"""
    randomize_initial_state: bool = False
    """if toggled, agent will start in random (non-terminal) grid location"""
    data_load_path: str = None
    """file path for offline data to be loaded"""
    small_state: bool = False

    # Algorithm specific arguments
    num_gradient_steps: int = 20000
    """total number of gradients steps to perform"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_network_layers: int = 6
    gamma: float = 0.9
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 50
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    q_weight: float = 0.5
    num_beams: int = 5
    """number of beams for evaluation"""
    lr_decay_rate: float = 0.99995
    """Multiplicative factor of learning rate decay"""
    temp: float = 0.5
    """Temperature for softmax"""
    delta: float = 1
    """Depth penalty multiplier"""
    q_loss_multiplier: float = 10


class BVE(nn.Module):
    def __init__(self, in_size, env, num_layers, hidden_size=256):
        super().__init__()
        layers = [
            nn.Linear(in_size, hidden_size),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])

        layers.append(nn.Linear(hidden_size, 2 * env.grid_dimension + 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x, a):
        x = torch.cat((x, a), dim=1)
        return self.network(x)


def compute_children(env, idx):
    action_shape = (2,) * 2 * env.grid_dimension
    current_action = env.compute_action_from_index(idx)
    last_activated_sub_action = np.max(np.where(current_action == 1)[0], initial=-1)

    children = []
    for i in range(last_activated_sub_action + 1, len(current_action)):
        child_action = np.copy(current_action)
        child_action[i] = 1
        child_index = np.ravel_multi_index(tuple(child_action), action_shape)
        children.append(child_index)

    return children


def compute_action_branch(env, idx):
    action_shape = (2,) * 2 * env.grid_dimension
    current_action = env.compute_action_from_index(idx)
    branch = [idx]

    last_action = current_action.copy()
    for i in range(len(last_action) - 1, -1, -1):
        if last_action[i] == 1:
            last_action[i] = 0
            parent_index = np.ravel_multi_index(tuple(last_action), action_shape)
            branch.append(parent_index)

    return torch.tensor(branch)


def compute_bve_loss(env, action_branches, bves, q_values, device, sab, delta):
    loss_terms = []
    all_actions_idx = 0

    for i, branch in enumerate(action_branches):
        action = branch[0]
        target = q_values[i]
        loss = (bves[all_actions_idx][0] - target).pow(2)
        loss_terms.append(loss)
        all_actions_idx += 1

        for depth, a in enumerate(branch[1:]):
            children = compute_children(env, a)
            children = [child for child in children if child in sab]
            idx = children.index(action)
            action_bves = bves[all_actions_idx][:len(children) + 1]
            depth_penalty = delta * (depth + 1)
            loss = ((action_bves[idx + 1] - target) * depth_penalty).pow(2)
            loss_terms.append(loss)

            max_bve = torch.max(action_bves)
            target = torch.max(max_bve, target)
            action = a
            all_actions_idx += 1

    return torch.mean(torch.stack(loss_terms)) if loss_terms else torch.zeros(1, device=device)


def compute_action(env, network, obs, device, sa, sab, temperature):
    obs = torch.Tensor(obs).to(device)
    action = 0

    while True:
        children = compute_children(env, action)
        children = [c for c in children if c in sab]

        with torch.no_grad():
            action_tensor = torch.tensor(env.compute_action_from_index(action), device=device).view(1, -1)
            values = network(obs.unsqueeze(0), action_tensor).flatten()[:len(children) + 1]

        if not children:
            return np.array([action])

        probabilities = torch.softmax(values, dim=-1)
        action_index = torch.multinomial(probabilities / temperature, 1).item()

        if action_index == 0:
            if action in sa:
                return np.array([action])
            masked_values = values.clone()
            masked_values[action_index] = float('-inf')
            action_index = torch.argmax(masked_values).item()

        action = children[action_index - 1]


def train_net(net, a_optimizer, device, env, actions, observations, td_targets, sab, delta, rb, data, q_loss_multiplier):
    with torch.no_grad():
        full_actions = np.array([env.compute_action_from_index(a.cpu().numpy()) for a in actions])
        full_actions = torch.from_numpy(full_actions).to(device)

        action_branches = [compute_action_branch(env, int(a.item())) for a in actions]
        all_obs = torch.cat([observations[i].repeat(len(branch), 1, 1) for i, branch
                             in enumerate(action_branches)], dim=0)
        all_obs = all_obs.view(-1, *all_obs.size()[2:])

        all_actions = torch.cat([action.clone().detach().to(device).unsqueeze(0)
                                 for branch in action_branches for action in branch], dim=0).unsqueeze(1)
        full_all_actions = np.array([env.compute_action_from_index(a.cpu().numpy()) for a in all_actions])
        full_all_actions = torch.from_numpy(full_all_actions).to(device)

    bves = net(all_obs, full_all_actions)
    q_values = net(observations, full_actions)[:, 0:1].flatten()
    bve_loss = compute_bve_loss(env, action_branches, bves, td_targets, device, sab, delta)
    q_loss = ((q_values - td_targets) ** 2).mean()
    total_loss = bve_loss + q_loss_multiplier*q_loss

    a_optimizer.zero_grad()
    total_loss.backward()
    a_optimizer.step()

    td_error = torch.abs(td_targets - q_values)
    rb_weights = (td_error + 1e-8).detach().cpu().numpy()  # small constant added to avoid zero weights
    rb.update_weights(data.indices, rb_weights)

    return bve_loss, q_loss


def hard_update(local_model, target_model):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)


def learn(args, env, rb, net, target_net, optimizer, scheduler, device, writer, sa, sab):
    for global_step in range(args.num_gradient_steps):
        data = rb.sample(args.batch_size, beta=1)
        observations = data.observations.to(torch.float)
        actions = data.actions
        next_observations = data.next_observations.to(torch.float)
        behavior_next_actions = data.next_actions
        dones = data.dones.flatten()
        rewards = data.rewards.flatten()

        with torch.no_grad():
            next_actions = [compute_action(env, net, o, device, sa, sab, args.temp) for o in next_observations]
            next_actions = np.array([env.compute_action_from_index(na) for na in next_actions])
            next_actions = torch.from_numpy(next_actions).to(device)

            behavior_next_actions = np.array([env.compute_action_from_index(bna.cpu().numpy()) for bna in
                                              behavior_next_actions])
            behavior_next_actions = torch.from_numpy(behavior_next_actions).to(device)

            target_q_value = target_net(next_observations, next_actions)[:, 0:1].flatten()
            target_q_value = args.q_weight * target_q_value
            bc_penalty = ((behavior_next_actions - next_actions) ** 2).sum(dim=1)
            td_targets = rewards + args.gamma * target_q_value - bc_penalty * (1 - dones)

        bve_loss, q_loss = train_net(net, optimizer, device, env, actions, observations, td_targets, sab, args.delta,
                                     rb, data, args.q_loss_multiplier)
        scheduler.step()

        if global_step % 100 == 0:
            net.eval()
            reward, _ = evaluate(env, 1, net, device, seen_actions,
                                 seen_action_branches, args.num_beams)
            print(f"global_step={global_step}, episodic_return={reward.mean()}")

            writer.add_scalar("learning/episodic_return", reward.mean(), global_step)
            writer.add_scalar("learning/good_examples", (data.rewards == 10).sum().item(), global_step)
            writer.add_scalar("loss/critic1_loss", q_loss, global_step)
            writer.add_scalar("loss/bve_loss", bve_loss.mean().item(), global_step)
            net.train()

            if arguments.save_model:
                path = f"models/bve/{grid_dimension}-{grid_size}-{num_pits}-{suboptimal_rate}/{args.seed}/{global_step}"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(agent_network.state_dict(), path)
                print(f'Saved model to {path}')

        if global_step % args.target_network_frequency == 0:
            hard_update(net, target_net)


if __name__ == "__main__":
    arguments = tyro.cli(Args)

    parts = arguments.data_load_path.split('/')[1].split('-', 4)[:4]
    grid_dimension, grid_size, num_pits, suboptimal_rate = (*map(int, parts[:3]), parts[3])

    exp_name = f'{grid_dimension}-{grid_size}-{num_pits}-{suboptimal_rate}-{os.path.basename(__file__)[: -len(".py")]}'
    run_name = f"{exp_name}__{arguments.seed}__{int(time.time())}"

    if arguments.track:
        wandb.init(project=arguments.wandb_project_name, entity=arguments.wandb_entity, sync_tensorboard=True,
                   config=vars(arguments), name=run_name, monitor_gym=True, save_code=False)
    wtr = SummaryWriter(f"{arguments.runs_dir}/{run_name}")
    wtr.add_text("hyperparameters", "|param|value|\n|-|-|\n%s"
                 % ("\n".join([f"|{key}|{value}|" for key, value in vars(arguments).items()])))

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.backends.cudnn.deterministic = arguments.torch_deterministic
    d = torch.device("cuda" if torch.cuda.is_available() and arguments.cuda else "cpu")

    environment = GridWorld(grid_dimension=grid_dimension, grid_size=grid_size, num_total_pits=num_pits,
                            num_clusters=1, distribute_pits_evenly=True, small_state=arguments.small_state,
                            max_steps_per_episode=100, randomize_initial_state=arguments.randomize_initial_state)

    if arguments.small_state:
        input_size = 3 * environment.grid_dimension
    else:
        input_size = 2 * environment.grid_dimension + len(environment.terminal_states) + environment.grid_dimension

    agent_network = BVE(input_size, environment, arguments.num_network_layers).to(d)
    opt = optim.Adam(agent_network.parameters(), lr=arguments.learning_rate)
    a_scheduler = ExponentialLR(opt, gamma=arguments.lr_decay_rate)

    target_network = BVE(input_size, environment, arguments.num_network_layers).to(d)
    target_network.load_state_dict(agent_network.state_dict())

    all_transitions = load_transitions(arguments.data_load_path)
    num_possible_actions = 2 ** (2 * grid_dimension) - 1
    state_size = environment.grid_dimension if arguments.small_state else len(environment.terminal_states) + environment.grid_dimension

    replay_buffer = PrioritizedReplayBuffer(
        buffer_size=len(all_transitions),
        alpha=1,
        observation_space=spaces.MultiDiscrete([grid_size] * state_size),
        action_space=spaces.Discrete(num_possible_actions),
        device=d
    )

    seen_actions = set()
    num_successful_transitions = 0
    for t in all_transitions:
        # obs, next_obs, action, next_action, reward, termination, info
        replay_buffer.add(t[0][:state_size], t[1][:state_size], t[2], t[3], t[4], t[5])
        seen_actions.add(t[2])
        if t[4] == 10:
            num_successful_transitions += 1

    print(f'Loaded {len(all_transitions)} transitions of which '
          f'{num_successful_transitions} are transitions to the goal')

    seen_action_branches = set()
    for act in seen_actions:
        for ab in compute_action_branch(environment, act):
            seen_action_branches.add(ab.item())

    learn(arguments, environment, replay_buffer, agent_network, target_network, opt, a_scheduler, d, wtr, seen_actions,
          seen_action_branches)

    wtr.close()

    if arguments.save_model:
        model_save_path = f"models/bve/{grid_dimension}-{grid_size}-{num_pits}-{suboptimal_rate}-{arguments.num_gradient_steps}"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(agent_network.state_dict(), model_save_path)
        print(f'Saved model to {model_save_path}')
