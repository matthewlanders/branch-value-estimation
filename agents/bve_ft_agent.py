import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
import wandb

from agents.common import load_transitions
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
    runs_dir: str = "ft_runs"
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
    suboptimal_rate: float = 0.9
    """percentage of the time random, suboptimal actions are taken"""

    # Algorithm specific arguments
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
    beam_width: int = 5
    """number of beams for evaluation"""
    lr_decay_rate: float = 0.99995
    """Multiplicative factor of learning rate decay"""
    temp: float = 0.5
    """Temperature for softmax"""
    delta: float = 1
    """Depth penalty multiplier"""
    q_loss_multiplier: float = 10
    total_timesteps: int = 5000
    learning_starts: int = 100
    buffer_size: int = 5000


class BVE(nn.Module):
    def __init__(self, env, num_layers, hidden_size=256):
        super().__init__()
        layers = [
            nn.Linear(3 * env.grid_dimension, hidden_size),
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


def compute_action(env, network, obs, device, sa, sab, k):
    obs = torch.Tensor(obs[:env.grid_dimension]).to(device)
    beams = [0]
    beam_values = [-float('inf')]
    beams_to_explore = [0]
    explored_beams = set()

    while beams_to_explore:
        action = beams_to_explore.pop(0)
        if action in explored_beams:
            continue
        explored_beams.add(action)

        children = compute_children(env, action)
        children = [c for c in children if c in sab]

        with torch.no_grad():
            action_tensor = torch.tensor(env.compute_action_from_index(action), device=device).view(1, -1)
            values = network(obs.unsqueeze(0), action_tensor).flatten()[:len(children) + 1]

        # Use Q value instead of BVE
        action_idx = beams.index(action)
        beam_values[action_idx] = values[0].item()

        top_action_values, top_action_indices = torch.topk(values, min(k, len(children) + 1))

        if 0 in top_action_indices:
            if action not in sa:
                masked_values = values.clone()
                masked_values[0] = float('-inf')
                top_action_values, top_action_indices = torch.topk(masked_values, min(k, len(children) + 1))

        for i, action_value in enumerate(top_action_values):
            new_action = children[top_action_indices[i] - 1] if top_action_indices[i] > 0 else action
            if new_action not in explored_beams and new_action not in beams_to_explore:
                if len(beams) == k:
                    if action_value.item() >= min(beam_values):
                        min_action_value_idx = beam_values.index(min(beam_values))
                        action_to_remove = beams[min_action_value_idx]

                        if action_to_remove not in explored_beams:
                            beams_to_explore_idx = beams_to_explore.index(action_to_remove)
                            beams_to_explore.pop(beams_to_explore_idx)

                        beams[min_action_value_idx] = new_action
                        beam_values[min_action_value_idx] = action_value.item()
                        beams_to_explore.append(new_action)

                else:
                    beams.append(new_action)
                    beam_values.append(action_value.item())
                    beams_to_explore.append(new_action)

    max_index = beam_values.index(max(beam_values))
    return np.array([beams[max_index]])


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

    return bve_loss, q_loss


def hard_update(local_model, target_model):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)


def learn(args, env, rb, net, target_net, optimizer, scheduler, device, writer, sa, sab):
    obs = env.reset()
    for global_step in range(args.total_timesteps):
        actions = compute_action(env, net, obs, device, sa, sab, args.beam_width)
        next_obs, rewards, terminations, info = env.step(actions)
        real_next_obs = next_obs.copy()

        if "final_info" in info:
            data = info['final_info']['episode']
            print(f"global_step={global_step}, episodic_return={data['r']}")
            writer.add_scalar("learning/episodic_return", data["r"], global_step)
            writer.add_scalar("charts/episodic_length", data["l"], global_step)
            real_next_obs = data['final_observation']

        rb.add(obs, real_next_obs, actions, rewards, terminations, info)

        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            observations = data.observations.to(torch.float)
            actions = data.actions
            next_observations = data.next_observations.to(torch.float)

            with torch.no_grad():
                next_actions = np.stack([compute_action(env, net, o, device, sa, sab, 1) for o in next_observations])
                next_actions = np.array([env.compute_action_from_index(na) for na in next_actions])
                next_actions = torch.from_numpy(next_actions).to(device)

                target_q_value = target_net(next_observations, next_actions)[:, 0]
                td_targets = data.rewards.flatten() + args.gamma * target_q_value * (1 - data.dones.flatten())

            bve_loss, q_loss = train_net(net, optimizer, device, env, actions, observations, td_targets, sab,
                                         args.delta, rb, data, args.q_loss_multiplier)
            scheduler.step()

            if global_step % 100 == 0:
                writer.add_scalar("learning/q_values", q_loss, global_step)
                writer.add_scalar("learning/bve_loss", bve_loss.mean().item(), global_step)

        if global_step % args.target_network_frequency == 0:
            for target_param, q_param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(args.tau * q_param.data + (1.0 - args.tau) * target_param.data)


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

    agent_network = BVE(environment, arguments.num_network_layers).to(d)
    model_load_path = f"models/bve/{grid_dimension}-{grid_size}-{num_pits}-{suboptimal_rate}"
    agent_network.load_state_dict(torch.load(model_load_path))
    opt = optim.Adam(agent_network.parameters(), lr=arguments.learning_rate)
    a_scheduler = ExponentialLR(opt, gamma=arguments.lr_decay_rate)

    target_network = BVE(environment, arguments.num_network_layers).to(d)
    target_network.load_state_dict(agent_network.state_dict())

    all_transitions = load_transitions(arguments.data_load_path)
    num_possible_actions = 2 ** (2 * grid_dimension) - 1
    state_size = environment.grid_dimension

    replay_buffer = ReplayBuffer(
        arguments.buffer_size,
        spaces.MultiDiscrete([grid_size] * state_size),
        spaces.Discrete(num_possible_actions),
        d,
        handle_timeout_termination=False,
    )

    seen_actions = set()
    for t in all_transitions:
        seen_actions.add(t[2])

    seen_action_branches = set()
    for act in seen_actions:
        for ab in compute_action_branch(environment, act):
            seen_action_branches.add(ab.item())

    learn(arguments, environment, replay_buffer, agent_network, target_network, opt, a_scheduler, d, wtr, seen_actions,
          seen_action_branches)

    wtr.close()
