import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
import wandb

from agents.common import load_transitions, terminal_states_dict
from agents.evaluate.dqn import evaluate
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
    wandb_project_name: str = "bve"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    randomize_initial_state: bool = False
    """if toggled, agent will start in random (non-terminal) grid location"""
    data_load_path: str = None
    """file path for offline data to be loaded"""
    small_state: bool = False
    suboptimal_rate: float = 0.9
    """percentage of the time random, suboptimal actions are taken"""

    # Algorithm specific arguments
    num_gradient_steps: int = 20000
    """total number of gradients steps to perform"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_network_layers: int = 2
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 50
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    q_weight: float = 0.5
    """the batch size of sample from the reply memory"""


class QNetwork(nn.Module):
    def __init__(self, env, num_actions, num_layers, hidden_size=256):
        super().__init__()
        layers = [
            nn.Linear(len(env.terminal_states) + env.grid_dimension, hidden_size),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])

        layers.append(nn.Linear(hidden_size, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def hard_update(local_model, target_model):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)


def learn(args, env, rb, q_network, target_network, optimizer, device, writer, actions_map):
    for global_step in range(args.num_gradient_steps):
        data = rb.sample(args.batch_size, beta=1)
        observations = data.observations.to(torch.float)
        actions = data.actions
        next_observations = data.next_observations.to(torch.float)
        behavior_next_actions = data.next_actions
        dones = data.dones.flatten()
        rewards = data.rewards.flatten()

        with torch.no_grad():
            behavior_next_actions = np.array([env.compute_action_from_index(bna.cpu().numpy()) for bna in
                                              behavior_next_actions])
            behavior_next_actions = torch.from_numpy(behavior_next_actions).to(device)
            next_actions = q_network(next_observations.to(torch.float)).argmax(dim=1)

            target_max = target_network(next_observations.to(torch.float)).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

            next_actions = np.array([env.compute_action_from_index(actions_map[na]) for na in next_actions])
            next_actions = torch.from_numpy(next_actions).to(device)

            bc_penalty = ((behavior_next_actions - next_actions) ** 2).sum(dim=1)

            target_q_value = args.q_weight * target_max - bc_penalty
            td_target = rewards + args.gamma * target_q_value * (1 - dones)

        old_val = q_network(observations.to(torch.float)).gather(1, actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        td_error = torch.abs(td_target - old_val)
        rb_weights = (td_error + 1e-8).detach().cpu().numpy()  # small constant added to avoid zero weights
        rb.update_weights(data.indices, rb_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 100 == 0:
            q_network.eval()
            reward = evaluate(env, 1, q_network, device, actions_map)
            print(f"global_step={global_step}, episodic_return={reward.mean()}")

            writer.add_scalar("learning/episodic_return", reward.mean(), global_step)
            writer.add_scalar("learning/good_examples", (data.rewards == 10).sum().item(), global_step)
            writer.add_scalar("learning/td_loss", loss, global_step)
            writer.add_scalar("learning/q_values", old_val.mean().item(), global_step)
            q_network.train()

        if global_step % args.target_network_frequency == 0:
            hard_update(q_network, target_network)


if __name__ == "__main__":
    args = tyro.cli(Args)

    grid_dimension, grid_size, num_pits, num_clusters = [int(x) for x in args.data_load_path.split('/')[1].split('-', 4)[:4]]

    exp_name = f'{grid_dimension}-{grid_size}-{num_pits}-{arguments.suboptimal_rate}-{os.path.basename(__file__)[: -len(".py")]}'
    run_name = f"{exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args),
                   name=run_name, monitor_gym=True, save_code=False)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    terminal_states = None if num_pits == 0 else terminal_states_dict.get(grid_dimension)
    environment = GridWorld(grid_dimension=grid_dimension, grid_size=grid_size, num_total_pits=num_pits,
                            num_clusters=num_clusters, distribute_pits_evenly=True, only_independent=False,
                            max_steps_per_episode=100, terminal_states=terminal_states,
                            randomize_initial_state=args.randomize_initial_state)

    all_transitions = load_transitions(args.data_load_path)
    num_possible_actions = 2 ** (2 * grid_dimension) - 1
    state_size = len(environment.terminal_states) + environment.grid_dimension

    replay_buffer = PrioritizedReplayBuffer(
        buffer_size=len(all_transitions),
        alpha=1,
        observation_space=spaces.MultiDiscrete([grid_size] * state_size),
        action_space=spaces.Discrete(num_possible_actions),
        device=device
    )

    seen_actions = set()
    num_successful_transitions = 0
    for t in all_transitions:
        # obs, next_obs, action, next_action, reward, termination, info
        seen_actions.add(t[2])
        if t[4] == 10:
            num_successful_transitions += 1

    sa_list = list(seen_actions)
    for t in all_transitions:
        action = t[2]
        a = sa_list.index(action)
        replay_buffer.add(t[0], t[1], np.array(a), t[3], t[4], t[5])

    q_network = QNetwork(environment, len(seen_actions), args.num_network_layers).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(environment, len(seen_actions), args.num_network_layers).to(device)
    target_network.load_state_dict(q_network.state_dict())

    learn(args, environment, replay_buffer, q_network, target_network, optimizer, device, writer, sa_list)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    writer.close()
