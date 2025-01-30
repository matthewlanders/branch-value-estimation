import copy
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
import wandb
from gymnasium import spaces
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from agents.common import load_transitions
from agents.evaluate.fas import evaluate
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
    randomize_initial_state: bool = False
    """if toggled, agent will start in random (non-terminal) grid location"""
    data_load_path: str = None
    """file path for offline data to be loaded"""
    small_state: bool = False

    num_gradient_steps: int = 20000
    """total number of gradients steps to perform"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    lr_decay_rate: float = 0.99995
    """Multiplicative factor of learning rate decay"""
    num_network_layers: int = 2
    alpha: float = 0.005
    """weight of new data in the exponential moving average"""
    tau: float = 0.99
    """expectile"""
    beta: float = 3.0
    """scales advantages before applying an exponential function"""
    gamma: float = 0.9
    """the discount factor gamma"""
    target_network_frequency: int = 50
    """the timesteps it takes to update the target network"""


class BCQf_Net(nn.Module):
    def __init__(self, state_dim, action_dim, num_layers=2, hidden_size=256):
        super().__init__()

        q_layers = [
            nn.Linear(state_dim, hidden_size),
            nn.ReLU()
        ]
        for _ in range(num_layers):
            q_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])
        q_layers.append(nn.Linear(hidden_size, action_dim))
        self.q = nn.Sequential(*q_layers)

        pi_layers = [
            nn.Linear(state_dim, hidden_size),
            nn.ReLU()
        ]
        for _ in range(num_layers):
            pi_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])
        pi_layers.append(nn.Linear(hidden_size, action_dim))
        self.πb = nn.Sequential(*pi_layers)

    def forward(self, x):
        q_values = self.q(x)
        p_logits = self.πb(x)
        return q_values, F.log_softmax(p_logits, dim=-1), p_logits


class BCQf(object):
    def __init__(
            self,
            action_dim,
            state_dim,
            device,
            env,
            BCQ_threshold=0.3,
            discount=0.9,
            target_update_frequency=8e3,
            tau=0.005,
            lr=2.5e-4,
            num_layers=2,
            decay_rate=0.99995,
    ):
        self.device = device
        self.Q = BCQf_Net(state_dim, 2*action_dim, num_layers).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.Q_optimizer, gamma=decay_rate)

        # Freeze target network to avoid accidental training
        for param in self.Q_target.parameters():
            param.requires_grad = False

        self.discount = discount
        self.target_update_frequency = target_update_frequency
        self.tau = tau
        self.threshold = BCQ_threshold
        self.iterations = 0
        self.action_dim = action_dim
        self.env = env

    def hard_update(self):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(local_param.data)

    def get_action(self, state, batch_size):
        with torch.no_grad():
            q_values, _, i = self.Q(state.to(self.device))
            q_values = q_values.reshape(batch_size, self.action_dim, 2)

            imt = F.softmax(i.reshape(batch_size, self.action_dim, 2), dim=-1)
            imt = (imt / imt.max(axis=-1, keepdim=True).values > self.threshold).float()

            masked_q_values = imt * q_values + (1 - imt) * -1e8
            return torch.argmax(masked_q_values, dim=-1)

    def get_q_value(self, batch_size, q_values, sub_action_vec):
        selection_mask = torch.zeros_like(q_values)
        indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, self.action_dim)
        selection_mask[indices, torch.arange(self.action_dim).repeat(batch_size, 1), sub_action_vec] = 1
        action_q_value = q_values * selection_mask
        action_q_value = action_q_value.sum(dim=(1, 2))

        return action_q_value

    def learn(self, args, buffer):
        experiences = buffer.sample(args.batch_size, beta=1)
        states = experiences.observations.to(torch.float).to(self.device)
        actions = experiences.actions.to(self.device)
        next_states = experiences.next_observations.to(torch.float).to(self.device)
        rewards = experiences.rewards.to(torch.float).to(self.device)
        dones = experiences.dones.to(self.device)

        with torch.no_grad():
            next_actions = self.get_action(next_states, args.batch_size)

            next_q_values, _, _ = self.Q_target(next_states)
            next_q_values = next_q_values.reshape(args.batch_size, self.action_dim, 2)
            next_action_q_values = self.get_q_value(args.batch_size, next_q_values, next_actions)
            target_q = rewards.flatten() + (1 - dones.flatten()) * self.discount * next_action_q_values

        current_q, _, i = self.Q(states)
        current_q = current_q.reshape(args.batch_size, self.action_dim, 2)

        sub_action_vec = np.array([self.env.compute_action_from_index(a.cpu().numpy()) for a in actions])
        sub_action_vec = torch.tensor(sub_action_vec, dtype=torch.long, device=self.device)

        action_q_value = self.get_q_value(args.batch_size, current_q, sub_action_vec)

        q_loss = F.smooth_l1_loss(action_q_value, target_q)
        imt = F.log_softmax(i.reshape(args.batch_size, self.action_dim, 2), dim=-1)
        i_loss = F.nll_loss(imt.view(-1, 2), sub_action_vec.flatten().long())
        total_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        self.Q_optimizer.zero_grad()
        total_loss.backward()
        self.Q_optimizer.step()
        self.scheduler.step()

        td_error = torch.abs(target_q - action_q_value)
        rb_weights = (td_error + 1e-8).detach().cpu().numpy()  # small constant added to avoid zero weights
        buffer.update_weights(experiences.indices, rb_weights)

        self.iterations += 1
        if self.iterations % args.target_network_frequency == 0:
            self.hard_update()

        return total_loss, q_loss, i_loss


def learn(args, env, rb, writer):
    for global_step in range(args.num_gradient_steps):
        total_loss, q_loss, i_loss = agent.learn(args, rb)

        if global_step % 100 == 0:
            agent.Q.eval()
            reward = evaluate(env, 1, agent).mean()
            print(f"global_step={global_step}, episodic_return={reward}")
            writer.add_scalar("learning/episodic_return", reward, global_step)
            agent.Q.train()

            writer.add_scalar("learning/total_loss", total_loss, global_step)
            writer.add_scalar("learning/q_loss", q_loss, global_step)
            writer.add_scalar("learning/i_loss", i_loss, global_step)


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
    wtr.add_text("hyperparameters",
                 "|param|value|\n|-|-|\n%s" % (
                     "\n".join([f"|{key}|{value}|" for key, value in vars(arguments).items()])))

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.backends.cudnn.deterministic = arguments.torch_deterministic
    grid_dimension, grid_size = [int(x) for x in arguments.data_load_path.split('/')[1].split('-', 2)[:2]]
    d = torch.device("cuda" if torch.cuda.is_available() and arguments.cuda else "cpu")

    # terminal_states = None if num_pits == 0 else terminal_states_dict.get(grid_dimension)
    environment = GridWorld(grid_dimension=grid_dimension, grid_size=grid_size, num_total_pits=num_pits,
                            num_clusters=1, distribute_pits_evenly=True, max_steps_per_episode=100,
                            small_state=arguments.small_state, randomize_initial_state=arguments.randomize_initial_state)

    state_size = environment.grid_dimension if arguments.small_state else len(environment.terminal_states)+environment.grid_dimension
    agent = BCQf(state_dim=state_size,
                 action_dim=2*environment.grid_dimension,
                 env=environment,
                 device=d,
                 target_update_frequency=arguments.target_network_frequency,
                 discount=arguments.gamma,
                 lr=arguments.learning_rate,
                 num_layers=arguments.num_network_layers,
                 decay_rate=arguments.lr_decay_rate)

    all_transitions = load_transitions(arguments.data_load_path)
    print(f'Loaded {len(all_transitions)} transitions')

    num_possible_actions = 2 ** (2 * grid_dimension) - 1
    replay_buffer = PrioritizedReplayBuffer(
        buffer_size=len(all_transitions),
        alpha=1,
        observation_space=spaces.MultiDiscrete([grid_size] * state_size),
        action_space=spaces.Discrete(num_possible_actions),
        device=d
    )

    for t in all_transitions:
        # obs, next_obs, action, next_action, reward, termination, info
        replay_buffer.add(t[0][:state_size], t[1][:state_size], t[2], t[3], t[4], t[5])
        # replay_buffer.add(t[0], t[1], t[2], t[3], t[4], t[5])

    learn(arguments, environment, replay_buffer, wtr)

    wtr.close()
