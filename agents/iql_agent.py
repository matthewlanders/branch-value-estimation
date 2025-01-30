import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro
import wandb
import random
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import torch.nn.functional as F

from gymnasium.vector.utils import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from agents.common import load_transitions
from agents.evaluate.iql import evaluate
from grid_world import GridWorld


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

    num_gradient_steps: int = 20000
    """total number of gradients steps to perform"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_network_layers: int = 2
    alpha: float = 0.005
    """weight of new data in the exponential moving average"""
    tau: float = 0.9
    """expectile"""
    beta: float = 3.0
    """scales advantages before applying an exponential function"""
    gamma: float = 0.9
    """the discount factor gamma"""
    small_state: bool = False

    finetune: bool = False
    total_timesteps: int = 5000
    learning_starts: int = 100
    buffer_size: int = 5000


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, num_layers=6, hidden_size=32):
        super(Actor, self).__init__()
        layers = [
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])

        layers.append(nn.Linear(hidden_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

    def evaluate(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist

    def get_action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, num_layers=6, hidden_size=32):
        super(Critic, self).__init__()

        layers = [
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])

        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x, a):
        x = torch.cat((x, a), dim=1)
        return self.network(x)


class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, state_size, num_layers=6, hidden_size=32):
        super(Value, self).__init__()
        layers = [
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])

        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class IQL(nn.Module):
    def __init__(self,
                 state_size,
                 action_space_size,
                 single_action_size,
                 env,
                 device,
                 num_layers
                 ):
        super(IQL, self).__init__()
        self.state_size = state_size
        self.action_size = action_space_size
        self.single_action_size = single_action_size
        self.env = env

        self.device = device

        self.gamma = torch.FloatTensor([0.99]).to(device)
        self.hard_update_every = 10
        hidden_size = 256
        learning_rate = 3e-4
        self.clip_grad_param = 100
        self.temperature = torch.FloatTensor([100]).to(device)
        self.expectile = torch.FloatTensor([0.8]).to(device)

        # Actor Network
        self.actor_local = Actor(state_size, action_space_size, num_layers, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, single_action_size, num_layers, hidden_size).to(device)
        self.critic2 = Critic(state_size, single_action_size, num_layers, hidden_size).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(state_size, single_action_size, num_layers, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, single_action_size, num_layers, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.value_net = Value(state_size=state_size, num_layers=num_layers, hidden_size=hidden_size).to(device)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.step = 0

    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, actions):
        with torch.no_grad():
            full_actions = np.array([self.env.compute_action_from_index(a.cpu().numpy()) for a in actions])
            full_actions = torch.from_numpy(full_actions).to(self.device)
            v = self.value_net(states)
            q1 = self.critic1_target(states, full_actions)
            q2 = self.critic2_target(states, full_actions)
            min_Q = torch.min(q1, q2)

        exp_a = torch.exp((min_Q - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(states.device)).squeeze(-1)

        _, dist = self.actor_local.evaluate(states)
        log_probs = dist.log_prob(actions.squeeze(-1))
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss

    def loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def calc_value_loss(self, states, actions):
        with torch.no_grad():
            actions = np.array([self.env.compute_action_from_index(a.cpu().numpy()) for a in actions])
            actions = torch.from_numpy(actions).to(self.device)
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1, q2)

        value = self.value_net(states)
        value_loss = self.loss(min_Q - value, self.expectile).mean()
        return value_loss

    def calc_q_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.gamma * (1 - dones) * next_v)

            actions = np.array([self.env.compute_action_from_index(a.cpu().numpy()) for a in actions])
            actions = torch.from_numpy(actions).to(self.device)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = ((q1 - q_target) ** 2).mean()
        critic2_loss = ((q2 - q_target) ** 2).mean()
        return critic1_loss, critic2_loss

    def learn(self, experiences):
        self.step += 1
        states = experiences.observations.to(torch.float)
        actions = experiences.actions
        next_states = experiences.next_observations.to(torch.float)
        rewards = experiences.rewards.to(torch.float)
        dones = experiences.dones

        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(states, actions)
        value_loss.backward()
        self.value_optimizer.step()

        actor_loss = self.calc_policy_loss(states, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)

        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        if self.step % self.hard_update_every == 0:
            # ----------------------- update target networks ----------------------- #
            self.hard_update(self.critic1, self.critic1_target)
            self.hard_update(self.critic2, self.critic2_target)

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), value_loss.item()

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


def learn(args, env, rb, writer):
    for global_step in range(args.num_gradient_steps):
        a_loss, c1_loss, c2_loss, v_loss = agent.learn(rb.sample(args.batch_size))

        if global_step % 100 == 0:
            agent.eval()
            reward = evaluate(env, 1, agent).mean()
            print(f"global_step={global_step}, episodic_return={reward}")
            writer.add_scalar("learning/episodic_return", reward, global_step)
            agent.train()

            writer.add_scalar("learning/actor_loss", a_loss, global_step)
            writer.add_scalar("learning/critic_1_loss", c1_loss, global_step)
            writer.add_scalar("learning/critic_2_loss", c2_loss, global_step)
            writer.add_scalar("learning/value_loss", v_loss, global_step)


def learn_ft(args, env, rb, writer):
    obs = env.reset()

    for global_step in range(args.total_timesteps):
        action = agent.get_action(obs)
        next_obs, rewards, terminations, info = env.step(action)
        real_next_obs = next_obs.copy()

        if "final_info" in info:
            data = info['final_info']['episode']
            print(f"global_step={global_step}, episodic_return={data['r']}")
            writer.add_scalar("learning/episodic_return", data["r"], global_step)
            writer.add_scalar("charts/episodic_length", data["l"], global_step)
            real_next_obs = data['final_observation']

        rb.add(obs, real_next_obs, action, rewards, terminations, info)

        obs = next_obs

        if global_step > args.learning_starts:
            a_loss, c1_loss, c2_loss, v_loss = agent.learn(rb.sample(args.batch_size))

            if global_step % 100 == 0:
                writer.add_scalar("learning/actor_loss", a_loss, global_step)
                writer.add_scalar("learning/critic_1_loss", c1_loss, global_step)
                writer.add_scalar("learning/critic_2_loss", c2_loss, global_step)
                writer.add_scalar("learning/value_loss", v_loss, global_step)


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
    d = torch.device("cuda" if torch.cuda.is_available() and arguments.cuda else "cpu")

    environment = GridWorld(grid_dimension=grid_dimension, grid_size=grid_size, num_total_pits=num_pits,
                            num_clusters=1, distribute_pits_evenly=True, max_steps_per_episode=100,
                            small_state=arguments.small_state, randomize_initial_state=arguments.randomize_initial_state)

    state_size = environment.grid_dimension if arguments.small_state else len(environment.terminal_states)+environment.grid_dimension
    action_size = 2 ** (2 * environment.grid_dimension)
    agent = IQL(state_size=state_size, action_space_size=action_size, single_action_size=2*environment.grid_dimension,
                env=environment, device=d, num_layers=arguments.num_network_layers)

    all_transitions = load_transitions(arguments.data_load_path)
    print(f'Loaded {len(all_transitions)} transitions')

    num_possible_actions = 2 ** (2 * grid_dimension) - 1
    replay_buffer = ReplayBuffer(
        len(all_transitions),
        spaces.MultiDiscrete([grid_size] * state_size),
        spaces.Discrete(num_possible_actions),
        d,
        handle_timeout_termination=False,
    )

    for t in all_transitions:
        # obs, real_next_obs, action, next_action, reward, termination, info
        replay_buffer.add(t[0][:state_size], t[1][:state_size], t[2], t[4], t[5], t[6])

    learn(arguments, environment, replay_buffer, wtr)

    if arguments.finetune:
        print('Beginning finetuning')
        replay_buffer = ReplayBuffer(
            arguments.buffer_size,
            spaces.MultiDiscrete([grid_size] * state_size),
            spaces.Discrete(num_possible_actions),
            d,
            handle_timeout_termination=False,
        )
        learn_ft(arguments, environment, replay_buffer, wtr)

    wtr.close()
