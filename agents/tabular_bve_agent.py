import random
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from grid_world import GridWorld


def linear_schedule(start, end, rate, step):
    return max(end, start - step * (start - end) / rate)


def compute_children(env, action):
    action_shape = (2,) * 2 * env.grid_dimension
    current_action = env.compute_action_from_index(action)
    last_activated_sub_action = np.max(np.where(current_action == 1)[0], initial=-1)

    children = []
    for i in range(last_activated_sub_action + 1, len(current_action)):
        child_action = np.copy(current_action)
        child_action[i] = 1
        child_index = np.ravel_multi_index(tuple(child_action), action_shape)
        children.append(child_index)

    return children


def get_best_action(env, bve_table, obs):
    action = 0

    while True:
        max_value = bve_table[tuple(obs)][action][0]  # TODO is this right or should be max over bve_table[tuple(obs)][action]?
        children = compute_children(env, action)
        if not children:
            return action

        actions = [action] + children
        for i, a in enumerate(actions):
            bve = bve_table[tuple(obs)][action][i]

            if bve > max_value:
                action = a
                max_value = bve

        if actions.index(action) == 0:
            return action


def compute_action_branch(env, idx):
    action_shape = (2,) * 2 * env.grid_dimension
    current_action = env.compute_action_from_index(idx)
    branch = []

    last_action = current_action.copy()
    for i in range(len(last_action) - 1, -1, -1):
        if last_action[i] == 1:
            last_action[i] = 0
            parent_index = np.ravel_multi_index(tuple(last_action), action_shape)
            branch.append(parent_index)

    return branch


def update_bves(env, bve_table, obs, action):
    branch = compute_action_branch(env, action)
    target = max(bve_table[tuple(obs)][action])

    for a in branch:
        children = compute_children(env, a)
        idx = children.index(action)
        bve_table[tuple(obs)][a][idx+1] = target
        max_bve = max(bve_table[tuple(obs)][a])
        target = max(bve_table[tuple(obs)][a][idx], max_bve)
        action = a


def learn(env, start_e, end_e, exploration_rate, total_timesteps, gamma, alpha):
    bve_table = defaultdict(lambda: defaultdict(lambda: [0]*(2*env.grid_dimension+1)))
    obs = env.reset()
    episode_rewards = []

    for global_step in range(total_timesteps):
        ep = linear_schedule(start_e, end_e, exploration_rate, global_step)
        action = np.random.randint(env.total_actions) if np.random.rand() < ep else get_best_action(env, bve_table, obs)

        next_obs, reward, done, info = env.step(action)

        if "final_info" in info:
            returns = info['final_info']['episode']['r']
            print(f"global_step={global_step}, episodic_return={returns}")
            episode_rewards.append(returns)

        old_q_value = bve_table[tuple(obs)][action][0]
        next_max_q = max((bves[0] for bves in bve_table[tuple(next_obs)].values()), default=0)
        td_target = reward + gamma * next_max_q * (1 - int(done))
        td_error = td_target - old_q_value

        new_q_value = old_q_value + alpha * td_error
        bve_table[tuple(obs)][action][0] = new_q_value

        update_bves(env, bve_table, obs, action)

        obs = next_obs

    return episode_rewards, bve_table


if __name__ == "__main__":
    # terminal_states = [(4, 4), (1, 1), (2, 1), (3, 3)]
    terminal_states = [(1, 1)]
    environment = GridWorld(grid_dimension=2, grid_size=2, num_total_pits=0, num_clusters=0,
                            distribute_pits_evenly=True, only_independent=False, max_steps_per_episode=1000,
                            terminal_states=terminal_states, randomize_initial_state=True)

    reward_per_episode, branch_value_estimates = learn(env=environment, start_e=1, end_e=0.05, exploration_rate=0.5,
                                                       total_timesteps=50000, gamma=0.9, alpha=0.9)
    plt.plot(reward_per_episode)
    plt.show()
