import heapq
import itertools
import os
import pickle
import random
from dataclasses import dataclass

import numpy as np
import tyro

from grid_world import GridWorld


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    save_transitions: bool = False
    """if toggled, transitioned will be saved for offline use"""
    only_find_optimal_path: bool = False
    """if toggled, don't generate transitions, just print optimal path for world"""
    suboptimal_rate: float = 0.9
    """percentage of the time random, suboptimal actions are taken"""
    num_transitions: int = 10000
    """number of transitions to store in dataset"""
    randomize_initial_state: bool = False
    """if toggled, agent will start in random (non-terminal) grid location"""
    grid_dimension: int = 2
    """number of grid dimensions"""
    grid_size: int = 2
    """size of grid"""
    num_pits: int = 0
    """number of pits in gridworld"""
    num_clusters: int = 0
    """number of pit clusters in gridworld"""
    batch_size: int = 10000
    small_state: bool = False
    save_terminal_states: bool = False
    load_terminal_states: bool = False


def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_neighbors(node, grid_size, grid_dimension):
    neighbors = []
    for delta in itertools.product([-1, 0, 1], repeat=grid_dimension):
        if all(d == 0 for d in delta):  # Skip the current node itself
            continue
        neighbor = tuple(np.clip(np.array(node) + np.array(delta), 0, grid_size - 1))
        neighbors.append(neighbor)
    return neighbors


def a_star(env, start, goal, find_optimal_path):
    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))
    pits = set(tuple(s) for s in env.terminal_states[1:])

    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): heuristic(start, goal)}

    while open_set:
        current = tuple(heapq.heappop(open_set)[1])

        if np.array_equal(current, goal):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current, env.grid_size, env.grid_dimension):
            if neighbor in pits and find_optimal_path:
                continue

            tentative_g_score = g_score[current] + heuristic(neighbor, goal)
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))


def compute_action_idx_for_step(env, current_location, next_step):
    action = [0] * (2 * env.grid_dimension)
    for i in range(env.grid_dimension):
        if next_step[i] > current_location[i]:
            action[2 * i] = 1
        elif next_step[i] < current_location[i]:
            action[2 * i + 1] = 1

    action_shape = (2,) * 2 * env.grid_dimension
    return np.ravel_multi_index(tuple(action), action_shape)


def select_action(env, obs, suboptimal_rate, find_optimal_path):
    goal = env.terminal_states[0]

    if random.random() < suboptimal_rate:
        neighbors = get_neighbors(obs[:env.grid_dimension], env.grid_size, env.grid_dimension)
        next_obs = random.choice(neighbors)
    else:
        next_obs = a_star(env, obs[:env.grid_dimension], goal, find_optimal_path)[1]

    return compute_action_idx_for_step(env, obs[:env.grid_dimension], next_obs)


def learn(env, suboptimal_rate, num_transitions, find_optimal_path, batch_size=10000, path=None):
    if save_path and not find_optimal_path:
        os.makedirs(save_path, exist_ok=True)

    transitions = []
    obs = env.reset()
    transition_count = 0
    action = select_action(env, obs, suboptimal_rate, find_optimal_path)

    while transition_count < num_transitions:
        next_obs, reward, termination, info = env.step(action)
        real_next_obs = next_obs.copy()

        if "final_info" in info:
            data = info['final_info']['episode']
            real_next_obs = data['final_observation']
            if find_optimal_path:
                transitions.append((obs, real_next_obs, action, None, reward, termination, info))
                print(f"episodic_return={info['final_info']['episode']['r']}")
                actions = [t[2] for t in transitions]
                print(actions)
                states = [t[0][:5] for t in transitions]
                print(states)
                break

        next_action = select_action(env, next_obs, suboptimal_rate, find_optimal_path)
        transitions.append((obs, real_next_obs, action, next_action, reward, termination, info))

        obs = next_obs
        action = next_action
        transition_count += 1
        print(f'Generated transitions {transition_count}')

        if len(transitions) >= batch_size and path is not None:
            batch_file = os.path.join(path, f"transitions_batch_{transition_count // batch_size}.pkl")
            with open(batch_file, 'wb') as f:
                pickle.dump(transitions, f)
            transitions.clear()

    if transitions and save_path and not find_optimal_path:
        batch_file = os.path.join(path, f"transitions_batch_{(transition_count // batch_size) + 1}.pkl")
        with open(batch_file, 'wb') as f:
            pickle.dump(transitions, f)


if __name__ == "__main__":
    args = tyro.cli(Args)
    environment = GridWorld(grid_dimension=args.grid_dimension, grid_size=args.grid_size, num_total_pits=args.num_pits,
                            num_clusters=args.num_clusters, distribute_pits_evenly=True, max_steps_per_episode=1000,
                            save_terminal_states=args.save_terminal_states, small_state=args.small_state,
                            load_terminal_states=args.load_terminal_states,
                            randomize_initial_state=args.randomize_initial_state)

    run_directory = f"{args.grid_dimension}-{args.grid_size}-{args.num_pits}-{args.suboptimal_rate}"
    save_path = f"offline_data/{run_directory}" if args.save_transitions else None

    if not args.only_find_optimal_path:
        learn(env=environment, suboptimal_rate=float(int(args.suboptimal_rate)/100),
              num_transitions=args.num_transitions, find_optimal_path=False, batch_size=args.batch_size, path=save_path)

    # Find and print optimal path
    learn(env=environment, suboptimal_rate=0, num_transitions=args.num_transitions, find_optimal_path=True)
