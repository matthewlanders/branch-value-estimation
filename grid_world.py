import itertools
import math
import os
import random

import numpy as np


class GridWorld:
    def __init__(self, grid_dimension: int, grid_size: int, num_total_pits: int, num_clusters: int,
                 distribute_pits_evenly: bool, max_steps_per_episode: int,
                 load_terminal_states: bool = False, save_terminal_states: bool = True,
                 randomize_initial_state: bool = False, small_state: bool = False):
        self.grid_dimension = grid_dimension
        self.grid_size = grid_size
        self.num_total_pits = num_total_pits
        self.num_pit_clusters = num_clusters
        self.distribute_pits_evenly = distribute_pits_evenly
        self.max_steps_per_episode = max_steps_per_episode
        self.randomize_initial_state = randomize_initial_state
        self.grid = {}
        self.episode_reward = 0
        self.episode_length = 0
        self.current_location = np.zeros(grid_dimension, dtype=int)
        self.total_actions = 2 ** (2 * grid_dimension)
        self.small_state = small_state

        run_directory = f"{grid_dimension}-{grid_size}-{num_total_pits}-90"
        terminal_states_file = f"offline_data/{run_directory}/terminal_states.npy"

        load_terminal_states = load_terminal_states if num_total_pits > 5 else False

        if load_terminal_states:
            self.terminal_states = self._load_terminal_states(terminal_states_file)
        else:
            self.terminal_states = self._place_goal_and_pits()
            if save_terminal_states:
                self._save_terminal_states(terminal_states_file)

        self._setup_terminal_states()
        print(f'Created gridworld with {len(self.terminal_states)} terminal states')

    def _save_terminal_states(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.terminal_states)

    @staticmethod
    def _load_terminal_states(filepath: str) -> np.ndarray:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Terminal states file not found at {filepath}")
        return np.load(filepath)

    def reset(self):
        self.episode_reward = 0
        self.episode_length = 0

        if self.randomize_initial_state:
            while tuple(self.current_location) in [tuple(state) for state in self.terminal_states]:
                self.current_location = np.random.randint(0, self.grid_size, size=self.grid_dimension)
        else:
            self.current_location = np.zeros(self.grid_dimension, dtype=int)

        if self.small_state:
            return self.current_location
        distances = np.array([np.linalg.norm(self.current_location - ts) for ts in self.terminal_states])
        return np.append(self.current_location, distances)

    def get_cell_value(self, position):
        distance_from_goal = np.linalg.norm(self.current_location - self.terminal_states[0])
        return self.grid.get(tuple(position), -distance_from_goal)

    def _get_pit_starting_points(self, goal):
        factors = (np.linspace(1 / (self.num_pit_clusters + 1), 1, self.num_pit_clusters) if self.distribute_pits_evenly
                   else np.random.choice(np.linspace(0, 1, 100), self.num_pit_clusters, replace=False))

        return [np.array([max(1, min(int(self.current_location[j] + (goal[j] - self.current_location[j]) * factor),
                                     self.grid_size - 2)) for j in range(self.grid_dimension)])
                for factor in factors]

    def _place_goal_and_pit_clusters(self, goal):
        # Precompute the deltas (adjacent position offsets)
        deltas = np.array([delta for delta in itertools.product([-1, 0, 1], repeat=self.grid_dimension)
                           if not all(d == 0 for d in delta)])
        deltas = deltas.astype(int)

        occupied = {tuple(goal)}
        cluster_pits = []
        pit_starting_locations = self._get_pit_starting_points(goal)
        total_pits_remaining = self.num_total_pits

        occupied_positions = set()
        occupied_positions.add(tuple(goal))

        for pit in pit_starting_locations:
            num_pits_in_cluster = math.ceil(total_pits_remaining / len(pit_starting_locations))
            cluster_pits.append(pit)
            occupied_positions.add(tuple(pit))
            total_pits_remaining -= 1

            additional_pits_in_cluster = min(num_pits_in_cluster - 1, total_pits_remaining)
            for _ in range(additional_pits_in_cluster):
                adj_positions = pit + deltas

                # Filter positions that are within bounds (greater than 0 and less than grid_size - 1)
                valid_mask = np.all((adj_positions > 0) & (adj_positions < self.grid_size - 1), axis=1)
                valid_positions = adj_positions[valid_mask]
                valid_positions_tuples = [tuple(pos) for pos in valid_positions]

                # Filter out already occupied positions
                expansion_candidates = [pos for pos, pos_tup in zip(valid_positions, valid_positions_tuples)
                                        if pos_tup not in occupied_positions]

                if not expansion_candidates:
                    break

                # Randomly select a new pit from the candidates
                new_pit = random.choice(expansion_candidates)
                cluster_pits.append(new_pit)
                occupied_positions.add(tuple(new_pit))
                total_pits_remaining -= 1

        return [np.array(goal)] + cluster_pits

    def _place_goal_and_pits(self):
        goal = np.array([self.grid_size - 1] * self.grid_dimension)
        return self._place_goal_and_pit_clusters(goal)

    def _setup_terminal_states(self):
        max_distance_from_goal = np.linalg.norm(self.current_location - self.terminal_states[0])
        for state in self.terminal_states:
            self.grid[tuple(state)] = 10 if np.array_equal(state, self.terminal_states[0]) \
                else max_distance_from_goal * -10

    def compute_action_from_index(self, idx):
        return np.array(np.unravel_index(idx, (2,) * 2 * self.grid_dimension)).reshape(-1)

    def step(self, action):
        action = self.compute_action_from_index(action)
        self.episode_length += 1
        movement = np.zeros(self.grid_dimension, dtype=int)

        for i in range(self.grid_dimension):
            if action[2 * i] == 1 and action[2 * i + 1] == 0:
                movement[i] = 1
            if action[2 * i] == 0 and action[2 * i + 1] == 1:
                movement[i] = -1

        new_location = np.clip(self.current_location + movement, 0, self.grid_size - 1)
        if self.small_state:
            current_obs = new_location
        else:
            distances = np.array([np.linalg.norm(new_location - ts) for ts in self.terminal_states])
            current_obs = np.append(new_location, distances)
        self.current_location = new_location
        reward = self.get_cell_value(new_location)
        self.episode_reward += reward
        in_terminal_state = tuple(new_location) in map(tuple, self.terminal_states)
        is_terminal = in_terminal_state or self.episode_length == self.max_steps_per_episode

        info = {
            'final_info': {
                'episode': {
                    'r': self.episode_reward,
                    'l': self.episode_length,
                    'final_observation': current_obs
                }
            }
        } if is_terminal else {}

        if is_terminal:
            self.reset()

        if self.small_state:
            return self.current_location, reward, in_terminal_state, info
        else:
            distances = np.array([np.linalg.norm(self.current_location - ts) for ts in self.terminal_states])
            return np.append(self.current_location, distances), reward, in_terminal_state, info
