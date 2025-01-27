from collections import defaultdict
import gymnasium as gym
import numpy as np
import random 


class BalatroAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: dict) -> int:
        """
        Returns the best action with probability (1 - epsilon),
        otherwise a random action with probability epsilon to ensure exploration.
        Ensures the selected action is always in the set of valid actions.
        """

        obs_key = self._convert_obs_to_key(obs)

        valid_actions = self.env.valid_actions()

        # With probability epsilon, return a random valid action to explore
        if np.random.random() < self.epsilon:
            return random.choice(valid_actions)

        # With probability (1 - epsilon), act greedily (exploit)
        else:
            if obs_key not in self.q_values:
                return random.choice(valid_actions)

            q_values = self.q_values[obs_key]
            best_action = max(valid_actions, key=lambda action: q_values[action])

            return best_action



    def update(
            self,
            obs: dict,
            action: int,
            reward: float,
            terminated: bool,
            next_obs: dict,
        ):
        """Updates the Q-value of an action using the observation space."""
        obs_key = self._convert_obs_to_key(obs)
        next_obs_key = self._convert_obs_to_key(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_obs_key])

        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs_key][action]
        )

        self.q_values[obs_key][action] += self.lr * temporal_difference

        self.training_error.append(temporal_difference)

    def _convert_obs_to_key(self, obs: dict) -> tuple:
        """
        Converts the observation dictionary into a tuple to use as a key for Q-values.
        Handles NumPy arrays, lists, and scalar values appropriately.
        """
        def safe_tuple(value):
            if isinstance(value, np.ndarray): 
                return tuple(value.tolist())
            elif isinstance(value, list): 
                return tuple(value)
            elif isinstance(value, (int, float)):
                return value
            else:
                raise ValueError(f"Unsupported data type for observation value: {type(value)}")

        try:
            return (
                safe_tuple(obs["deck"]["cards"]),         
                safe_tuple(obs["deck"]["cards_played"]),  
                safe_tuple(obs["hand"]),                 
                safe_tuple(obs["highlighted"]),          
                obs["round_score"],                      
                safe_tuple(obs["round_hands"]),          
                safe_tuple(obs["round_discards"]),      
            )
        except KeyError as e:
            raise KeyError(f"Missing expected key in observation: {e}")
        except ValueError as e:
            raise ValueError(f"Error processing observation: {e}")

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)