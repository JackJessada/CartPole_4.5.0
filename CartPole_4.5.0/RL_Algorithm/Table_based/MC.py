from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        done: bool
    ):
        """
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        """
        # store in 
        self.obs_hist.append(state)
        self.action_hist.append(action)
        self.reward_hist.append(reward)

        if done:
            G = 0 # G = R + (G_next * discount)
            for t in range(len(self.reward_hist)-1, -1 , -1):
                s_t = self.obs_hist[t]
                a_t = self.action_hist[t]
                r_t = self.reward_hist[t]

                G = r_t + (self.discount_factor * G)
                # save freq of stage, action
                self.n_values[s_t][a_t] += 1
                # # Q(S,A) <-- Q(S,A) + lr * [G - Q(S,A)]
                self.q_values[s_t][a_t] += self.lr * (G - self.q_values[s_t][a_t])
            self.obs_hist.clear()
            self.action_hist.clear()
            self.reward_hist.clear()