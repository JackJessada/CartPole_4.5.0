from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class Double_Q_Learning(BaseAlgorithm):
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
        Initialize the Double Q-Learning algorithm.

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
            control_type=ControlType.DOUBLE_Q_LEARNING,
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
        next_state: tuple,
        done: bool
    ):
        """
        Update Q-values using Double Q-Learning.

        This method applies the Double Q-Learning update rule to improve policy decisions by updating the Q-table.
        """
        if np.random.rand() < 0.5:
            current_qa = self.qa_values[state][action]
            
            if done:
                target = reward
            else:
                
                best_next_action_from_qa = np.argmax(self.qa_values[next_state])
                target = reward + (self.discount_factor * self.qb_values[next_state][best_next_action_from_qa])
                
            self.qa_values[state][action] = current_qa + self.lr * (target - current_qa)
            
        else:
            current_qb = self.qb_values[state][action]
            
            if done:
                target = reward
            else:
                best_next_action_from_qb = np.argmax(self.qb_values[next_state])
                target = reward + (self.discount_factor * self.qa_values[next_state][best_next_action_from_qb])
                
            self.qb_values[state][action] = current_qb + self.lr * (target - current_qb)

        # in RL_BASE get_discretize_action() return q_values so need to avg it.
        self.q_values[state][action] = (self.qa_values[state][action] + self.qb_values[state][action]) / 2.0