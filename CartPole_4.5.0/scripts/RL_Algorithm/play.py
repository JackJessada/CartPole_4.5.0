"""Script to play RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm
from RL_Algorithm.RL_base import ControlType
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--algorithm", 
    type=int, 
    choices=[e.value for e in ControlType], 
    default=ControlType.Q_LEARNING.value, 
    help="Select Algorithm: 1=MC, 2=TEMPORAL_DIFFERENCE (SARSA), 3=Q_LEARNING, 4=DOUBLE_Q_LEARNING"
)
parser.add_argument("--load_episode", type=int, default=1900, help="Episode number to load")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import numpy as np
import matplotlib.pyplot as plt

def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,  #use_fabric=not args_cli.disable_fabric
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    num_of_action = 5
    action_range = [-10.0, 10.0]  
    discretize_state_weight = [10, 10, 10, 10]
    learning_rate = 0.1
    n_episodes = 5 
    
    # no decay, no epsilon
    start_epsilon = 0.0
    epsilon_decay = 0.0  
    final_epsilon = 0.0
    discount = 0.99

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    algo_enum = ControlType(args_cli.algorithm)
    
    if algo_enum == ControlType.MONTE_CARLO:
        from RL_Algorithm.Table_based.MC import MC as AgentClass
    elif algo_enum == ControlType.SARSA:
        from RL_Algorithm.Table_based.SARSA import SARSA as AgentClass
    elif algo_enum == ControlType.DOUBLE_Q_LEARNING:
        from RL_Algorithm.Table_based.Double_Q_Learning import Double_Q_Learning as AgentClass
    else:
        from RL_Algorithm.Table_based.Q_Learning import Q_Learning as AgentClass

    agent = AgentClass(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount
    )

    episode_to_load = args_cli.load_episode 
    q_value_file = f"{algo_enum.name}_{episode_to_load}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
    full_path = os.path.join(f"q_value/{task_name}", algo_enum.name)
    

    
    try:
        agent.load_q_value(full_path, q_value_file)
        print("[INFO] Agent successful loading")
    except FileNotFoundError:
        print(f"[ERROR] not found {q_value_file}")
        env.close()
        simulation_app.close()
        sys.exit(1)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
        
            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                cumulative_reward = 0

                while not done:
                    action_tensor, action_idx = agent.get_action(obs)

                    # env stepping
                    action_tensor_env = action_tensor.reshape(1, 1)
                    next_obs, reward, terminated, truncated, _ = env.step(action_tensor_env)

                    cumulative_reward += reward.item()
                    done = terminated.item() or truncated.item()
                    obs = next_obs
                
                print(f"Episode {episode+1}/{n_episodes} เล่นจบแล้ว! | Total Reward (คะแนนตานี้): {cumulative_reward:.2f}")

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
                
        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()