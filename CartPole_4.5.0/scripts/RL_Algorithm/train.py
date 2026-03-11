"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.RL_base import ControlType
from tqdm import tqdm

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
parser.add_argument("--num_episodes", default=2000, type=int)
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
import random

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

from isaaclab.utils.dict import print_dict
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Hyperparameters
    num_of_action = 3                      
    action_range = [-3.0,3.0]           
    discretize_state_weight = [1, 20, 1, 5]#[pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.1
    n_episodes = args_cli.num_episodes                    
    start_epsilon = 1.0
    epsilon_decay = 0.995 #not use for now                  # decay rate. (original_e * decay rate)
    final_epsilon = 0.01
    discount = 0.9                        # gamma
    warmup_rario = 0.2
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
    tb_log_dir = os.path.join(log_dir, "tensorboard", algo_enum.name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    timestep = 0
    sum_reward = 0
    max_abs_cart_pos = 0.0
    max_abs_pole_pos = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
        
            for episode in tqdm(range(n_episodes)):
                obs, _ = env.reset()
                done = False
                cumulative_reward = 0
                
                action_tensor, action_idx = agent.get_action(obs)

                ####DEBUG
                sub_steps = 0
                while not done:
                    # env stepping
                    state_array = obs["policy"].squeeze().cpu().numpy()
                    cart_pos = abs(state_array[0])
                    pole_pos = abs(state_array[1]) 
                    
                    max_abs_cart_pos = max(max_abs_cart_pos, float(cart_pos))
                    max_abs_pole_pos = max(max_abs_pole_pos, float(pole_pos))
                    action_tensor_env = action_tensor.reshape(1, 1)
                    next_obs, reward, terminated, truncated, _ = env.step(action_tensor_env)

                    reward_value = reward.item()
                    terminated_value = terminated.item()
                    truncated_value = truncated.item()
                    done = terminated_value or truncated_value
                    # print(f"done {done}, terminate {terminated_value}, truncate {truncated_value}")                   
                    cumulative_reward += reward_value

                    # discretize 
                    state_dis = agent.discretize_state(obs)
                    next_state_dis = agent.discretize_state(next_obs)

                
                    if algo_enum == ControlType.SARSA:
                        next_action_tensor, next_action_idx = agent.get_action(next_obs)
                        agent.update(state_dis, action_idx, reward_value, next_state_dis, next_action_idx, done)
                        action_tensor, action_idx = next_action_tensor, next_action_idx
                    
                    elif algo_enum == ControlType.MONTE_CARLO:
                        agent.update(state_dis, action_idx, reward_value, done)
                        if not done:
                            action_tensor, action_idx = agent.get_action(next_obs)
                            
                    else:
                        # Q_Learning and Double_Q_Learning
                        agent.update(state_dis, action_idx, reward_value, next_state_dis, done)
                        if not done:
                            action_tensor, action_idx = agent.get_action(next_obs)
                    sub_steps+=1
                    # print(f"Ep: {episode}, substep: {sub_steps}")
                    obs = next_obs
                   
                writer.add_scalar("Train/Episode_Reward", cumulative_reward, episode)
                writer.add_scalar("Train/Steps_per_Episode", sub_steps, episode)
                writer.add_scalar("Metrics/Max_Abs_Cart_Pos", max_abs_cart_pos, episode)
                writer.add_scalar("Metrics/Max_Abs_Pole_Pos", max_abs_pole_pos, episode)
                sum_reward += cumulative_reward
                if episode % 100 == 0:
                    avg_score = sum_reward / 100.0
                    print(f"avg_score: {avg_score:.2f} | Epsilon: {agent.epsilon:.4f}")
                    sum_reward = 0
                    writer.add_scalar("Train/Average_Reward_100_Eps", avg_score, episode)
                    writer.add_scalar("Parameters/Epsilon", agent.epsilon, episode)
                    q_value_file = f"{algo_enum.name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                    full_path = os.path.join(f"q_value/{task_name}", algo_enum.name)
                    os.makedirs(full_path, exist_ok=True)
                    
                    agent.save_q_value(full_path, q_value_file)

                agent.decay_epsilon(episode, n_episodes, warmup_rario)
             
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        writer.close()
        print("!!! Training is complete !!!")
        break
    # ==================================================================== #
    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()