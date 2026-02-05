import torch
import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
import pandas as pd

import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)

def create_model(args, env):
    # T4 TODO
    if args.algo == 'ppo':        
        model = PPO("MlpPolicy", env, learning_rate=args.lr, seed=args.seed, verbose=1)
    
    elif args.algo == 'sac':
        model = SAC("MlpPolicy", env, learning_rate=args.lr, seed=args.seed, verbose=1, ent_coef='auto_0.1')
    
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model
        
def load_model(args, env, path):
    # T4 TODO
    if args.algo == 'ppo':
        model = PPO.load(path, env=env)

    elif args.algo == 'sac':
        model = SAC.load(path, env=env)
    
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve", args = None):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    x = x[len(x) - len(y) :]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig("./figures/model_%s_%s_%d.png" % (args.algo, args.lr, args.total_timesteps))
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=25000, help="The total number of samples to train on")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--algo', default='ppo', type=str, help='RL Algo [ppo, sac]')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--gradient_steps', default=-1, type=int, help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
    parser.add_argument('--test_episodes', default=100, type=int, help='# episodes for test evaluations')
    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make(args.env)
    log_dir = f"./tmp/gym/{args.algo}"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    if args.test is None:
        try:
            model = create_model(args, env)

            eval_callback = EvalCallback(
            env, 
            best_model_save_path=f'./checkpoints/{args.algo}/',
            log_path='./tmp/gym/', 
            eval_freq=1000,
            deterministic=True, 
            render=False)

            # Policy training (T4) TODO
            model.learn(args.total_timesteps, callback= eval_callback)
            # Saving model (T4) TODO
            save_path = os.path.join("checkpoints", f"model_{args.env}_{args.algo}_{args.total_timesteps}.zip")
            os.makedirs("checkpoints", exist_ok=True)
            model.save(save_path)
            print(f"Model saved to {save_path}")
            plot_results(log_dir, args= args)
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        print("Testing...")
        model = load_model(args, env, args.test)
        # Policy evaluation (T4) TODO
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.test_episodes, render=args.render_test)
        print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")

    env.close()