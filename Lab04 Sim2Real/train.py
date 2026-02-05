import gymnasium as gym
import argparse
import yaml
import os
import numpy as np

from env.custom_hopper import *  
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy


class HopperUDRWrapper(gym.Wrapper):
    def __init__(self, env, thigh_range, leg_range, foot_range, torso_mass=None):
        super().__init__(env)
        self.thigh_range = thigh_range
        self.leg_range = leg_range
        self.foot_range = foot_range
        self.torso_mass = torso_mass if torso_mass is not None else self.unwrapped.model.body_mass[0]

    def reset(self, **kwargs):
        self.unwrapped.model.body_mass[0] = self.torso_mass  # torso fisso
        self.unwrapped.model.body_mass[1] = np.random.uniform(*self.thigh_range)  # thigh
        self.unwrapped.model.body_mass[2] = np.random.uniform(*self.leg_range)    # leg
        self.unwrapped.model.body_mass[3] = np.random.uniform(*self.foot_range)   # foot
        return self.env.reset(**kwargs)


def main(args):

    train_env_name = "CustomHopper-source-v0" if args.train_env == "source" else "CustomHopper-target-v0"
    test_env_name = "CustomHopper-source-v0" if args.test_env == "source" else "CustomHopper-target-v0"

    temp_env = gym.make(train_env_name)
    original_masses = temp_env.unwrapped.model.body_mass.copy()
    torso_mass = original_masses[0]
    temp_env.close()

    # Training env
    if args.udr:
        env_train = HopperUDRWrapper(
            gym.make(train_env_name),
            thigh_range=(original_masses[1]*0.8, original_masses[1]*1.2),
            leg_range=(original_masses[2]*0.8, original_masses[2]*1.2),
            foot_range=(original_masses[3]*0.8, original_masses[3]*1.2),
            torso_mass=torso_mass
        )
    else:
        env_train = gym.make(train_env_name)

    # Test env
    env_test = gym.make(test_env_name, render_mode="human" if args.render_test else None)

    print("State space:", env_train.observation_space)
    print("Action space:", env_train.action_space)
    print("Original dynamics parameters:", original_masses)

    os.makedirs(f"./checkpoints/{args.model}/", exist_ok=True)

    # Config
    config_kwargs = {}
    if args.config:
        with open("config.yaml", "r") as f:
            all_configs = yaml.safe_load(f)
        config_kwargs = all_configs[args.config]

    # Checkpoint path
    checkpoint_path = f"./checkpoints/{args.model}/best_model_{args.train_env}"
    if args.config:
        checkpoint_path += f"_{args.config}"
    if args.udr:
        checkpoint_path += "_udr"


    if not args.test:
        if args.model == "PPO":
            ppo_kwargs = {}
            if "learning_rate" in config_kwargs:
                ppo_kwargs["learning_rate"] = float(config_kwargs["learning_rate"])
            if "batch_size" in config_kwargs:
                ppo_kwargs["batch_size"] = int(config_kwargs["batch_size"])
            if "gamma" in config_kwargs:
                ppo_kwargs["gamma"] = float(config_kwargs["gamma"])
            model = PPO("MlpPolicy", env_train, verbose=1, seed=21, **ppo_kwargs)

        elif args.model == "SAC":
            sac_kwargs = {}
            if "learning_rate" in config_kwargs:
                sac_kwargs["learning_rate"] = float(config_kwargs["learning_rate"])
            if "gamma" in config_kwargs:
                sac_kwargs["gamma"] = float(config_kwargs["gamma"])
            model = SAC("MlpPolicy", env_train, verbose=1, seed=21, **sac_kwargs)

        model.learn(total_timesteps=args.train_episodes)
        model.save(checkpoint_path)
        print(f"Model saved in {checkpoint_path}")


    else:
        if args.model == "PPO":
            model = PPO.load(checkpoint_path + ".zip")
        elif args.model == "SAC":
            model = SAC.load(checkpoint_path + ".zip")

        mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=50, deterministic=True, render=args.render_test)
        print(f"For the {args.model} trained on {args.train_env} "
              f"and tested on {args.test_env}: {mean_reward:.2f} mean and {std_reward:.2f} std.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test")
    parser.add_argument("--train_episodes", type=int, default=200000, help="Number of timesteps to train for")
    parser.add_argument("--render_test", action="store_true", help="Render test")
    parser.add_argument("--train_env", choices=["source", "target"], required=True)
    parser.add_argument("--test_env", choices=["source", "target"], required=True)
    parser.add_argument("--model", choices=["SAC", "PPO"], required=True)
    parser.add_argument("--config", type=str, choices=["config1", "config2", "config3"], help="Hyperparameter config")
    parser.add_argument("--udr", action="store_true", help="Enable Uniform Domain Randomization")

    args = parser.parse_args()
    main(args)
