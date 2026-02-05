"""
    Robot Learning
    Exercise 1

    Reinforcement Learning 

    Polito A-Y 2025-2026
"""
import torch
import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim
import os
import sys


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--central_point", type=float, default=0.0,
                        help="Point x0 to fluctuate around")
    parser.add_argument("--random_policy", action='store_true', help="Applying a random policy training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--reward_mode", type=str,
                    choices=["center", "arbitrary", "move"],
                    help="Choose which custom reward to use")

    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False,
          silent=False, train_run_id=0, x0=0, random_policy=False, reward_mode= 'center'):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state (it's a random initial state with small values)
        observation, _ = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            if random_policy:
                # Task 1.1
                """
                Sample a random action from the action space
                """
                action = env.action_space.sample()

            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            # note that after env._max_episode_steps the episode is over, if we stay alive that long
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            if reward_mode is not None:
                reward = new_reward(observation, x0, mode=reward_mode)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                # Gymnasium recommends setting render_mode at env creation
                pass

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False, x0=0, reward_mode= 'center'):
    test_reward, test_len, max_vel = 0, 0, 0

    episodes = 100
    print('Num testing episodes:', episodes)

    for ep in range(episodes):
        done = False
        observation, _ = env.reset()
        timestep = 0
        
        while not done and timestep<500:
        # Task 1.3
            """
            Test on 500 timesteps
            """
            action, _ = agent.get_action(observation, evaluation=True)  # Similar to the training loop above -
                                                                        # get the action, act on the environment, save total reward
                                                                        # (evaluation=True makes the agent always return what it thinks to be
                                                                        # the best action - there is no exploration at this point)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if observation[1] > max_vel:
                max_vel = observation[1]
            
            if render:
                env.render()
            test_reward += reward
            test_len += 1
            timestep += 1
            

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            
            reward = new_reward(observation, x0=x0, mode=reward_mode)

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)
    return max_vel

def new_reward(state, x0, mode):
    # Task 3.1
    """
        Use a different reward, overwriting the original one.
        
        From the documentation we have:
        The episode ends if any one of the following occurs:
            * Termination: Pole Angle is greater than ±12°
            * Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
            * Truncation: Episode length is greater than 500 (200 for v0)
    """
    # ...
    x, x_dot, theta, theta_dot = state

    if mode == "center":
    # Reward highest when cart is near x=0 and pole is upright
        """
        The closer the cart is to the center and the pole is upright, the higher the reward. So we use the abs(x) to penalize the position
        and the abs(theta) to penalize if the pole is not upright. Angle is given in radiants by gym, so we need to converts
        """
        reward = 1.0 - (abs(x)/2.4) - (abs(theta)/0.2095)
        reward = max(reward, 0) 

    elif mode == "arbitrary":
        # Reward highest when cart is near x=x0 and pole is upright
        """
        The same as "center" with the (x-x0) because it has a goal position
        """
        reward = 1.0 - (abs(x-x0)/2.4) - (abs(theta)/0.2095)
        reward = max(reward, 0)

    elif mode == "move":
    
        # Normalized cart position in [0,1]
        norm_x = abs(x) / 2.4

        # Reward for horizontal velocity:
        # slightly reduced so that speed does not dominate over pole balance
        velocity_term = 0.9 * abs(x_dot)

        # Reward for reaching the edges of the track:
        # increased to strongly encourage full-range left→right movement
        position_term = 2.5 * (norm_x ** 2)

        # Reward for changing direction when far from the center:
        # encourages performing intentional left↔right sweeps
        if abs(x) > 0.5:
            # Moving toward the center (negative dot product) = desired redirection
            if (x / 2.4) * x_dot < 0:
                direction_term = 3.5 * norm_x * abs(x_dot)
            # Moving outward while already far from center = mildly penalized
            else:
                direction_term = -0.8 * abs(x_dot)
        else:
            direction_term = 0.0

        # Penalties on pole angle and angular velocity:
        # slightly reduced to allow controlled oscillation during fast motion
        pole_angle_penalty = 2.0 * (theta ** 2)
        pole_vel_penalty = 0.4 * (theta_dot ** 2)

        # Final reward
        reward =  (velocity_term + position_term + direction_term - pole_angle_penalty - pole_vel_penalty)

    return reward


# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v1 (already embedded in)
    render_mode = "human" if (args.render_training or args.render_test) else None
    env = gym.make(args.env, render_mode=render_mode, max_episode_steps=500)

    # Task 1.3

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, lr=args.lr)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        # Train
        training_history = train(agent, env, args.train_episodes, False, 
                                 args.render_training, 
                                 x0=args.central_point, 
                                 random_policy=args.random_policy,
                                 reward_mode=args.reward_mode)

        # Save the model
        os.makedirs("models", exist_ok=True)

        if args.random_policy:
            filename = f"{args.env}_params_random_policy.ai"
        elif args.lr and args.reward_mode is None:
            # Formatta il learning rate in modo leggibile (ad esempio 1e-3 -> 0.001)
            lr_str = f"{args.lr:.0e}" if args.lr < 1 else str(args.lr)
            filename = f"{args.env}_params_lr{lr_str}.ai"
        elif args.reward_mode and args.lr:
            lr_str = f"{args.lr:.0e}" if args.lr < 1 else str(args.lr)
            filename = f"{args.env}_params_lr{lr_str}_reward_{args.reward_mode}.ai"
            
        else:
            filename = f"{args.env}_params.ai"

        model_file = os.path.join("models", filename)
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
        sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
        plt.legend()
        plt.title("Reward history (%s)" % args.env)
        plt.show()
        print(f"Training finished with {len(training_history)} episodes, "
      f"{training_history['reward'].sum():.3f} total rewards and "
      f"{training_history['reward'].mean():.3f} mean reward.")



    else:

        model_path = args.test
        if not os.path.dirname(model_path):
            model_path = os.path.join("models", model_path)

        print("Loading model from", model_path, "...")
        state_dict = torch.load(model_path)
        policy.load_state_dict(state_dict)
        print("Testing...")
        max_vel = test(agent, env, episodes=args.train_episodes, render=args.render_test, reward_mode=args.reward_mode) 
        print(f'Max velocity reached {max_vel}')


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

