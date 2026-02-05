import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd
import os
import sys

np.random.seed(123)

# Create output folders
os.makedirs("q_val", exist_ok=True)
os.makedirs("plots", exist_ok=True)

env = gym.make('CartPole-v1')

# Whether to perform training or use the stored .npy file
MODE = 'TRAINING'  # TRAINING, TEST

episodes = 20000
test_episodes = 100
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
constant_eps = 0.2

def get_b(eps: float, k : int):
    """computer the value of b to have epsilon = eps after episodes = k

    Args:
        eps (int): epsiolon value for GLIE schedule
        k (int): number of episodes for GLIE schedule

    Returns:
        b (int): b value in GLIE schedule
    """
    return round((k * eps) / (1 - eps))

b = get_b(eps= 0.1, k= 20000)

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# Initialize Q values
q_grid = np.full((discr, discr, discr, discr, num_of_actions), 50)

if MODE == 'TEST':
    q_grid = np.load('q_val/test_greedy.npy')


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    """Returns discrete state from continuous state"""
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, greedy=False):
    """
    Select an action for a given state using either a greedy or epsilon-greedy policy.

    Parameters:
        state (np.ndarray): The current state of the environment.
        q_values (np.ndarray): The Q-table containing Q-values for all state-action pairs.
        greedy (bool, optional): If True, selects the action with the highest Q-value (greedy policy).
                                 If False, selects actions according to an epsilon-greedy policy. Defaults to False.

    Returns:
        int: The index of the chosen action.
    """
    x, v, th, av = get_cell_index(state)

    if greedy:  # TEST -> greedy policy
        best_action_estimated = np.argmax(q_values[x, v, th, av])
        return best_action_estimated

    else:  # TRAINING -> GLIE epsilon-greedy policy
        if np.random.rand() < epsilon:
            # Random action
            action_chosen = np.random.randint(num_of_actions)
            return action_chosen
        else:
            # Greedy action
            best_action_estimated = np.argmax(q_values[x, v, th, av])
            return best_action_estimated


def update_q_value(old_state, action, new_state, reward, done, q_array):
    """Update the Q value function

    Args:
        old_state (np.ndarray): The previous state before taking the action.
        action (int): The action taken in the old_state.
        new_state (np.ndarray): The resulting state after taking the action.
        reward (float): The reward received after taking the action.
        done (bool): Whether the new_state is terminal (True) or not (False).
        q_array (np.ndarray): The Q-table storing Q-values for all state-action pairs.
    """
    
    old_idx = get_cell_index(old_state)
    new_idx = get_cell_index(new_state)

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        target_value = reward + gamma * np.max(q_array[new_idx])

    # Update Q value
    q_grid[old_idx][action] = q_grid[old_idx][action] + alpha * (target_value - q_grid[old_idx][action])
    return


def plot(ep_lengths, epl_avg):
    """Plots and saves training performance in one figure"""
    plt.figure(figsize=(12, 6))
    plt.plot(ep_lengths, alpha=0.4, label="Return per episode")
    plt.plot(epl_avg, linewidth=2, label="Smoothed return (last 500)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-learning Performance on CartPole - 50_greedy")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/50_greedy.png")
    plt.close()


# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes + test_episodes):
    test = ep > episodes

    if MODE == 'TEST':
        test = True

    state, _ = env.reset(seed=321)
    done = False
    steps = 0

    # GLIE schedule (Task 3.1)
    epsilon = 0 #constant_eps, 0, (b/ (b + ep))

    if test:
        epsilon = 0

    while not done:
        action = get_action(state, q_grid, greedy=test)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            env.render()

        state = new_state
        steps += 1

    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep - 500):]))

    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))
        print("Epsilon:", epsilon)


if MODE == 'TEST':
    sys.exit()

# Compute value function from trained Q-values 
# v_grid_values = np.max(q_grid, axis=-1)  # max over actions
# v_avg = np.mean(v_grid_values, axis=(1, 2))  # average over x and theta

# Plot heatmap 
# plt.figure(figsize=(8, 6))
# sns.heatmap(v_avg, xticklabels=np.round(av_grid, 2), yticklabels=np.round(v_grid, 2), cmap='viridis')
# plt.xlabel('Pole Angular Velocity (theta_dot)')
# plt.ylabel('Cart Velocity (x_dot)')
# plt.title(f'Value Function Heatmap (averaged over x_dot and theta)')
# plt.savefig("./plots/value_function_heatmap.png")
# plt.show()

# Save the Q-value array
np.save("q_val/50_greedy.npy", q_grid)

plot(ep_lengths, epl_avg)
