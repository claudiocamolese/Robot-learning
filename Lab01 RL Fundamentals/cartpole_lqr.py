"""
    Robot Learning
    Exercise 1

    Linear Quadratic Regulator

    Polito A-Y 2025-2026
"""
import gymnasium as gym
import numpy as np
from scipy import linalg     # get riccati solver
import argparse
import matplotlib.pyplot as plt
import sys
from utils import get_space_dim, set_seed
import pdb 
import time
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment to use")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--time_sleep", action='store_true',
                        help="Add timer for visualizing rendering with a slower frame rate")
    parser.add_argument("--mode", type=str, default="control",
                        help="Type of test ['control', 'multiple_R']")
    return parser.parse_args(args)

def linerized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    # state matrix
    # a1 = 0
    a1 = (g*mp)/((mk + mp)*(mp/(mk + mp) - 4/3))
    a2 = - g /(lp*((mp/mt)-4/3))
    
    A = np.array([[0, 1, 0,  0],
                  [0, 0, a1, 0],
                  [0, 0, 0,  1],
                  [0, 0, a2, 0]])

    # input matrix
    # b1 = 1/mt
    b1 = -(mp/(mt*(mp/mt - 4/3)) - 1)/mt
    b2 = 1 / (lp*mt*((mp/mt)-4/3))
    B = np.array([[0], [b1], [0], [b2]])
    
    return A, B

def optimal_controller(A, B, R_value=1):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=True, seed=0):
    """
    Vary R in [0.01, 0.1, 10, 100] and plot forces applied for the first 500 timesteps.
    """
    R_values = [0.01, 0.1, 10, 100]
    max_steps = 500

    plt.figure(figsize=(10,6))

    for R_val in R_values:
        # Reset environment for comparability
        set_seed(seed)
        env.env.reset(seed=seed)
        
        obs, _ = env.reset()
        A, B = linerized_cartpole_system(mp, mk, l, g)
        K = optimal_controller(A, B, R_value=R_val)

        forces = []

        for i in range(max_steps):
            if time_sleep:
                time.sleep(.01)

            action, force = apply_state_controller(K, obs)
            clipped_force = float(np.clip(force[0], -10.0, 10.0))
            abs_force = abs(clipped_force)
            env.unwrapped.force_mag = abs_force

            # invert force if first applied force > 0
            if i == 0 and clipped_force > 0:
                invert = True
            elif i == 0:
                invert = False

            if invert:
                forces.append(-clipped_force)
            else:
                forces.append(clipped_force)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminate and done:
                break

        plt.plot(range(len(forces)), forces, label=f"R={R_val}")

    plt.xlabel("Timestep")
    plt.ylabel("Applied Force (N)")
    plt.title("Forces applied for different R values (LQR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Salva plot
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_path = os.path.join("plots", "forces_multiple_R.png")
    plt.show()
    #plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")

    


def control(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Task 1: Control using LQR
    """
    states = []

    obs, _ = env.reset()    # Reset the environment for a new episode
    
    A, B = linerized_cartpole_system(mp, mk, l, g)
    K = optimal_controller(A, B)    # Re-compute the optimal controller for the current R value

    for i in range(1000):
        if time_sleep:
            time.sleep(.1)
        
        action, force = apply_state_controller(K, obs)
        clipped_force = float(np.clip(force[0], -10.0, 10.0))
        abs_force = abs(clipped_force)
        env.unwrapped.force_mag = abs_force

        obs, reward, terminated, truncated, _ = env.step(action)
        states.append(obs)
        done = terminated or truncated
        
        if terminate and done:
            print(f'Terminated after {i+1} iterations.')
            break

    # ---- PLOT STATES ----
    states = np.array(states)
    t = np.arange(states.shape[0])

    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label="Cart position")
    plt.plot(t, states[:, 1], label="Cart velocity")
    plt.plot(t, states[:, 2], label="Pole angle")
    plt.plot(t, states[:, 3], label="Pole angular velocity")

    plt.xlabel("Timestep")
    plt.ylabel("State value")
    plt.title("CartPole States over Time (LQR Control)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    if not os.path.exists("plots"):
        os.makedirs("plots")

    #plt.savefig("plots/cartpole_lqr_states.png")
    
    # Convergence of all states within a tolerance. 
    tolerance = 0.003
    within_tol = np.all(np.abs(states) <= tolerance, axis=1)

    converged_timestep = None
    for i in range(len(within_tol)):
        if np.all(within_tol[i:]):
            converged_timestep = i
            break

    if converged_timestep is not None:
        print(f"All states converge to ±{tolerance} after timestep {converged_timestep}.")
    else:
        print(f"States never fully converge within ±{tolerance} in the recorded timesteps.")

# The main function
def main(args):
    # Create a Gymnasium environment
    env = gym.make(args.env, render_mode="human", max_episode_steps=500)

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print("Environment:", args.env)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    set_seed(args.seed)    # seed for reproducibility
    env.reset(seed=args.seed)
    
    mp, mk, l, g = env.unwrapped.masspole, env.unwrapped.masscart, env.unwrapped.length, env.unwrapped.gravity

    if args.mode == "control":
        control(env, mp, mk, l, g, args.time_sleep, terminate=True)
    elif args.mode == "multiple_R":
        multiple_R(env, mp, mk, l, g, args.time_sleep, terminate=True)

    env.close()

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

