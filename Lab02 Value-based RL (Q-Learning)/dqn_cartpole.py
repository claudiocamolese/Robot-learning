import os
import random
from collections import deque, namedtuple
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Repro
SEED = 123
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]   # 4
print(state_dim)
print(env.observation_space.shape)
n_actions = env.action_space.n #2

# Hyperparameters
lr = 1e-3
gamma = 0.98
batch_size = 64
buffer_size = 100000
b = 100000
min_replay_size = 1000
target_update_freq = 1000   # number of gradient steps between target updates
max_episodes = 2000
max_steps = 500
save_dir = "dqn_checkpoints"
os.makedirs(save_dir, exist_ok=True)


env_variables = namedtuple("Step", ("s", "a", "r", "s2", "done"))

class Memory:
    """
    Simple replay buffer that stores transitions and returns
    random minibatches as PyTorch tensors on the configured device.
    
    capacity (int): maximum number of transitions stored. Implemented via deque.
    buffer (deque) : storage for transitions.
    """
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, *args):
        """
        Add a transition into the buffer.

        Arg:
        *args (tuple): Values expected: (state, action, reward, next_state, done).
            They are stored as a Transition namedtuple.
        """
        self.buffer.append(env_variables(*args))
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions uniformly at random and convert
        them to PyTorch tensors on the global device.

        Args :
        batch_size : int
            Number of transitions to sample.

        Returns :
        s, a, r, s2, done : Tuple[torch.Tensor, ...]
            Tensors with shapes:
              s: (batch, state_dim)
              a: (batch, 1) dtype int64
              r: (batch, 1) dtype float32
              s2: (batch, state_dim)
              done: (batch, 1) dtype float32
        """
        batch = random.sample(self.buffer, batch_size)
        s = torch.tensor(np.vstack([b.s for b in batch]), device=device)
        a = torch.tensor([b.a for b in batch], device=device).unsqueeze(1)
        r = torch.tensor([b.r for b in batch], device=device).unsqueeze(1)
        s2 = torch.tensor(np.vstack([b.s2 for b in batch]), device=device)
        done = torch.tensor([b.done for b in batch], device=device).unsqueeze(1)
        return s, a, r, s2, done
    
    def __len__(self):
        """
        Return current size of the buffer.

        Returns
        (int) : Number of stored transitions.
        """
        return len(self.buffer)

class QNetwork(nn.Module):
    """
    Simple feed-forward network that maps continuous states to Q-values
    for each discrete action.

    Args :
        in_dim (int) : 
            Dimension of input state.
        out_dim (int) : Number of discrete actions (size of Q output).
        hidden (tuple) : Sizes of hidden layers.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)             
        return x


policy_net = QNetwork(state_dim, n_actions).to(device)
target_net = QNetwork(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.SGD(policy_net.parameters(), lr=lr)
replay = Memory(buffer_size)


def GLIE_eps(frame_idx):
    """
    Compute epsilon value using linear decay schedule.

    Args:
        frame_idx (int): Total number of environment steps taken so far.

    Returns:
        float: Current epsilon value for ε-greedy action selection.
    """
    return b / (b + frame_idx)


def select_action(state, epsilon, greedy=False):
    """
    Select an action using ε-greedy policy.

    Args:
        state (np.ndarray): Current state.
        epsilon (float): Probability of choosing a random action.
        greedy (bool, optional): If True, always select argmax action. Default False.

    Returns:
        int: Chosen action index.
    """
    if greedy or np.random.rand() > epsilon:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            qvals = policy_net(state_t)     # shape (1, n_actions)
            action = qvals.argmax(dim=1).item()
        return int(action)
    else:
        return env.action_space.sample()

def train_step():
    """
    Perform a single Q-learning update using a batch sampled
    from the replay buffer.

    Semi-gradient Q-learning update:
        target is detached to prevent gradient from flowing
        through the bootstrap estimate from the target network.

    Returns:
        float or None: Loss value if training occurred, otherwise None while warming up buffer.
    """
    global grad_steps
    if len(replay) < min_replay_size:
        return None
    s, a, r, s2, done = replay.sample(batch_size)

    q_values = policy_net(s).gather(1, a)                

    with torch.no_grad():
        q_next = target_net(s2)                         
        q_next_max, _ = q_next.max(dim=1, keepdim=True) 
        target = r + gamma * (1.0 - done) * q_next_max  

    loss = F.mse_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    grad_steps += 1
    
    if grad_steps % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    return loss.item()

state, _ = env.reset(seed=SEED)

for _ in range(min_replay_size):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    replay.add(state, action, reward, next_state, float(done))
    if done:
        state, _ = env.reset()
    else:
        state = next_state


episode_rewards = []
frame_idx = 0
grad_steps = 0
losses = []

for ep in range(1, max_episodes + 1):
    state, _ = env.reset()
    ep_reward = 0.0
    for t in range(max_steps):
        epsilon = GLIE_eps(frame_idx)
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay.add(state, action, reward, next_state, float(done))
        state = next_state
        ep_reward = ep_reward + reward
        frame_idx += 1

        loss = train_step()
        if loss is not None:
            losses.append(loss)

        if done:
            break

    episode_rewards.append(ep_reward)

    if ep % 50 == 0:
        avg_recent = np.mean(episode_rewards[-50:])
        print(f"Ep {ep:4d} | frames {frame_idx:6d} | eps {epsilon:.3f} | avg50_reward {avg_recent:.2f}")

    
    if ep % 200 == 0:
        torch.save(policy_net.state_dict(), os.path.join(save_dir, f"policy_ep{ep}.pth"))

# Save final model
torch.save(policy_net.state_dict(), os.path.join(save_dir, "policy_final.pth"))

# Plot training curve
plt.figure(figsize=(10,5))
plt.plot(episode_rewards, alpha=0.6, label='episode reward')
plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), label='moving avg50')
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DQN-style Q-learning with function approximation")
plt.legend()
plt.savefig("plots/dqn_training.png")
plt.show()


eval_episodes = 20
eval_rewards = []
for _ in range(eval_episodes):
    state, _ = env.reset()
    ep_r = 0.0
    done = False
    while not done:
        action = select_action(state, epsilon=0.0, greedy=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_r += reward
    eval_rewards.append(ep_r)
print("Eval avg reward:", np.mean(eval_rewards))
