import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# Define the Policy Network for Continuous Action Space
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Separate heads for mean and standard deviation
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(output_dim))  # Learnable log std

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std_layer)  # Ensure std is positive
        return mean, std

# Function to compute discounted rewards
def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    return torch.tensor(discounted_rewards)

# Function to normalize rewards
def normalize_rewards(rewards):
    mean = rewards.mean()
    std = rewards.std() + 1e-7  # Avoid division by zero
    return (rewards - mean) / std

# Hyperparameters
env = gym.make("Pendulum-v1", render_mode="human")
input_dim = env.observation_space.shape[0]
hidden_dim = 128
output_dim = env.action_space.shape[0]  # Dimensionality of the action space
learning_rate = 0.001
num_episodes = 1000
gamma=0.99

# Initialize the policy network and optimizer
policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Main REINFORCE Algorithm for Continuous Action Space
for episode in range(num_episodes):
    state = env.reset()[0]
    log_probs = []
    rewards = []

    done = False
    i = 0
    while not done:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)

        # Get mean and std from the policy network
        mean, std = policy_net(state_tensor)

        # Sample an action from a Gaussian distribution
        action_dist = Normal(mean, std)
        action = action_dist.sample()

        # Store the log probability of the chosen action
        log_prob = action_dist.log_prob(action).sum(axis=-1)  # Sum for multi-dimensional actions
        log_probs.append(log_prob)

        # Take the action in the environment (clip action to valid range if needed)
        action_clipped = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])
        state, reward, done, _ = env.step(action_clipped.numpy())[:-1]
        rewards.append(reward)
        i += 1
        if i==400:
            done = True
        
    # Compute discounted rewards
    discounted_rewards = compute_discounted_rewards(rewards, gamma)

    # Normalize discounted rewards
    discounted_rewards = normalize_rewards(discounted_rewards)

    # Compute the loss
    policy_loss = []
    for log_prob, reward in zip(log_probs, discounted_rewards):
        policy_loss.append((-log_prob * reward).unsqueeze(0))  # Add a dimension to make it 1D
    policy_loss = torch.cat(policy_loss).sum()  # Concatenate and sum over all losses

    # Update the policy network
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # Print progress
    total_reward = sum(rewards)
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

state = env.reset()[0]
done = False
i = 0
while not done:
    # Convert state to tensor
    state_tensor = torch.FloatTensor(state)

    # Get mean and std from the policy network
    with torch.no_grad():
        mean, std = policy_net(state_tensor)

    # Sample an action from a Gaussian distribution
    action_dist = Normal(mean, std)
    action = action_dist.sample()

    # Store the log probability of the chosen action
    log_prob = action_dist.log_prob(action).sum(axis=-1)  # Sum for multi-dimensional actions
    log_probs.append(log_prob)

    # Take the action in the environment (clip action to valid range if needed)
    action_clipped = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])
    state, reward, done, _ = env.step(action_clipped.numpy())[:-1]
    env.render()
    i+=1
    if i > 1000:
        env.close
        break

