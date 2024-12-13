import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import pygame
from scipy.linalg import expm
import cvxpy as cp
from collections import deque
import math

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages = []
    advantage = 0
    next_value = next_value.item()
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_value
        else:
            next_value = values[t + 1].item()
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t].item()
        advantage = delta + gamma * lam * (1 - dones[t]) * advantage
        advantages.insert(0, advantage)
    
    returns = np.array(advantages) + np.array([v.item() for v in values])
    advantages = np.array(advantages)
    
    return returns, advantages

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPONetwork, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)  # Mean and std for each action
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        action_params = self.actor(state)
        value = self.critic(state)
        
        # Split into means and standard deviations
        action_mean, action_std = torch.chunk(action_params, 2, dim=-1)
        action_std = F.softplus(action_std) + 1e-5  # Ensure positive std
        
        return action_mean, action_std, value
    
    def get_action(self, state):
        action_mean, action_std, value = self(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, c1=1.0, c2=0.01):
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1  # Value loss coefficient
        self.c2 = c2  # Entropy coefficient
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Multiple PPO update epochs
        for _ in range(10):
            # Get current policy distributions and values
            action_mean, action_std, values = self.network(states)
            dist = Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()
            
            # Calculate PPO policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class PPOAgentWithReplay(PPOAgent):
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, c1=1.0, c2=0.01, buffer_size=10000, batch_size=64):
        super().__init__(state_dim, action_dim, lr, gamma, epsilon, c1, c2)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def store_transition(self, state, action, log_prob, reward, done):
        self.replay_buffer.append((state, action, log_prob, reward, done))

    def sample_replay_buffer(self):
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        sampled_transitions = [self.replay_buffer[i] for i in indices]

        states, actions, log_probs, rewards, dones = zip(*sampled_transitions)
        return (
            np.array(states),
            np.array(actions),
            np.array(log_probs),
            np.array(rewards),
            np.array(dones)
        )

    def update_from_replay_buffer(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples in buffer

        states, actions, log_probs, rewards, dones = self.sample_replay_buffer()

        # Calculate returns and advantages using sampled rewards
        next_value = torch.FloatTensor([0.0])  # Assume next value is zero for buffer samples
        returns, advantages = compute_gae(rewards, np.zeros_like(rewards), next_value, dones)

        self.update(states, actions, log_probs, returns, advantages)


def train_ppo_with_replay(vis_interval=50):
    env = MPCGameEnv(visualize=True)
    state_dim = 10  # Combined agent and MPC states
    action_dim = 2  # Same as original control inputs

    agent = PPOAgentWithReplay(state_dim, action_dim)
    max_episodes = 10000
    max_steps = 1000

    # Training metrics
    episode_rewards_history = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_rewards = []

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, _ = agent.network.get_action(state_tensor)

            # Take step in training environment
            next_state, reward, done, _ = env.step(action.detach().numpy()[0])

            # Store transition in replay buffer
            agent.store_transition(state, action.detach().numpy()[0], log_prob.detach(), reward, done)

            # If this is a visualization episode, also step the visualization environment
            if episode % vis_interval == 0:
                env.render()
                pygame.time.wait(50)  # Add delay to make visualization visible

            episode_rewards.append(reward)
            state = next_state

            if done:
                break

        # Update PPO agent using replay buffer
        agent.update_from_replay_buffer()

        # Store metrics
        episode_rewards_history.append(np.mean(episode_rewards))

        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards):.2f}")

        # Clean up visualization after all training is done
        if episode % vis_interval == 0:
            pygame.quit()
            pygame.init()  # Reinitialize for next visualization

    return agent, episode_rewards_history

def state_space_model(x, u, friction_coefficient):
    _, _, theta, vx, vy = x
    u1, u2 = u
    dx = np.zeros_like(x)
    dx[0] = vx
    dx[1] = vy
    dx[2] = u2
    dx[3] = u1 * np.cos(theta) - friction_coefficient * vx
    dx[4] = u1 * np.sin(theta) - friction_coefficient * vy
    return dx

def get_linear_matrices(theta, vx, vy, u1, friction_coefficient):
    A = np.array([
        [0,    0,       0,                    1,                   0],
        [0,    0,       0,                    0,                   1],
        [0,    0,       0,                    0,                   0],
        [0,    0, -u1*np.sin(theta), -friction_coefficient,        0],
        [0,    0,  u1*np.cos(theta),          0,        -friction_coefficient]
    ])
    B = np.array([
        [0,           0],
        [0,           0],
        [0,           1],
        [np.cos(theta),0],
        [np.sin(theta),0]
    ])
    return A, B

class MPCGameEnv:
    def __init__(self, visualize=False):
        # Initialize parameters
        self.dt = 0.1
        self.friction_coefficient = 1
        self.x_agent = np.zeros(5)
        self.x_mpc = np.zeros(5)
        
        # MPC parameters
        self.Q = np.diag([1, 1, 0.1, 0.001, 0.001])
        self.R = np.diag([0.0001, 0.01])
        self.horizon = 10
        self.u_guess = 0
        self.max_distance = 30  # Maximum allowed distance from origin
        self.collision_threshold = 1  # Distance at which collision occurs
        self.warning_distance = 5  # Distance at which warning reward is given
        
        # Visualization setup
        self.visualize = visualize
        if visualize:
            pygame.init()
            self.WIDTH, self.HEIGHT = 600, 600
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("PPO vs MPC Vehicle")
            self.font = pygame.font.SysFont('Arial', 16)
            
            # Colors
            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.RED = (255, 0, 0)
            self.BLUE = (0, 0, 255)
            self.GREEN = (0, 255, 0)
            self.GRAY = (200, 200, 200)
            self.LIGHT_RED = (255, 200, 200)
            
            # Track history
            self.agent_history = []
            self.mpc_history = []
    
    def _get_mpc_action(self):
        # Simplified MPC controller from your original code
        A, B = get_linear_matrices(self.x_mpc[2], self.x_mpc[3], self.x_mpc[4], self.u_guess, self.friction_coefficient)
        A_d = expm(A * self.dt)
        B_d = B * self.dt
        
        x_var = cp.Variable((5, self.horizon + 1))
        u_var = cp.Variable((2, self.horizon))
        
        # Target point for MPC (could be modified to chase the agent)
        x_ref = self.x_agent # np.array([0, 0, 0, 0, 0])  # Or some other reference point
        
        cost = 0
        constraints = [x_var[:, 0] == self.x_mpc]
        
        for t in range(self.horizon):
            cost += cp.quad_form(x_var[:, t] - x_ref, self.Q) + cp.quad_form(u_var[:, t], self.R)
            constraints += [x_var[:, t + 1] == A_d @ x_var[:, t] + B_d @ u_var[:, t]]
            constraints += [cp.abs(u_var[0, t]) <= 10]
            constraints += [cp.abs(u_var[1, t]) <= np.pi]
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        
        if u_var.value is None:
            return np.array([0.0, 0.0])
        return u_var[:, 0].value
    
    def reset(self):
        self.x_agent = np.array([np.random.uniform(-20, 20), np.random.uniform(-20, 20),
                                np.random.uniform(-np.pi, np.pi), 5, 0.0])
        self.x_mpc = np.array([0.0, 0.0, -self.x_agent[2], 1, 0.0])

        if self.visualize:
            self.agent_history = [self.x_agent.copy()]
            self.mpc_history = [self.x_mpc.copy()]
            
        return self._get_state()
    
    def to_screen_coords(self, x, y):
        return int(self.WIDTH / 2 + x * 10), int(self.HEIGHT / 2 - y * 10)
    
    def draw_text(self, text, pos, color=(0, 0, 0)):
        label = self.font.render(text, True, color)
        self.screen.blit(label, pos)
    
    def draw_grid(self):
        for x in range(0, self.WIDTH, 50):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 50):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.WIDTH, y))

    def draw_boundary_circle(self):
        radius_px = self.max_distance * 10  # Scale factor of 10 from the to_screen_coords method
        
        # Draw filled circle with light red color
        pygame.draw.circle(self.screen, self.LIGHT_RED, 
                        (self.WIDTH // 2, self.HEIGHT // 2), 
                        int(radius_px), 0)
        
        # Draw circle border with red color
        pygame.draw.circle(self.screen, self.RED,
                        (self.WIDTH // 2, self.HEIGHT // 2), 
                        int(radius_px), 2)
    
    def render(self):
        if not self.visualize:
            return
            
        self.screen.fill(self.WHITE)
        self.draw_boundary_circle()
        self.draw_grid()
        
        # Draw paths
        for i in range(max(0, len(self.agent_history) - 20), len(self.agent_history)):
            x, y = self.to_screen_coords(self.agent_history[i][0], self.agent_history[i][1])
            pygame.draw.circle(self.screen, self.RED, (x, y), 2)
            
        for i in range(max(0, len(self.mpc_history) - 20), len(self.mpc_history)):
            x, y = self.to_screen_coords(self.mpc_history[i][0], self.mpc_history[i][1])
            pygame.draw.circle(self.screen, self.BLUE, (x, y), 2)
        
        # Draw PPO agent
        agent_x, agent_y = self.to_screen_coords(self.x_agent[0], self.x_agent[1])
        pygame.draw.circle(self.screen, self.RED, (agent_x, agent_y), 5)
        agent_heading_x = agent_x + 15 * math.cos(self.x_agent[2])
        agent_heading_y = agent_y - 15 * math.sin(self.x_agent[2])
        pygame.draw.line(self.screen, self.RED, (agent_x, agent_y), 
                        (agent_heading_x, agent_heading_y), 2)
        
        # Draw MPC vehicle
        mpc_x, mpc_y = self.to_screen_coords(self.x_mpc[0], self.x_mpc[1])
        pygame.draw.circle(self.screen, self.BLUE, (mpc_x, mpc_y), 5)
        mpc_heading_x = mpc_x + 15 * math.cos(self.x_mpc[2])
        mpc_heading_y = mpc_y - 15 * math.sin(self.x_mpc[2])
        pygame.draw.line(self.screen, self.BLUE, (mpc_x, mpc_y), 
                        (mpc_heading_x, mpc_heading_y), 2)
        
        # Display states
        self.draw_text(f"PPO Agent State:", (10, 10))
        self.draw_text(f"X: {self.x_agent[0]:.2f}, Y: {self.x_agent[1]:.2f}", (10, 30))
        self.draw_text(f"Theta: {self.x_agent[2]:.2f}, VX: {self.x_agent[3]:.2f}, VY: {self.x_agent[4]:.2f}", 
                      (10, 50))
        
        self.draw_text(f"MPC State:", (self.WIDTH - 200, 10))
        self.draw_text(f"X: {self.x_mpc[0]:.2f}, Y: {self.x_mpc[1]:.2f}", (self.WIDTH - 200, 30))
        self.draw_text(f"Theta: {self.x_mpc[2]:.2f}, VX: {self.x_mpc[3]:.2f}, VY: {self.x_mpc[4]:.2f}", 
                      (self.WIDTH - 200, 50))
        
        pygame.display.flip()

    def _get_state(self):
        # Combine agent and MPC states into observation
        return np.concatenate([self.x_agent, self.x_mpc])
    
    def _compute_reward(self):
        # Calculate distance between agent and MPC
        distance_from_mpc = np.sqrt(np.square(self.x_mpc[:2]-self.x_agent[:2]).sum())
        
        # Calculate distance from origin
        distance_from_origin = np.sqrt(np.square(self.x_agent[:2]).sum())
        
        # Initialize reward and done flag
        reward = 0
        done = False
        
        # Check boundary conditions first
        if distance_from_origin > self.max_distance:
            
            reward = -1
            done = True
            return reward, done
            
        # Base reward for staying in bounds
        if distance_from_origin < 20:
            reward += 0.1
        else:
            reward -= 0.05  # Smaller penalty for being near the boundary
        
        # Collision check
        if distance_from_mpc < self.collision_threshold:
            reward -= 1
            done = True
        # Warning zone
        elif distance_from_mpc < self.warning_distance:
            reward -= 0.1 * (1 - distance_from_mpc/self.warning_distance)  # Graduated penalty
        else:
            # Small reward for maintaining safe distance
            reward += 0.05
        
        return reward, done
    
    def step(self, action):

        # Apply PPO action to agent
        action = np.clip(action, [-10, -np.pi], [10, np.pi])
        next_agent_state = self.x_agent + self.dt * state_space_model(self.x_agent, action, self.friction_coefficient)

        # Update MPC
        u_mpc = self._get_mpc_action()
        u_mpc = np.clip(u_mpc, [-10, -np.pi], [10, np.pi])
        next_mpc_state = self.x_mpc + self.dt * state_space_model(self.x_mpc, u_mpc, self.friction_coefficient)

        # Update states
        self.x_mpc = next_mpc_state
        self.x_agent = next_agent_state
        
        if self.visualize:
            self.agent_history.append(self.x_agent.copy())
            self.mpc_history.append(self.x_mpc.copy())
        
        # Calculate reward and check if done
        reward, done = self._compute_reward()
        
        # Update MPC guess only if not done
        if not done:
            self.u_guess = u_mpc[1]
        else:
            self.u_guess = 0
        
        return self._get_state(), reward, done, {}

if __name__ == "__main__":
    # Train the PPO agent with visualization every 50 episodes
    trained_agent, rewards_history = train_ppo_with_replay(vis_interval=1)
    pygame.quit()
