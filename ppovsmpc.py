import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import pygame
from scipy.linalg import expm
import cvxpy as cp
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
# [Previous PPONetwork, PPOAgent, compute_gae classes remain the same]

class MPCGameEnv:
    def __init__(self, visualize=False):
        # Initialize parameters
        self.dt = 0.1
        self.friction_coefficient = 1
        self.x_agent = np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5),
                                np.random.uniform(-np.pi, np.pi), 1, 0.0])
        self.x_mpc = np.array([0.0, 0.0, -self.x_agent[2], 1, 0.0])
        
        # MPC parameters
        self.Q = np.diag([0.1, 0.1, 1, 0.001, 0.001])
        self.R = np.diag([0.0001, 0.01])
        self.horizon = 10
        self.u_guess = 0
        
        # Visualization setup
        self.visualize = visualize
        if visualize:
            pygame.init()
            self.WIDTH, self.HEIGHT = 800, 600
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
                                np.random.uniform(-np.pi, np.pi), 1, 0.0])
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
    
    def render(self):
        if not self.visualize:
            return
            
        self.screen.fill(self.WHITE)
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
        distance_from_mpc = np.sqrt((self.x_agent[0] - self.x_mpc[0])**2 + 
                         (self.x_agent[1] - self.x_mpc[1])**2)
        
        done = False

        if abs(self.x_agent[0])<20 and abs(self.x_agent[1])<20:
            reward = 0.1
        else:
            reward = -0.2

        if distance_from_mpc < 1.5:
            reward = -1
            done = True
        elif distance_from_mpc < 5:
            reward -= 0.1

        return reward, done
    
    def step(self, action):
        # Update MPC
        u_mpc = self._get_mpc_action()
        self.x_mpc = self.x_mpc + self.dt * state_space_model(self.x_mpc, u_mpc, self.friction_coefficient)
        
        # Apply PPO action to agent
        action = np.clip(action, [-10, -np.pi], [10, np.pi])
        self.x_agent = self.x_agent + self.dt * state_space_model(self.x_agent, action, self.friction_coefficient)
        
        
        if self.visualize:
            self.agent_history.append(self.x_agent.copy())
            self.mpc_history.append(self.x_mpc.copy())
        
        # Calculate reward and check if done
        reward, done = self._compute_reward()
        self.u_guess = u_mpc[0]
        if done:
            self.u_guess = 0

        
        return self._get_state(), reward, done, {}

def train_ppo(vis_interval=50):
    env = MPCGameEnv(visualize=False)  # Start without visualization
    state_dim = 10  # Combined agent and MPC states
    action_dim = 2  # Same as original control inputs
    
    agent = PPOAgent(state_dim, action_dim)
    max_episodes = 1000
    max_steps = 400
    
    # Training metrics
    episode_rewards_history = []
    
    for episode in range(max_episodes):
        # Create visualization environment for specific episodes
        if episode % vis_interval == 0:
            vis_env = MPCGameEnv(visualize=True)
            vis_state = vis_env.reset()
        
        # Regular training environment
        state = env.reset()
        episode_rewards = []
        states = []
        actions = []
        log_probs = []
        values = []
        dones = []
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = agent.network.get_action(state_tensor)
            
            # Take step in training environment
            next_state, reward, done, _ = env.step(action.detach().numpy()[0])
            
            # If this is a visualization episode, also step the visualization environment
            if episode % vis_interval == 0:
                vis_state_tensor = torch.FloatTensor(vis_state).unsqueeze(0)
                vis_action, _, _ = agent.network.get_action(vis_state_tensor)
                vis_next_state, _, vis_done, _ = vis_env.step(vis_action.detach().numpy()[0])
                vis_env.render()
                pygame.time.wait(50)  # Add delay to make visualization visible
                vis_state = vis_next_state
                
                if vis_done:
                    break
            
            # Store transition
            states.append(state)
            actions.append(action.detach().numpy()[0])
            log_probs.append(log_prob.detach())
            values.append(value)
            dones.append(done)
            episode_rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        # Calculate returns and advantages
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        _, _, next_value = agent.network(next_state_tensor)
        returns, advantages = compute_gae(episode_rewards, values, next_value, dones)
        
        # Update PPO agent
        agent.update(states, actions, log_probs, returns, advantages)
        
        # Store metrics
        episode_rewards_history.append(np.mean(episode_rewards))
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards):.2f}")
            
        # Clean up visualization
        if episode % vis_interval == 0:
            pygame.quit()
            pygame.init()  # Reinitialize for next visualization
    
    return agent, episode_rewards_history

if __name__ == "__main__":
    # Train the PPO agent with visualization every 50 episodes
    trained_agent, rewards_history = train_ppo(vis_interval=50)
    pygame.quit()