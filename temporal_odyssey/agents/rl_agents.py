import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Neural Network Definitions

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = nn.functional.relu(self.layer_1(x))
        x = nn.functional.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = nn.functional.relu(self.layer_1(x))
        x = nn.functional.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

# TD3 (Twin Delayed DDPG) Implementation

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=3e-4)

        self.max_action = max_action
        self.device = device
        self.replay_buffer = deque(maxlen=int(1e6))
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

    def train(self, iterations):
        for _ in range(iterations):
            self.total_it += 1
            
            # Sample a batch of transitions from the replay buffer
            if len(self.replay_buffer) < self.batch_size:
                return
            
            batch = random.sample(self.replay_buffer, self.batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
            
            # Compute target Q-values using target networks
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                target_Q1 = self.critic_target_1(next_state, next_action)
                target_Q2 = self.critic_target_2(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * self.gamma * target_Q

            # Compute current Q-values and critic loss
            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

            # Optimize critic networks
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # Optimize actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                self.soft_update_target_networks()

    def soft_update_target_networks(self):
        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# SAC (Soft Actor-Critic) Implementation

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=3e-4)

        self.max_action = max_action
        self.device = device
        self.replay_buffer = deque(maxlen=int(1e6))
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.automatic_entropy_tuning = True

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

    def train(self, iterations):
        for _ in range(iterations):
            # Sample a batch of transitions from the replay buffer
            if len(self.replay_buffer) < self.batch_size:
                return

            batch = random.sample(self.replay_buffer, self.batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

            # Compute target Q-values using target networks
            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_state)
                target_Q1 = self.critic_target_1(next_state, next_action)
                target_Q2 = self.critic_target_2(next_state, next_action)
                target_V = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
                target_Q = reward + (1 - done) * self.gamma * target_V
                
            # Compute current Q-values and critic loss
            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

            # Optimize critic networks
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            pi, log_pi, _ = self.actor.sample(state)
            actor_Q1 = self.critic_1(state, pi)
            actor_Q2 = self.critic_2(state, pi)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha * log_pi - actor_Q).mean()

            # Optimize actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Tune entropy coefficient (alpha) automatically
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

            # Soft update target networks
            self.soft_update_target_networks()

    def soft_update_target_networks(self):
        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Device Handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example Usage
if __name__ == "__main__":
    state_dim = 8
    action_dim = 2
    max_action = 1.0

    td3_agent = TD3Agent(state_dim, action_dim, max_action, device)
    sac_agent = SACAgent(state_dim, action_dim, max_action, device)

    # Collect experiences and train the agents
    num_episodes = 100
    for episode in range(num_episodes):
        state = np.random.randn(state_dim)
        done = False
        episode_reward = 0

        while not done:
            action = td3_agent.select_action(state)
            next_state = np.random.randn(state_dim)
            reward = np.random.randn(1)
            done = np.random.choice([True, False])
            td3_agent.add_experience(state, action, reward, next_state, done)
            sac_agent.add_experience(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            td3_agent.train(1)
            sac_agent.train(1)

        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    # Select actions using the trained agents
    test_state = np.random.randn(state_dim)
    td3_action = td3_agent.select_action(test_state)
    sac_action = sac_agent.select_action(test_state)
    print(f"TD3 Action: {td3_action}")
    print(f"SAC Action: {sac_action}")
