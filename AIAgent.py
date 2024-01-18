import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from game import Game
from Player import Player
import random
from collections import deque


# Define the Q-network architecture
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define Deep Q-Learning agent
class DQNAgent(Player):
    def __init__(self, state_size, action_size, color, playernumber):
        super().__init__(color, playernumber)
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.target_q_network = QNetwork(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)  # Experience replay memory
        self.batch_size = 200
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state_tensor = torch.FloatTensor(state)

        if torch.cuda.is_available():
            state_tensor = state_tensor.cuda()

        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_experience(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *batch
        )

        state_batch = torch.FloatTensor(np.array(state_batch))
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        action_batch = torch.LongTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        done_batch = torch.FloatTensor(done_batch)

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            next_state_batch = next_state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            done_batch = done_batch.cuda()

        q_values = self.q_network(state_batch)
        next_q_values = self.target_q_network(next_state_batch)
        target_q_values = (
            reward_batch + (1 - done_batch) * self.gamma * next_q_values.max(dim=1)[0]
        )

        loss = self.loss_fn(
            q_values.gather(1, action_batch.unsqueeze(1).long()),
            target_q_values.unsqueeze(1),
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


if __name__ == "__main__":
    # Initialize the game environment and agent
    game = Game()
    action_size = 8
    state_size = 17
    agent1 = DQNAgent(state_size, action_size, "darkgreen", 0)
    agent2 = DQNAgent(state_size, action_size, "darkred", 1)
    game.addPlayers(agent1, agent2)
    # Training loop
    num_episodes = 2000
    for episode in range(num_episodes):
        state = game.reset()
        total_reward = 0
        done = False
        while not done:
            action1 = agent1.select_action(state)
            action2 = agent2.select_action(state)
            next_state, reward, done = game.playStep([action1, action2])
            total_reward += sum(reward)
            agent1.store_experience(state, action1, reward[0], next_state, done)
            agent2.store_experience(state, action2, reward[1], next_state, done)
            state = next_state
            agent1.replay_experience()
            agent2.replay_experience()
        agent1.update_target_network()
        agent2.update_target_network()
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    # Save the trained model
    torch.save(agent1.q_network.state_dict(), "dqn_model1.pth")
    torch.save(agent2.q_network.state_dict(), "dqn_model2.pth")
