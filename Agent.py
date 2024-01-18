import math
from torch.autograd import Variable
import torch.nn as nn
from collections import deque
import torch
import random
import os
from const import width
from Player import Player
import torch.optim as optim
import numpy as np

MAX_MEMORY = 10000
BATCH_SIZE = 500
LR = 0.005


class Agent(Player):
    def __init__(self, colour, playernumber):
        super().__init__(colour, playernumber)
        self.n_games = 0
        self.epsilon = 0.95
        self.epsilonDecay = 0.99995
        self.epsilonMin = 0.1
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.NN = DQN()
        self.optimizer = optim.Adam(self.NN.parameters(), lr=LR)
        if torch.cuda.is_available():
            # Macht das Sinn?
            self.NN = self.NN.cuda()
        if os.path.isfile("./model/weights" + str(playernumber + 1) + ".pth"):
            print("Loading weights")
            if torch.cuda.is_available():
                self.NN.load_state_dict(
                    torch.load("./model/weights" + str(playernumber + 1) + ".pth")
                )
            else:
                self.NN.load_state_dict(
                    torch.load(
                        "./model/weights" + str(playernumber + 1) + ".pth",
                        map_location=torch.device("cpu"),
                    )
                )
        # self.trainer = Trainer(self.NN, lr=LR, gamma=self.gamma)

    def getAction(self, input):
        ## Inputs: 0 = left, 1 = right, 2 = up, 3 = down, 4 = shootleft, 5 = shootright, 6 = shootup, 7 = shootdown

        if random.random() < self.epsilon:
            return random.randint(0, 7)

        else:
            return self.getActionNoRandom(input)

    def getState(self):
        return (
            self.controll.x,
            self.controll.y,
            self.controll.moveDir.x,
            self.controll.moveDir.y,
        )

    def getActionNoRandom(self, input):
        frameTensor = torch.Tensor(input)

        if torch.cuda.is_available():
            frameTensor = frameTensor.cuda()

        out = self.NN(frameTensor)
        return torch.argmax(out).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        if torch.cuda.is_available():
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()

        q_targets_next = self.NN(next_states).detach().max(1)[0]
        q_targets = rewards + self.gamma * q_targets_next * dones

        q_expected = self.NN(states).gather(1, actions.unsqueeze(1).long())

        loss = nn.MSELoss()(q_expected, q_targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def findNearestBullet(self):
        minDistance = 100000
        if len(self.enemy.getBullets()) == 0:
            return 0, 0, 0, 0
        for bullet in self.enemy.getBullets():
            if bullet.collider is not None:
                distance = math.sqrt(
                    (self.controll.x - bullet.x) ** 2
                    + (self.controll.y - bullet.y) ** 2
                )
                if distance < minDistance:
                    minDistance = distance
                    self.nearestBullet = bullet
        return (
            self.nearestBullet.x,
            self.nearestBullet.y,
            self.nearestBullet.direction.x,
            self.nearestBullet.direction.y,
        )
