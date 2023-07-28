import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import random


if 'inline' in matplotlib.get_backend():
    from IPython import display

plt.ion()

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor
Tensor = FloatTensor

class Memory:
    def __init__(self, capacity) -> None:
        self.memory = []
        self.pos = 0
        self.capacity = capacity

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = (state, action, next_state, reward)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(788544, 128)
        )

        self.layer2 = nn.Sequential(
            #nn.ReLu(128,64),
            #nn.ReLu(64,32),
           # nn.ReLU(32,7)
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        #print(x)
        return x
    
