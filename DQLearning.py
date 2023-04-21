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
        self.cl1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.norm1 = nn.BatchNorm2d(16)
        self.cl2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.norm2 = nn.BatchNorm2d(32)
        self.cl3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.norm3 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(448, 2)
