import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import numpy as np

if 'inline' in matplotlib.get_backend():
    from IPython import display

plt.ion()

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor
Tensor = FloatTensor

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        #self.layer1 = nn.Sequential(
        #    nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
        #    nn.ReLU(),
        #    nn.MaxPool2d(3, 3),
        #    nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
        #    nn.ReLU(),
        #    nn.MaxPool2d(3, 3),
        #    nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
        #    torch.nn.Flatten(),
        #    torch.nn.Linear(98568, 128)
        #)

        self.layer2 = nn.Sequential(
            nn.Linear(17,128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Softmax()
        )


    def forward(self, x):
        #if len(x.shape) == 4:
        #    x = x.permute(0,3,1,2)
        #else:
        #    x = x.permute(2,0,1)
        #    x = x.unsqueeze(0)

        #x = self.layer1(x)
        x = self.layer2(x)
        #print("movex: " + str(x[0][0].item()) + " movey: " + str(x[0][1].item()))
        return x
    

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
class Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):

        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        if len(state.shape) == 1:
            #(1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        # 1: predicted Q values with current state
        if torch.cuda.is_available():
            state = state.cuda()
            next_state = next_state.cuda()
            action = action.cuda()
            reward = reward.cuda()

        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        #2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        