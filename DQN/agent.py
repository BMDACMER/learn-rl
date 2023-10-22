import random
import torch
import torch.optim as optim
import numpy as np
from collections import deque

from network import *


class Agent:

    def __init__(self, state_size, action_size, bs, lr, tau, gamma, device, visual=False):
        """
        When dealing with visual inputs, state_size should work as num_of_frame
        """
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        if visual:
            self.Q_local = Visual_Q_Network(self.state_size, self.action_size).to(self.device)
            self.Q_target = Visual_Q_Network(self.state_size, self.action_size).to(self.device)
        else:
            self.Q_local = Q_Network(self.state_size, self.action_size).to(self.device)
            self.Q_target = Q_Network(self.state_size, self.action_size).to(self.device)
        self.soft_update(1)
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)
        self.memory = deque(maxlen=100000)

    def act(self, state, eps=0):
        """
        平衡探索和利用
        :param state:
        :param eps:
        :return:
        """
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_values = self.Q_local(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        experience = random.sample(self.memory, self.bs)
        states = torch.from_numpy(np.vstack([e[0] for e in experience])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experience])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experience])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experience])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experience]).astype(np.uint8)).float().to(self.device)

        Q_values = self.Q_local(states)
        Q_values = torch.gather(input=Q_values, dim=-1, index=actions)

        with torch.no_grad():
            Q_targets = self.Q_target(next_states)
            Q_targets, _ = torch.max(input=Q_targets, dim=-1, keepdim=True)
            Q_targets = rewards + self.gamma * (1 - dones) * Q_targets

        loss = (Q_values - Q_targets).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

