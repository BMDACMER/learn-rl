import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_Network(nn.Module):
    def __init__(self, state_size, action_size, hidden=[64, 64]):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Visual_Q_Network(nn.Module):
    """
    The input of this network should have shape (num_frame, 80, 80)
    """
    def __init__(self, num_frame, num_action):
        super(Visual_Q_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_frame, out_channels=16, kernel_size=8, stride=4, padding=2)  # 16, 20, 20
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)   # 32, 9, 9
        self.fc1 = nn.Linear(32 * 81, 256)
        self.fc2 = nn.Linear(256, num_action)

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 81)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


