import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network model
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, seed, fc_units=128, device="cpu"):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.fc3 = nn.Linear(fc_units, action_dim)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)