import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        fc1_dims,
        fc2_dims,
        fc3_dims,
        n_actions,
        action_range,
        name,
        chkpt_dir="Models/ddpg",
    ) -> None:
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.checkpoint_dir = os.path.join(chkpt_dir, name + "_ddpg")
        self.action_range = action_range

        self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc1.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        f4 = 0.003
        self.mu = nn.Linear(self.fc3_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f4, f4)
        T.nn.init.uniform_(self.mu.bias.data, -f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = T.tanh(self.mu(x)) * self.action_range

        return x

    def save_checkpoint(self):
        print("...saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_dir)

    def load_checkpoint(self):
        print("...loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_dir))
