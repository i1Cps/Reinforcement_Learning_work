import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, beta, name):
        super(CriticNetwork, self).__init__()
        self.beta = beta
        self.chkpt_dir = "model_weights"
        self.checkpoint_file = os.path.join(self.chkpt_dir, name + "_ddpg")

        # Three Fully Connected Layers
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + n_actions, fc2_dims)
        # self.fc3 = nn.Linear(fc2_dims, fc2_dims)

        # Batch Normalization vs Layer Normalization
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        # self.bn3 = nn.LayerNorm(fc2_dims)

        # State-Action Value ~ Q Value
        self.q = nn.Linear(fc2_dims, 1)

        # Initalize weights
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        # f3 = 1.0 / np.sqrt(self.fc3.weight.data.size()[0])
        # self.fc3.weight.data.uniform_(-f3, f3)
        # self.fc3.bias.data.uniform_(-f3, f3)

        q4 = 0.003
        self.q.weight.data.uniform_(-q4, q4)
        self.q.bias.data.uniform_(-q4, q4)

        # Adam Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta, weight_decay=0.01)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    # Concatanate State and Action in second layer
    def forward(self, state, action):
        state_value = F.relu(self.bn1(self.fc1(state)))
        state_action_value = T.cat([state_value, action], 1)
        state_action_value = F.relu(self.bn2(self.fc2(state_action_value)))
        # state_action_value = F.relu(self.bn3(self.fc3(state_action_value)))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, alpha, name):
        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.chkpt_dir = "model_weights"
        self.checkpoint_file = os.path.join(self.chkpt_dir, name + "_ddpg")

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # self.fc3 = nn.Linear(fc2_dims, fc2_dims)

        self.bn1 = nn.LayerNorm(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        # self.bn3 = nn.LayerNorm(fc2_dims)

        self.mu = nn.Linear(fc2_dims, n_actions)

        # Initalize weights
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        # f3 = 1.0 / np.sqrt(self.fc3.weight.data.size()[0])
        # self.fc3.weight.data.uniform_(-f3, f3)
        # self.fc3.bias.data.uniform_(-f3, f3)

        f4 = 0.003
        self.mu.weight.data.uniform_(-f4, f4)
        self.mu.bias.data.uniform_(-f4, f4)

        # Adam Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    # Get Action from State
    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        # x = F.relu(self.bn3(self.fc3(x)))
        x = T.tanh(self.mu(x))
        return x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
