import os
import torch as T
import torch.nn.functional as F
from torch.distributions import Beta
import torch.optim as optim
import torch.nn as nn


class ContinuousActorNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        alpha,
        fc1=128,
        fc2=128,
        checkpoint_dir="models/",
        scenario=None,
    ):
        super(ContinuousActorNetwork, self).__init__()
        self.checkpoint_dir += scenario
        self.checkpoint_file = os.path.join(checkpoint_dir, "actor_continous_ppo")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.fc1 = nn.Linear(*input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.alpha = nn.Linear(fc2, n_actions)
        self.beta = nn.Linear(fc2, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        alpha = F.relu(self.alpha(x)) + 1.0
        beta = F.relu(self.beta(x)) + 1.0
        dist = Beta(alpha, beta)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ContinuousCriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        alpha,
        fc1=128,
        fc2=128,
        checkpoint_dir="model/",
        scenario=None,
    ):
        super(ContinuousCriticNetwork, self).__init__()

        self.checkpoint_dir += scenario
        self.checkpoint_file = os.path.join(checkpoint_dir, "critic_continous_ppo")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.fc1 = nn.Linear(*input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.v = nn.Linear(fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        v = self.v(x)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
