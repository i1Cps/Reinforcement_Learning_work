from pathlib import Path
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(
        self,
        beta: float,
        input_dims: int,
        fc1: int = 400,
        fc2: int = 300,
        name: str = "critic",
        checkpoint_dir: str = "models",
        scenario: str = "unclassified",
    ):
        super(CriticNetwork, self).__init__()
        self.checkpoint_dir = Path(checkpoint_dir) / scenario
        self.checkpoint_file = self.checkpoint_dir / (name + "_maddpg")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.beta = beta
        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.q = nn.Linear(fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action) -> T.Tensor:
        state_action_value = F.relu(self.fc1(T.cat([state, action], dim=1)))
        state_action_value = F.relu(self.fc2(state_action_value))
        return self.q(state_action_value)

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha: float,
        input_dims: int,
        n_actions: int,
        max_action: int = 1,
        fc1: int = 300,
        fc2: int = 400,
        name: str = "actor",
        checkpoint_dir: str = "models",
        scenario: str = "unclassified",
    ):
        super(ActorNetwork, self).__init__()
        self.checkpoint_dir = Path(checkpoint_dir) / scenario
        self.checkpoint_file = self.checkpoint_dir / (name + "_maddpg")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.alpha = alpha

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.mu = nn.Linear(fc2, n_actions)

        self.max_action = max_action

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state) -> T.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = T.tanh(self.mu(x))
        return self.max_action * x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
