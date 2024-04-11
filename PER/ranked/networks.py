import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class LinearDeepQNetwork(nn.Module):
    def __init__(self, name, input_dims, n_actions, learning_rate, chkpt_dir):
        """
        Initialize the Deep Q-Network (DQN).

        Parameters:
            name (str): Name of the DQN model.
            input_dims (tuple): Dimensions of the input state.
            n_actions (int): Number of actions in the action space.
            learning_rate (float): Learning rate for the optimizer.
            chkpt_dir (str): Directory for saving/loading checkpoints.
        """
        super(LinearDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # Define fully connected layers
        self.fc1 = nn.Linear(*input_dims, 32)
        self.fc2 = nn.Linear(32, 32)
        self.q = nn.Linear(32, n_actions)

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        # Determine device (GPU or CPU) for computation
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the DQN network.

        Parameters:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output tensor representing action values for each action.
        """

        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        q = self.q(flat2)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
