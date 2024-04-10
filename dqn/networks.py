import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class DeepQNetwork(nn.Module):
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
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # Define convolutional layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # Calculate input dimensions for the fully connected layer
        fc_input_dims = self._calculate_conv_output_dims(input_dims)

        # Define fully connected layers
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        # Determine device (GPU or CPU) for computation
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def _calculate_conv_output_dims(self, input_dims):
        """
        Calculate the output dimensions after passing through the convolutional layers.

        Parameters:
            input_dims (tuple): Dimensions of the input state.

        Returns:
            int: Number of output dimensions after passing through the convolutional layers.
        """
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        """
        Forward pass through the DQN network.

        Parameters:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output tensor representing action values for each action.
        """
        # Pass through convolutional layers and apply ReLU activation
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        # Flatten the output from convolutional layers
        conv_state = conv3.view(conv3.size()[0], -1)  # -1 flattens

        # Pass through fully connected layers and apply ReLU activation
        flattened = F.relu(self.fc1(conv_state))
        actions = self.fc2(flattened)
        return actions

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
