import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # Helper function to calculate output of 2D conv
        conv_shape = self.calc_conv_output(input_dims)

        self.gru = nn.GRUCell(conv_shape, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def calc_conv_output(self, input_dims):
        state = T.zeros(1, *input_dims)  # add the 1 for dimensianlity

        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        dims = self.conv4(dims)
        return int(np.prod(dims.size()))

    def forward(self, state, hx):
        conv = F.elu(self.conv1(state))
        conv = F.elu(self.conv2(state))
        conv = F.elu(self.conv3(state))
        conv = F.elu(self.conv4(state))

        conv_state = conv.view((conv.size()[0], -1))
        hx = self.gru(conv_state, (hx))

        pi = self.pi(hx)
        v = self.v(hx)

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy()[0], v, log_prob, hx

    def calculate_R(self, done, rewards, values):
        # Convert values:List[tensor] into a tensor
        values = T.cat(values).squeeze()

        # Either a batch of states or single state
        if len(values.size()) == 1:
            R = values[-1] * (1 - int(done))
        elif len(values.size()) == 0:
            R = values * (1 - int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float).reshape(values.size())

        return batch_return

    def calculate_cost(self, new_state, hx, done, rewards, values, log_probs):
        returns = self.calculate_R(done, rewards, values)

        next_v = (
            T.zeros(1, 1)
            if done
            else self.forward(T.tensor([new_state], dtype=T.float), hx)[1]
        )

        values.append(next_v.detach())
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)

        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps - t):
                temp = (self.gamma * self.tau) ** k * delta_t[t + k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)

        actor_loss = -(log_probs * gae).sum()
        # if single then values is rank 1 and return rank 0
        #  want to have same shape to avoid a warninggggg
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)

        entrophy_loss = (-log_probs * T.exp(log_probs)).sum()

        total_loss = actor_loss = 0.01 * entrophy_loss
        return total_loss
