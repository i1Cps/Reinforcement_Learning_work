from .replay_memory import ReplayBuffer
from .networks import DeepQNetwork
import numpy as np
import torch as T


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        learning_rate,
        n_actions,
        input_dims,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        algo=None,
        env_name=None,
        chkpt_dir="tmp/dqn",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_counter = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        # Set up deep network number 1, this one is for evaluating state
        self.q_network_eval = DeepQNetwork(
            self.env_name + "_" + self.algo + "_q_eval",
            self.input_dims,
            self.n_actions,
            self.learning_rate,
            self.chkpt_dir,
        )
        # Set up deep network number 2, this is our target network which we continuously update
        self.q_network_next = DeepQNetwork(
            self.env_name + "_" + self.algo + "_q_next",
            self.input_dims,
            self.n_actions,
            self.learning_rate,
            self.chkpt_dir,
        )

    # Choose action using epsilon greedy policy:
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Add the brackets because..... we always pass an list of inputs to networks (batches), just the way it goes
            state = T.tensor([observation], dtype=T.float).to(
                self.q_network_eval.device
            )
            actions = self.q_network_eval.forward(state)
            # Pick the best action with the biggest probability
            action = T.argmax(actions).item()  # dereference returned tensor with .item
        else:
            action = np.random.choice(self.action_space)
        return action

    # This function just updates the target network to be the same as the eval network
    # Basic principle of DQN
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_counter == 0:
            self.q_network_next.load_state_dict(self.q_network_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def save_models(self):
        self.q_network_eval.save_checkpoint()
        self.q_network_next.save_checkpoint()

    def load_models(self):
        self.q_network_eval.load_checkpoint()
        self.q_network_next.load_checkpoint()
