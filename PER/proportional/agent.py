import numpy as np
from networks import LinearDeepQNetwork
from memory import SumTree
import torch as T


class DQN:
    def __init__(
        self,
        n_actions,
        input_dims,
        mem_size,
        batch_size=32,
        gamma=0.99,
        learning_rate=0.00025,
        epsilon=1.0,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        alpha_PER=0.5,
        beta_PER=0.0,
        rebalance_iter=32,
        algo="dqn_standard",
        env_name="atari_probably",
        chkpt_dir="model_weights",
    ):
        """
        Initialize the DQN (Deep Q-Network) agent.

        Parameters:
        - n_actions (int): Number of actions in the action space.
        - input_dims (int): Dimensionality of the input observations.
        - mem_size (int): Maximum size of the replay memory buffer.
        - batch_size (int): Size of batches sampled from replay memory for training
        - gamma (float): Discount factor for future rewards
        - learning_rate (float): Learning rate for the neural network optimizer
        - epsilon (float): Initial value of epsilon for epsilon-greedy action selection
        - eps_min (float): Minimum value of epsilon
        - eps_dec (float): Epsilon decrement factor
        - replace (int): Interval for updating the target Q-network
        - algo (str): Algorithm identifier
        - env_name (str): Name of the environment
        - chkpt_dir (str): Directory to save model checkpoints
        """
        # Initialise parameters
        self.gamma = gamma
        self.learn_step_counter = 0
        self.replace_target_counter = replace
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.learning_rate = learning_rate
        self.memory_size = mem_size
        self.memory = SumTree(input_dims, mem_size, batch_size, alpha_PER, beta_PER)
        self.rebalance_iter = rebalance_iter
        self.fname = algo + "_" + env_name + "_lr" + str(learning_rate)
        self.algo = algo

        # Initialize the neural network used for evaluating states and choosing actions.
        # This network is trained to approximate the Q-values of state-action pairs.
        self.q_network_eval = LinearDeepQNetwork(
            env_name + "_" + algo + "_q_eval",  # Name of the evaluation network
            self.input_dims,  # Input dimensions of the network
            self.n_actions,  # Number of actions in the action space
            self.learning_rate,  # Learning rate for the optimizer
            chkpt_dir,  # Directory to save model checkpoints
        )

        # Initialize the target network, which is used to stabilize training.
        # This network's parameters are updated less frequently to provide more stable target Q-values.
        self.q_network_next = LinearDeepQNetwork(
            env_name + "_" + algo + "_q_next",  # Name of the target network
            self.input_dims,  # Input dimensions of the network
            self.n_actions,  # Number of actions in the action space
            self.learning_rate,  # Learning rate for the optimizer
            chkpt_dir,  # Directory to save model checkpoints
        )

        # Set the device for tensor computations based on the device used by the evaluation network.
        self.device = self.q_network_eval.device

    def store_transition(self, state, action, reward, new_state, terminal):
        """
        Store a transition tuple in memory buffer.

        Parameters:
        - state (array): Current state.
        - action (array): Action taken.
        - reward (float): Reward received.
        - next_state (array): Next state after taking action.
        - terminal (bool): Flag indicating terminal state.
        """

        # Store experience in memory buffer
        self.memory.store_transition([state, action, reward, new_state, terminal])

    def sample_memory(self):
        """
        Sample a batch of transitions from the replay memory buffer.

        Returns:
        - tensor_states (tensor): Batch of states.
        - tensor_actions (tensor): Batch of actions.
        - tensor_rewards (tensor): Batch of rewards.
        - tensor_new_states (tensor): Batch of new states.
        - tensor_terminals (tensor): Batch of terminal flags.
        """
        # Sample transitions from the replay memory buffer

        sarsd, sample_idx, weights = self.memory.sample()

        states, actions, rewards, new_states, terminals = sarsd

        # Convert sampled data to tensors for evaluation by the neural network
        tensor_states = T.tensor(states, dtype=T.float).to(self.q_network_eval.device)
        tensor_actions = T.tensor(actions).to(self.q_network_eval.device)
        tensor_rewards = T.tensor(rewards, dtype=T.float).to(self.q_network_eval.device)
        tensor_new_states = T.tensor(new_states, dtype=T.float).to(
            self.q_network_eval.device
        )
        tensor_terminals = T.tensor(terminals).to(self.q_network_eval.device)

        weights = T.tensor(weights, dtype=T.float).to(self.q_network_eval.device)

        return (
            tensor_states,
            tensor_actions,
            tensor_rewards,
            tensor_new_states,
            tensor_terminals,
            sample_idx,
            weights,
        )

    def choose_action(self, observation):
        """
        Choose an action using an epsilon-greedy policy based on the observation.

        Parameters:
        - observation (array): Current state observation.

        Returns:
        - action (int): Action selected by the policy.
        """
        if np.random.random() > self.epsilon:
            # Exploit: Select the action with the highest Q-value according to the evaluation network
            state = T.tensor(
                observation[
                    np.newaxis, :
                ],  # Convert shape: (batch_size) -> shape: (1,batch_size)
                dtype=T.float,
                device=self.q_network_eval.device,
            )
            actions = self.q_network_eval.forward(state)
            action = T.argmax(actions).item()  # Select action with the highest Q-value
        else:
            # Explore: Select a random action
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        """
        Update the parameters of the target Q-network to match those of the evaluation Q-network.
        This is performed periodically to stabilize training.
        """
        if self.learn_step_counter % self.replace_target_counter == 0:
            # Load the parameters of the evaluation network into the target network
            self.q_network_next.load_state_dict(self.q_network_eval.state_dict())

    def _learn(self):
        if not self.memory.ready():
            return

        self.q_network_eval.optimizer.zero_grad()

        self.replace_target_network()

        (
            states,
            actions,
            rewards,
            states_,
            dones,
            sample_idx,
            weights,
        ) = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_network_eval.forward(states)[indices, actions]
        q_next = self.q_network_next.forward(states_)
        q_eval = self.q_network_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        td_error = np.abs(
            (q_target.detach().cpu().numpy() - q_pred.detach().cpu().numpy())
        )
        td_error = np.clip(td_error, 0.0, 1.0)

        self.memory.update_priorities(sample_idx, td_error)

        q_target *= weights
        q_pred *= weights

        loss = self.q_network_eval.loss(q_target, q_pred).to(self.q_network_eval.device)
        loss.backward()
        self.q_network_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

    def learn(self):
        """
        Update the Q-network based on experiences sampled from the replay memory buffer.

        This function implements the learning process outlined in the original Deep Q-Network (DQN) paper.
        It samples experiences from the replay buffer, computes target Q-values using the target Q-network,
        and updates the Q-network to minimize the Mean Squared Error (MSE) between the predicted Q-values
        and the target Q-values. The target Q-network is periodically updated to stabilize training.
        """
        # Check if enough samples are available in the replay memory buffer
        if not self.memory.ready():
            return

        # Reset gradients before backpropagation
        self.q_network_eval.optimizer.zero_grad()

        # Update the target Q-network periodically
        self.replace_target_network()

        # Sample a batch of experiences from the replay memory buffer
        (
            states,
            actions,
            rewards,
            new_states,
            terminals,
            samples_idx,
            weights,
        ) = self.sample_memory()

        indices = np.arange(self.batch_size)
        Q = self.q_network_eval.forward(states)[indices, actions]
        Q_eval = self.q_network_eval(new_states)  # Untouched for loss func and td error
        max_actions = T.argmax(Q_eval, dim=1)

        Q_next = self.q_network_next.forward(new_states)
        Q_next[terminals] = 0.0

        Q_target = rewards + self.gamma * Q_next[indices, max_actions]

        # Calculate TD error for prioritized replay
        td_error = np.abs((Q_target.detach().cpu().numpy() - Q.detach().cpu().numpy()))

        td_error = np.clip(td_error, 0.0, 1.0)

        self.memory.update_priorities(samples_idx, td_error)

        Q_target *= weights
        Q *= weights

        # Calculate the MSE loss between predicted and target Q-values
        # Q_target = T.unsqueeze(Q_target, 1)  # Reshape target Q-values
        loss = self.q_network_eval.loss(Q_target, Q).to(self.q_network_eval.device)

        # Backpropagate the loss and update the Q-network parameters
        loss.backward()  # Gradient descent, on weights involved in the learning
        self.q_network_eval.optimizer.step()

        # Increment the learning step counter
        self.learn_step_counter += 1

        # Decrement epsilon to decrease exploration over time
        self.decrement_epsilon()

    # Agents policy gradually become less stochastic
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
