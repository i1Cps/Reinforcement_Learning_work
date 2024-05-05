from replay_memory import ReplayBuffer
import numpy as np
from networks import DeepQNetwork
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
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.fname = algo + "_" + env_name + "_lr" + str(learning_rate)
        self.algo = algo

        # Initialize the neural network used for evaluating states and choosing actions.
        # This network is trained to approximate the Q-values of state-action pairs.
        self.q_network_eval = DeepQNetwork(
            env_name + "_" + algo + "_q_eval",  # Name of the evaluation network
            self.input_dims,  # Input dimensions of the network
            self.n_actions,  # Number of actions in the action space
            self.learning_rate,  # Learning rate for the optimizer
            chkpt_dir,  # Directory to save model checkpoints
        )

        # Initialize the target network, which is used to stabilize training.
        # This network's parameters are updated less frequently to provide more stable target Q-values.
        self.q_network_next = DeepQNetwork(
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
        self.memory.store_transition(state, action, reward, new_state, terminal)

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
        (
            states,
            actions,
            rewards,
            new_states,
            terminals,
        ) = self.memory.sample_buffer(self.batch_size)

        # Convert sampled data to tensors for evaluation by the neural network
        tensor_states = T.tensor(states).to(self.q_network_eval.device)
        tensor_actions = T.tensor(actions).to(self.q_network_eval.device)
        tensor_rewards = T.tensor(rewards).to(self.q_network_eval.device)
        tensor_new_states = T.tensor(new_states).to(self.q_network_eval.device)
        tensor_terminals = T.tensor(terminals).to(self.q_network_eval.device)

        return (
            tensor_states,
            tensor_actions,
            tensor_rewards,
            tensor_new_states,
            tensor_terminals,
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

    def learn(self):
        """
        Update the Q-network based on experiences sampled from the replay memory buffer.

        This function implements the learning process outlined in the original Deep Q-Network (DQN) paper.
        It samples experiences from the replay buffer, computes target Q-values using the target Q-network,
        and updates the Q-network to minimize the Mean Squared Error (MSE) between the predicted Q-values
        and the target Q-values. The target Q-network is periodically updated to stabilize training.
        """
        # Check if enough samples are available in the replay memory buffer
        if self.memory.mem_cntr < self.batch_size:
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
        ) = self.sample_memory()

        # Handle Loss---->>>>> Look up DQN formula please

        # Compute Q-values for the sampled states and actions
        actions = T.unsqueeze(actions, 1)  # Reshape actions for indexing
        Q = self.q_network_eval.forward(states).gather(1, actions.long())

        # Compute Q-values for the next states using the target Q-network
        Q_next = self.q_network_next.forward(new_states).max(dim=1)[0]
        Q_next[terminals] = 0.0  # Set terminal states to value 0.0

        # In the DQN paper, loss = loss_function(reward + gamma(max_action(sate_action_target_values)), state_action_predicted_values)
        # Compute the target Q-values for the Bellman equation
        with T.no_grad():
            Q_target = rewards + self.gamma * Q_next

        # Calculate the MSE loss between predicted and target Q-values
        Q_target = T.unsqueeze(Q_target, 1)  # Reshape target Q-values
        loss = self.q_network_eval.loss(Q_target, Q).to(self.q_network_eval.device)

        # Backpropagate the loss and update the Q-network parameters
        loss.backward()  # Gradient descent, on weights involved in the learning
        self.q_network_eval.optimizer.step()

        # Increment the learning step counter
        self.learn_step_counter += 1

        # Decrement epsilon to decrease exploration over time
        self.decrement_epsilon()
