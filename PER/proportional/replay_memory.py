import numpy as np


# Class for replay buffer, dqn will use uniformed samples of its memory to iterative update its network parameters
# Think two apples, apple g and apple h, we make decisions based on apple g, but we update apple h when we learn
# something new, then after a while we set apple g to equal apple h. and begin to improve apple h again. Thats DQN
class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        """
        Initialize a replay buffer for storing transitions.

        Parameters:
        - max_size (int): Maximum capacity of the replay buffer.
        - input_shape (tuple): Shape of the state observations.

        Initializes arrays to store states, actions, rewards, new states, and terminal flags.
        """
        self.memory_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.memory_size, *input_shape), dtype=np.float32
        )
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, new_state, terminated):
        """
        Store a transition tuple in the replay buffer.

        Parameters:
        - state (array): Current state observation.
        - action (array): Action taken in the current state.
        - reward (float): Reward received after taking the action.
        - next_state (array): Next state observation after taking the action.
        - terminal (bool): Flag indicating whether the episode terminated after this transition.

        Stores the transition tuple (state, action, reward, next_state, terminal) in the replay buffer.
        """
        index = self.mem_cntr % self.memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = terminated
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Parameters:
        - batch_size (int): Number of transitions to sample.

        Returns:
        - states (array): Batch of current state observations.
        - actions (array): Batch of actions taken.
        - rewards (array): Batch of rewards received.
        - new_states (array): Batch of new state observations.
        - terminals (array): Batch of terminal flags.

        Randomly samples a batch of transitions from the replay buffer and returns them as separate arrays.
        """
        max_mem = min(self.mem_cntr, self.memory_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminals
