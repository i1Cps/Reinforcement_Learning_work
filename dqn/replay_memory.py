import numpy as np


# Class for replay buffer, dqn will use uniformed samples of its memory to iterative update its network parameters
# Think two apples, apple g and apple h, we make decisions based on apple g, but we update apple h when we learn
# something new, then after a while we set apple g to equal apple h. and begin to improve apple h again. Thats DQN
class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        # Size of replay buffer
        print(input_shape)
        self.memory_size = max_size
        # Pointer to keep track of where we need to add memory in the buffer
        self.mem_cntr = 0
        # Different numpy arrays of types of memory we will store, state, reward etc
        self.state_memory = np.zeros(
            (self.memory_size, *input_shape), dtype=np.float32
        )  # use '*' to make input_shape ambigious
        self.new_state_memory = np.zeros(
            (self.memory_size, *input_shape), dtype=np.float32
        )  # PyTorch can be funny when not using float32
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, new_state, terminated):
        index = self.mem_cntr % self.memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = terminated
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.memory_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminals
