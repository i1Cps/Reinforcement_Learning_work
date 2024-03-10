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
        # Set up deep network number 1, this one is for evaluating state
        self.q_network_eval = DeepQNetwork(
            env_name + "_" + algo + "_q_eval",
            self.input_dims,
            self.n_actions,
            self.learning_rate,
            chkpt_dir,
        )
        # Set up deep network number 2, this is our target network which we continuously update
        self.q_network_next = DeepQNetwork(
            env_name + "_" + algo + "_q_next",
            self.input_dims,
            self.n_actions,
            self.learning_rate,
            chkpt_dir,
        )
        self.device = self.q_network_eval.device

    def store_transition(self, state, action, reward, new_state, terminal):
        self.memory.store_transition(state, action, reward, new_state, terminal)

    def sample_memory(self):
        (
            states,
            actions,
            rewards,
            new_states,
            terminals,
        ) = self.memory.sample_buffer(self.batch_size)

        # Conver to tensors, for evalutation by network
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

    # Choose action using epsilon greedy policy:
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Add the brackets because..... we always pass an list of inputs to networks (batches), just the way it goes
            state = T.tensor(
                observation[np.newaxis, :],
                dtype=T.float,
                device=self.q_network_eval.device,
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

    def learn_(self):
        # Use observation steps to fill memory, agent will not learn until memory is filled very nicely
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_network_eval.optimizer.zero_grad()
        # We call this every time step but in the function it will only
        # replace once we reach the required learning step
        self.replace_target_network()

        # sample a batch
        (
            states,
            actions,
            rewards,
            new_states,
            terminals,
        ) = self.sample_memory()
        print("before, actions: ", actions)

        actions = T.unsqueeze(actions, 1)  # Converts to shape [32,action]
        print("actions after: ", actions)
        # Handle Loss-----------------------------------------  Look up DQN formula please
        # Look up torch.gather() cant explain in comments. Its a tensor indexing function
        Q = self.q_network_eval.forward(states).gather(1, actions.long())
        Q_next = self.q_network_next.forward(new_states).max(dim=1)[0]
        Q_next[terminals] = 0.0  # Set terminal states to value 0.0

        # in the DQN paper, loss = loss_function(reward + gamma(max_action(sate_action_target_values)) - state_action_predicted_values)
        Q_target = rewards + self.gamma * Q_next
        Q_target = T.unsqueeze(Q_target, 1)

        loss = self.q_network_eval.loss(Q_target, Q).to(self.q_network_eval.device)
        # Always zero grad before learning

        loss.backward()
        self.q_network_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_network_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_network_eval.forward(states)[indices, actions]
        q_next = self.q_network_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_network_eval.loss(q_target, q_pred).to(self.q_network_eval.device)
        loss.backward()
        self.q_network_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
