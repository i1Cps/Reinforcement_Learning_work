import torch as T
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        tau,
        env,
        gamma=0.99,
        update_actor_interval=2,
        warmup=1000,
        n_actions=2,
        max_size=1000000,
        layer1_size=400,
        layer2_size=300,
        batch_size=100,
        noise=0.1,
    ):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="actor",
        )
        self.critic_1 = CriticNetwork(
            beta, input_dims, layer1_size, layer2_size, n_actions, name="critic_1"
        )
        self.critic_2 = CriticNetwork(
            beta, input_dims, layer1_size, layer2_size, n_actions, name="critic_2"
        )

        self.target_actor = ActorNetwork(
            alpha, input_dims, layer1_size, layer2_size, n_actions, name="target_actor"
        )
        self.target_critic_1 = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions,
            name="target_critic_1",
        )
        self.target_critic_2 = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions,
            name="target_critic_2",
        )

        self.noise = noise
        self.update_network_parameters(tau=1)

    # We pick randomly until warmup is finished
    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(
                np.random.normal(scale=self.noise, size=(self.n_actions,)),
                device=self.actor.device,
            )
        else:
            state = T.tensor(observation, dtype=T.float, device=self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(
            np.random.normal(scale=self.noise), dtype=T.float, device=self.actor.device
        )  # naive normal noise, we like OU or pink noise
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, terminal):
        self.memory.store_transition(state, action, reward, new_state, terminal)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, new_states, terminals = self.memory.sample_memory(
            self.batch_size
        )

        states = T.tensor(states, dtype=T.float, device=self.critic_1.device)
        actions = T.tensor(actions, dtype=T.float, device=self.critic_1.device)
        rewards = T.tensor(rewards, dtype=T.float, device=self.critic_1.device)
        new_states = T.tensor(new_states, dtype=T.float, device=self.critic_1.device)
        terminals = T.tensor(terminals, device=self.critic_1.device)

        # This is the clipping + noise mentioned in the TD3 paper to restrict overestimation bias
        noise = (T.randn_like(actions) * 0.2).clamp(-0.5, 0.5)

        next_actions = (self.target_actor.forward(new_states) + noise).clamp(
            self.min_action[0], self.max_action[0]
        )

        # might break if elements of min and max are not equal

        # Below is the crux of the double dueling technique first established in DDQN
        q1_next = self.target_critic_1.forward(new_states, next_actions)
        q2_next = self.target_critic_2.forward(new_states, next_actions)

        # target_Q1 = self.target_critic_1.forward(new_states, next_actions)
        # target_Q2 = self.target_critic_2.forward(new_states, next_actions)
        # target_Q = torch.min(target_Q1,target_Q2)
        # target_Q = rewards + self.gamma * target_Q

        q1_next[terminals] = 0.0
        q2_next[terminals] = 0.0
        print("version 1: ", q1_next)

        q1_next = q1_next.view(-1)
        print("version 2: ", q1_next)

        q2_next = q2_next.view(-1)

        critic_value_next = T.min(q1_next, q2_next)

        target = rewards + self.gamma * critic_value_next
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1 = self.critic_1.forward(states, actions)
        q2 = self.critic_2.forward(states, actions)

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss

        # Optimize the critic
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_iter != 0:
            return

        # Delayed policy updates
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(states, self.actor.forward(states))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Parameters for Actor and Critic Networks
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        # Parameters for target Actorn and Critic Networks
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        # State dictionary for actor and critic parameters
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)

        # State dictionary for target actor and critic parameters
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = (
                tau * critic_1_state_dict[name].clone()
                + (1 - tau) * target_critic_1_state_dict[name].clone()
            )

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = (
                tau * critic_2_state_dict[name].clone()
                + (1 - tau) * target_critic_2_state_dict[name].clone()
            )

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
