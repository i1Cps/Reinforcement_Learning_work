from agent import Agent
from memory import MultiAgentReplayBuffer
from typing import List, Dict, Any


class MADDPG:
    def __init__(
        self,
        actor_input_dims: List[int],
        critic_input_dims: int,
        n_agents: int,
        n_actions: List[int],
        env: Any,
        alpha: float,
        beta: float,
        gamma: float,
        tau: float,
        actor_fc1: int = 256,
        actor_fc2: int = 256,
        critic_fc1: int = 256,
        critic_fc2: int = 256,
    ):
        # Handle agent instances in a list
        self.agents = []
        for agent_idx in range(n_agents):
            agent = list(env.action_spaces.keys())[agent_idx]
            min_actions = env.action_space(agent).low
            max_actions = env.action_space(agent).high
            self.agents.append(
                Agent(
                    actor_dims=actor_input_dims[agent_idx],
                    critic_dims=critic_input_dims,
                    alpha=alpha,
                    beta=beta,
                    tau=tau,
                    gamma=gamma,
                    agent_idx=agent_idx,
                    n_actions=n_actions[agent_idx],
                    min_actions=min_actions,
                    max_actions=max_actions,
                    actor_fc1=actor_fc1,
                    actor_fc2=actor_fc2,
                    critic_fc1=critic_fc1,
                    critic_fc2=critic_fc2,
                )
            )

    # Choose actions for each agent
    def choose_actions(self, raw_obs: Dict, eval=False) -> Dict:
        actions = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            action = agent.choose_action(raw_obs[agent_id], eval=False)
            actions[agent_id] = action
        return actions

    # Choose random action for each agent
    def choose_random_actions(self) -> Dict:
        random_actions = {}
        for agent_idx, agent in self.agents:
            action = agent.choose_random_actions()
            random_actions[agent_idx] = action
        print("maddpg random actions: {}".format(random_actions))
        return random_actions

    def learn(self, memory: MultiAgentReplayBuffer):
        for agent in self.agents:
            agent.learn(memory, self.agents)

    def save(self, filepath):
        for agent in self.agents:
            agent.save(filepath)

    def load(self, filepath):
        for agent in self.agents:
            agent.load(filepath)
