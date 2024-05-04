from agent import Agent
from memory import MultiAgentReplayBuffer
from typing import List, Dict, Any


class MADDPG:
    def __init__(
        self,
        actor_dims: List[int],
        critic_dims: int,
        n_agents: int,
        n_actions: List[int],
        env: Any,
        alpha: float = 1e-4,
        beta: float = 1e-3,
        fc1: int = 64,
        fc2: int = 64,
        gamma: float = 0.95,
        tau: float = 0.01,
        checkpoint_dir: str = "models",
        scenario: str = "unclassified",
    ):
        self.agents = []
        for agent_idx in range(n_agents):
            agent = list(env.action_spaces.keys())[agent_idx]
            min_action = env.action_space(agent).low
            max_action = env.action_space(agent).high
            self.agents.append(
                Agent(
                    actor_dims=actor_dims[agent_idx],
                    critic_dims=critic_dims,
                    n_actions=n_actions[agent_idx],
                    agent_idx=agent_idx,
                    alpha=alpha,
                    beta=beta,
                    fc1=fc1,
                    fc2=fc2,
                    tau=tau,
                    gamma=gamma,
                    min_action=min_action,
                    max_action=max_action,
                    checkpoint_dir=checkpoint_dir,
                    scenario=scenario,
                )
            )

    def choose_actions(self, raw_obs: Dict, eval: bool = False) -> Dict:
        actions = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            action = agent.choose_action(raw_obs[agent_id], eval)
            actions[agent_id] = action
        return actions

    def learn(self, memory: MultiAgentReplayBuffer):
        for agent in self.agents:
            agent.learn(memory, self.agents)

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()
