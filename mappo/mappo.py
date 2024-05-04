from agent import Agent
from typing import List, Any, Dict

from memory import MAPPOMemory


class MAPPO:
    def __init__(
        self,
        actor_dims: List[int],
        critic_dims: int,
        n_agents: int,
        n_actions: List[int],
        env: Any,
        n_epochs: int,
        alpha: float = 1e-4,
        fc1: int = 64,
        fc2: int = 64,
        gamma: float = 0.95,
        checkpoint_dir: str = "models/",
        scenario: str = "unclassified",
    ):
        self.agents = []
        checkpoint_dir += scenario
        for agent_idx, agent in enumerate(env.agents):
            self.agents.append(
                Agent(
                    actor_dims=actor_dims[agent_idx],
                    critic_dims=critic_dims,
                    n_actions=n_actions[agent_idx],
                    agent_idx=agent_idx,
                    alpha=alpha,
                    fc1=fc1,
                    fc2=fc2,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    checkpoint_dir=checkpoint_dir,
                    scenario=scenario,
                )
            )

    def choose_actions(self, raw_obs: Dict) -> tuple[Dict, Dict]:
        actions = {}
        probs = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            action, prob = agent.choose_action(raw_obs[agent_id])
            actions[agent_id] = action
            probs[agent_id] = prob
        return actions, probs

    def learn(self, memory: MAPPOMemory):
        for agent in self.agents:
            agent.learn(memory)

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()
