from typing import List, Dict, Any
import numpy as np
from mappo import MAPPO
from memory import MAPPOMemory
from utils import obs_list_to_state_vector
from pettingzoo.mpe import simple_speaker_listener_v4


def run():
    env_id = "Simple_Speaker_Listener"
    parallel_env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
    _, _ = parallel_env.reset()
    N = 2048
    batch_size = 64
    n_epochs = 10
    alpha = 3e-4
    scenario = "simple_speaker_listener"

    n_agents = parallel_env.max_num_agents

    actor_dims = [
        parallel_env.observation_space(agent).shape[0] for agent in parallel_env.agents
    ]
    n_actions = [
        parallel_env.action_space(agent).shape[0] for agent in parallel_env.agents
    ]
    critic_dims = sum(actor_dims)

    mappo_agents = MAPPO(
        actor_dims=actor_dims,
        critic_dims=critic_dims,
        n_agents=n_agents,
        n_actions=n_actions,
        n_epochs=n_epochs,
        env=parallel_env,
        gamma=0.95,
        alpha=alpha,
        scenario=scenario,
    )

    memory = MAPPOMemory(
        batch_size=batch_size,
        T=N,
        n_agents=n_agents,
        agents=parallel_env.agents,
        critic_dims=critic_dims,
        actor_dims=actor_dims,
        n_actions=n_actions,
    )

    MAX_STEPS = 1_000_000
    total_steps = 0
    episode = 1
    traj_length = 0
    score_history, steps_history = [], []

    while total_steps < MAX_STEPS:
        obs, _ = parallel_env.reset()
        terminal = [False] * n_agents
        score = 0
        while not any(terminal):
            actions, prob = mappo_agents.choose_actions(obs)
            next_obs, reward, done, trunc, info = parallel_env.step(actions)

            total_steps += 1
            traj_length += 1

            list_obs = list(obs.values())
            list_actions = list(actions.values())
            list_probs = list(prob.values())
            list_reward = list(reward.values())
            list_next_obs = list(next_obs.values())
            list_trunc = list(trunc.values())
            list_terminated = list(done.values())

            state = obs_list_to_state_vector(list_obs)
            next_state = obs_list_to_state_vector(list_next_obs)
            terminal = [d or t for d, t in zip(list_terminated, list_trunc)]

            score += sum(list_reward)

            memory.store_memory(
                raw_obs=list_obs,
                state=state,
                action=list_actions,
                prob=list_probs,
                reward=list_reward,
                next_raw_obs=list_next_obs,
                next_state=next_state,
                terminal=any(terminal),
            )

            if traj_length % N == 0:
                mappo_agents.learn(memory)
                traj_length = 0
                memory.clear_memory()
            obs = next_obs

        score_history.append(score)
        steps_history.append(total_steps)
        avg_score = np.mean(score_history[-100:])
        print(
            f"{env_id} Episode {episode} total steps {total_steps}"
            f" avg score {avg_score :.1f}"
        )

        episode += 1

    np.save("data/mappo_scores.npy", np.array(score_history))
    np.save("data/mappo_steps.npy", np.array(steps_history))


if __name__ == "__main__":
    run()
