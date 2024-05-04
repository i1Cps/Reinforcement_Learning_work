from pathlib import Path
import numpy as np
from maddpg import MADDPG
from memory import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_speaker_listener_v4


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def run():
    parallel_env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)

    # Hyperparameters
    MEMORY_SIZE = 1_000_000
    BATCH_SIZE = 1024
    ALPHA = 1e-4
    BETA = 1e-3
    GAMMA = 0.95
    TAU = 0.01
    EVAL_INTERVAL = 1000
    MAX_STEPS = 1_000_000

    scenario = "simple_speaker_listener"
    _, _ = parallel_env.reset()
    n_agents = parallel_env.max_num_agents

    actor_dims = [
        parallel_env.observation_space(agent).shape[0] for agent in parallel_env.agents
    ]
    n_actions = [
        parallel_env.action_space(agent).shape[0] for agent in parallel_env.agents
    ]
    critic_dims = sum(actor_dims) + sum(n_actions)

    maddpg_agents = MADDPG(
        actor_dims=actor_dims,
        critic_dims=critic_dims,
        n_agents=n_agents,
        n_actions=n_actions,
        env=parallel_env,
        gamma=GAMMA,
        alpha=ALPHA,
        beta=BETA,
        tau=TAU,
        checkpoint_dir="data/models",
        scenario=scenario,
    )
    critic_dims = sum(actor_dims)

    memory = MultiAgentReplayBuffer(
        max_size=MEMORY_SIZE,
        critic_dims=critic_dims,
        actor_dims=actor_dims,
        n_actions=n_actions,
        n_agents=n_agents,
        batch_size=BATCH_SIZE,
    )

    total_steps = 0
    episode = 0
    eval_scores = []
    eval_steps = []

    score = evaluate(maddpg_agents, parallel_env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)

    while total_steps < MAX_STEPS:
        obs, _ = parallel_env.reset()
        terminal = [False] * n_agents
        while not any(terminal):
            actions = maddpg_agents.choose_actions(obs)
            next_obs, reward, done, truncated, info = parallel_env.step(actions)

            list_done = list(done.values())
            list_obs = list(obs.values())
            list_reward = list(reward.values())
            list_actions = list(actions.values())
            list_next_obs = list(next_obs.values())
            list_truncated = list(truncated.values())

            state = obs_list_to_state_vector(list_obs)
            next_state = obs_list_to_state_vector(list_next_obs)

            terminal = [d or t for d, t in zip(list_done, list_truncated)]
            memory.store_transition(
                raw_obs=list_obs,
                state=state,
                action=list_actions,
                reward=list_reward,
                next_raw_obs=list_next_obs,
                next_state=next_state,
                done=terminal,
            )

            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)

            obs = next_obs
            total_steps += 1

        if total_steps % EVAL_INTERVAL == 0:
            score = evaluate(maddpg_agents, parallel_env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)

        episode += 1

    data_dir = Path("")
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save("data/raw_data/maddpg_score.npy", np.array(eval_scores))
    np.save("data/raw_data/maddpg_steps.npy", np.array(eval_steps))


def evaluate(agents: MADDPG, env, ep: int, step: int, n_eval: int = 3):
    score_history = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        score = 0
        terminal = [False] * env.max_num_agents
        while not any(terminal):
            actions = agents.choose_actions(obs, eval=True)
            next_obs, rewards, done, truncated, info = env.step(actions)

            list_truncated = list(truncated.values())
            list_reward = list(rewards.values())
            list_done = list(done.values())

            terminal = [d or t for d, t in zip(list_done, list_truncated)]

            obs = next_obs
            score += sum(list_reward)
        score_history.append(score)
    avg_score = np.mean(score_history)
    print(f"Evaluation episode {ep} train steps {step} average score {avg_score:.1f}")
    return avg_score


if __name__ == "__main__":
    run()
