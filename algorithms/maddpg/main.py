from pathlib import Path
import numpy as np
from maddpg import MADDPG
from memory import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_speaker_listener_v4
from utils import plot_learning_curve


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
    LOAD_MODEL = False
    scenario = "simple_speaker_listener"

    # Initialise the environment wrapper with .reset() call
    _, _ = parallel_env.reset()

    # Get number of agents configured for the environment
    n_agents = parallel_env.max_num_agents

    # A list of input state spaces for each agent actor network
    actor_state_dims = [
        parallel_env.observation_space(agent).shape[0] for agent in parallel_env.agents
    ]

    # A list of each agents action space
    n_actions = [
        parallel_env.action_space(agent).shape[0] for agent in parallel_env.agents
    ]

    # MADDPG ~ The state space for the critic network is the sum of every actors state dimension (READ THE PAPER)
    critic_state_dims = sum(actor_state_dims)

    # MADDPG ~ The input dims for the critic network is the sum of actor states + sum of action spaces
    critic_input_dims = critic_state_dims + sum(n_actions)

    # Class to handle our agents
    maddpg_agents = MADDPG(
        actor_input_dims=actor_state_dims,
        critic_input_dims=critic_input_dims,
        n_agents=n_agents,
        n_actions=n_actions,
        env=parallel_env,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        tau=TAU,
        actor_fc1=256,
        actor_fc2=256,
        critic_fc1=256,
        critic_fc2=256,
    )

    # Centralised memory buffer
    memory = MultiAgentReplayBuffer(
        max_size=MEMORY_SIZE,
        critic_state_dims=critic_state_dims,
        actor_state_dims=actor_state_dims,
        n_actions=n_actions,
        n_agents=n_agents,
        batch_size=BATCH_SIZE,
    )

    file_path = (
        scenario
        + "_"
        + str(ALPHA)
        + "_learning_rate_"
        + str(BATCH_SIZE)
        + "_batch_size_"
        + str(MAX_STEPS)
        + "_total_steps"
    )

    model_file_dir = Path("model_weights") / file_path
    model_file_dir.mkdir(parents=True, exist_ok=True)

    if LOAD_MODEL:
        maddpg_agents.load(model_file_dir)

    total_steps = 0
    episode = 0
    eval_scores = []
    eval_steps = []
    best_score = -np.inf

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
                single_obs=list_obs,
                global_obs=state,
                actions=list_actions,
                rewards=list_reward,
                next_single_obs=list_next_obs,
                next_global_obs=next_state,
                dones=terminal,
            )

            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)

            obs = next_obs
            total_steps += 1

        if total_steps % EVAL_INTERVAL == 0:
            score = evaluate(maddpg_agents, parallel_env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)

            if score > best_score:
                best_score = score
                maddpg_agents.save(model_file_dir)

        episode += 1

    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file_path = plot_dir / (file_path + ".png")
    plot_learning_curve(eval_steps, eval_scores, plot_file_path)


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
