import numpy as np
import torch as T
from agent import Agent
import gymnasium as gym
from utils import plot_learning_curve


def action_adapter(a, max_a):
    return 2 * (a - 0.5) * max_a


def clip_reward(x):
    if x < -1:
        return -1
    elif x > 1:
        return 1
    else:
        return x


if __name__ == "__main__":
    env_id = "BipedalWalker-v3"
    seed = 0
    T.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_id)

    N = 2048
    batch_size = 64
    n_epochs = 10
    alpha = 3e-4
    max_steps = 1_000_000
    max_action = env.action_space.high[0]

    agent = Agent(
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.shape[0],
        gamma=0.99,
        alpha=alpha,
    )

    filename = (
        "bipedal_walker_"
        + str(agent.alpha)
        + "N_epochs_"
        + str(agent.n_epochs)
        + "_"
        + str(max_steps)
        + "_games"
    )
    figure_file = "plots/" + filename + ".png"
    score_history = []
    total_steps = 0
    trajectory_len = 0
    episode = 1

    while total_steps < max_steps:
        observation, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        score = 0
        while not (terminated or truncated):
            action, prob = agent.choose_action(observation)
            adapted_action = action_adapter(action, max_action)
            next_observation, reward, terminated, truncated, info = env.step(
                adapted_action
            )
            r = clip_reward(reward)
            score += reward
            agent.remember(
                observation,
                action,
                reward,
                next_observation,
                (terminated or truncated),
                prob,
            )
            total_steps += 1
            trajectory_len += 1
            if trajectory_len % N == 0:
                agent.learn()
                trajectory_len = 0
            observation = next_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print(
            "{} Episode {} total steps {}  avg_score {:.1f}".format(
                env_id, episode, total_steps, avg_score
            )
        )
        episode += 1
    x = [i + 1 for i in range(episode)]
    plot_learning_curve(x, score_history, figure_file)
