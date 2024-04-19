import numpy as np
import gymnasium
from agent import Agent
from utils import plot_learning_curve

if __name__ == "__main__":
    env = gymnasium.make("LunarLanderContinuous-v2")
    agent = Agent(
        alpha=3e-4,
        beta=3e-4,
        input_dims=env.observation_space.shape,
        tau=0.005,
        batch_size=100,
        fc1_dims=400,
        fc2_dims=300,
        n_actions=env.action_space.shape[0],
        max_actions=env.action_space.high[0],
    )

    n_games = 2000
    filename = (
        "LunarLander_alpha_"
        + str(agent.alpha)
        + "_beta_"
        + str(agent.beta)
        + "_"
        + str(n_games)
        + "_games"
    )
    figure_file = "plots/" + filename + ".png"

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation, info = env.reset()
        terminal = False
        truncated = False
        score = 0
        agent.noise.reset()
        while not (terminal or truncated):
            action = agent.choose_action(observation)
            new_observation, reward, terminal, truncated, info = env.step(action)
            agent.store_transition(
                observation, action, reward, new_observation, terminal or truncated
            )
            agent.learn()
            score += reward
            observation = new_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print("episode ", i, "score %.1f" % score, "average score %.1f" % avg_score)

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
