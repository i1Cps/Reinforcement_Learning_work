import numpy as np
import gymnasium
from agent import Agent
from utils import plot_learning_curve


N_GAMES = 2000
ALPHA = 3e-4
BETA = 3e-4
TAU = 0.005
BATCH_SIZE = 100
RANDOM_STEPS = 5000


if __name__ == "__main__":
    env = gymnasium.make("LunarLanderContinuous-v2")
    agent = Agent(
        alpha=ALPHA,
        beta=BETA,
        input_dims=env.observation_space.shape,
        tau=TAU,
        batch_size=BATCH_SIZE,
        fc1_dims=400,
        fc2_dims=300,
        n_actions=env.action_space.shape[0],
        max_actions=env.action_space.high[0],
    )

    filename = (
        "LunarLander_alpha_"
        + str(agent.alpha)
        + "_beta_"
        + str(agent.beta)
        + "_"
        + str(N_GAMES)
        + "_games"
    )
    figure_file = "plots/" + filename + ".png"

    best_score = env.reward_range[0]
    score_history = []
    total_steps = 0
    for i in range(N_GAMES):
        observation, info = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            if total_steps < RANDOM_STEPS:
                action = agent.choose_random_action()
            else:
                action = agent.choose_action(observation)
            new_observation, reward, terminal, truncated, info = env.step(action)
            total_steps += 1
            done = terminal or truncated
            agent.store_transition(observation, action, reward, new_observation, done)
            if total_steps >= RANDOM_STEPS:
                agent.learn()
            score += reward
            observation = new_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print("episode ", i, "score %.1f" % score, "average score %.1f" % avg_score)

    x = [i + 1 for i in range(N_GAMES)]
    plot_learning_curve(x, score_history, figure_file)
