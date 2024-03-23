import numpy as np
from utils import plot_learning_curve, make_env
from agent import DQN


if __name__ == "__main__":
    env = make_env("PongNoFrameskip-v4")

    # Negative scoring game,
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500
    dqn = DQN(
        n_actions=env.action_space.n,
        input_dims=(env.observation_space.shape),
        mem_size=50000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.00001,
        epsilon=1,
        eps_dec=1e-5,
        eps_min=0.01,
        replace=1000,
        env_name="Pong-v4",
    )

    if load_checkpoint:
        dqn.load_models()

    figure_file = "plots/" + "__" + str(n_games) + "_games" + ".png"

    n_steps = 0
    scores, steps_array, eps_history = [], [], []

    for i in range(n_games):
        # For each game reset the score and termination variables
        score = 0
        observation, info = env.reset()
        terminal = False
        truncated = False

        # While episode is not done
        while not (terminal or truncated):
            action = dqn.choose_action(observation)
            observation_, reward, terminal, truncated, info = env.step(action)
            score += reward

            # Store each step as an experience to train our networks
            dqn.store_transition(
                observation, action, reward, observation_, terminal or truncated
            )

            # Learn every time step
            dqn.learn()
            # Update observation
            observation = observation_
            n_steps += 1

        # At the end of every episode print scores, average scores and current epsilon
        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])
        print(
            "episode: ",
            i,
            "score: ",
            score,
            " average score %.1f" % avg_score,
            "best score %.2f" % best_score,
            "epsilon %.2f" % dqn.epsilon,
            "steps",
            n_steps,
        )

        # Save the model if our average scores surpasses our best score
        if avg_score > best_score:
            if not load_checkpoint:
                dqn.save_models()
            best_score = avg_score

        eps_history.append(dqn.epsilon)

    # After the model has executed every episode plot the learning curve
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
