# Import necessary libraries
import numpy as np
import gymnasium as gym

# Import custom classes and functions
from utils import plot_learning_curve
from agent import DQN


def clip_reward(r):
    if r > 1:
        return 1
    elif r < -1:
        return -1
    else:
        return r


if __name__ == "__main__":
    # Initialize the environment
    env = gym.make("CartPole-v0")

    # Initialize variables
    best_score = -np.inf  # Initialize the best score seen so far
    load_checkpoint = False  # Flag indicating whether to load a pre-trained model
    n_games = 500  # Number of games to play
    rebalance_iter = 64
    alpha_PER = 0.25
    beta_PER = 0.5
    replace = 250

    # Initialize the DQN agent with specific parameters
    dqn = DQN(
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        mem_size=20 * 1024,  # Size of the replay memory
        batch_size=64,  # Size of mini-batches sampled from memory
        gamma=0.99,  # Discount factor
        learning_rate=2.5e-4,  # Learning rate for the neural network
        epsilon=1,  # Initial value of epsilon for epsilon-greedy exploration
        eps_dec=1e-4,  # Epsilon decay rate
        eps_min=0.01,  # Minimum value of epsilon
        replace=replace,  # Frequency of updating target network
        alpha_PER=alpha_PER,  # PER variable
        beta_PER=beta_PER,  # PER variable
        rebalance_iter=rebalance_iter,
        env_name="cartpole-v0",  # Name of the environment
    )

    # Load pre-trained models if available
    if load_checkpoint:
        dqn.load_models()

    # Define filename for saving the learning curve plot
    figure_file = "plots/" + "__" + str(n_games) + "_games" + ".png"

    # Initialize lists to store scores, steps, and epsilon values
    scores, steps_array, eps_history = [], [], []

    # Main loop over the specified number of games
    n_steps = 0  # Initialize the total number of steps
    for i in range(n_games):
        # Reset the environment for each new game and get the initial observation
        score = 0  # Initialize the score for the current episode
        terminal = False
        truncated = False
        observation, info = env.reset()

        # Loop within each game until the episode is complete
        while not (terminal or truncated):
            # Choose an action based on the current observation
            action = dqn.choose_action(observation)

            # Take a step in the environment with the chosen action
            observation_, reward, terminal, truncated, info = env.step(action)
            score += reward  # Accumulate the reward to compute the total score
            r = clip_reward(reward)

            # Store the experience tuple in the agent's memory
            dqn.store_transition(
                observation, action, r, observation_, terminal or truncated
            )

            # Learn from the experiences stored in the memory
            dqn.learn()

            # Update the current observation for the next step
            observation = observation_
            n_steps += 1  # Increment the total number of steps

        # Append the total score of the current episode to the scores list
        scores.append(score)
        steps_array.append(n_steps)

        # Calculate the average score over the last 100 episodes
        avg_score = np.mean(scores[-100:])

        # Print the progress of training after each episode
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

        # Save the model if the average score surpasses the best score seen so far
        if avg_score > best_score:
            if not load_checkpoint:
                dqn.save_models()  # Save the model if it improves the best score
            best_score = avg_score

        # Store the current epsilon value
        eps_history.append(dqn.epsilon)

    # Plot the learning curve after all episodes have been executed
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
