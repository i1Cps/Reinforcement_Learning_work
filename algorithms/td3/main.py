# Import necessary libraries
import numpy as np
import gymnasium

# Import custom classes and functions
from agent import Agent
from utils import plot_learning_curve

if __name__ == "__main__":
    # Initialize the environment
    env = gymnasium.make("BipedalWalker-v3")

    # Initialize the agent with specific parameters
    agent = Agent(
        alpha=0.001,  # Actor network learning rate
        beta=0.001,  # Critic network learning rate
        input_dims=env.observation_space.shape,  # Observation space dimensions
        tau=0.005,  # Parameter for soft update of target networks
        env=env,  # environment
        batch_size=100,  # Size of mini-batches sampled from memory
        layer1_size=400,  # Size of the first hidden layer in neural networks
        layer2_size=300,  # Size of the second hidden layer in neural networks
        n_actions=env.action_space.shape[0],  # Number of possible actions
    )

    # Define the number of games to be played
    n_games = 1500

    # Define filename for saving the learning curve plot
    filename = "Walker2d_" + str(n_games) + ".png"
    figure_file = "plots/" + filename

    # Initialise the best score
    best_score = env.reward_range[0]

    # Initialise the list to store the scores achieved during training
    score_history = []

    # Load pre-trained models if available
    # agent.load_models()

    # Loop over the specified number of games
    for i in range(n_games):
        # Reset the environment for each new game and get initial observation
        observation, info = env.reset()

        # Initialize variables for tracking terminal and truncated flags, as well as the total score
        terminal = False
        truncated = False
        score = 0
        reward = 0

        # Loop within each game until the episode is complete
        while not (terminal or truncated):
            # Choose an action based on the current observation
            action = agent.choose_action(observation)

            # Take a step in the environment with the chosen action
            observation_, reward, terminal, truncated, info = env.step(action)

            # Store the experience tuple (state, action, reward, next_state, terminal) in the agent's memory
            agent.remember(
                observation, action, reward, observation_, terminal or truncated
            )

            # Update the agent's networks based on the experiences in the memory
            agent.learn()

            # Update the total score for the current episode
            score += reward

            # Update the current observation for the next step
            observation = observation_

        # Append the total score of the current episode to the score history list
        score_history.append(score)

        # Calculate the average score over the last 100 episodes
        avg_score = np.mean(score_history[-100:])

        # If the average score of the current episode is higher than the best score seen so far,
        # update the best score and save the agent's model
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        # Print the progress of training after each episode
        print(
            "episode ",
            i,
            "score %.2f" % score,
            "trailing 100 games avg %.3f" % avg_score,
        )

    # Generate x-axis values for the learning curve plot
    x = [i + 1 for i in range(n_games)]

    # Plot the learning curve
    plot_learning_curve(x, score_history, figure_file)
