# Import necessary libraries
import numpy as np

# Import custom classes and functions
from utils import plot_learning_curve, make_env
from agent import DQN

if __name__ == "__main__":
    # Initialize the environment
    env = make_env("PongNoFrameskip-v4")

    # Initialize the DQN agent with specific parameters
    dqn = DQN(
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        mem_size=50000,  # Size of the replay memory
        batch_size=64,  # Size of mini-batches sampled from memory
        gamma=0.99,  # Discount factor
        learning_rate=0.00001,  # Learning rate for the neural network
        epsilon=1,  # Initial value of epsilon for epsilon-greedy exploration
        eps_dec=1e-5,  # Epsilon decay rate
        eps_min=0.01,  # Minimum value of epsilon
        replace=1000,  # Frequency of updating target network
        env_name="Pong-v4",  # Name of the environment
    )
    # Initialize variables
    best_score = -np.inf  # Initialize the best score seen so far
    load_checkpoint = False  # Flag indicating whether to load a pre-trained model
    n_games = 500  # Number of games to play

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

            # Store the experience tuple in the agent's memory
            dqn.store_transition(
                observation, action, reward, observation_, terminal or truncated
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
