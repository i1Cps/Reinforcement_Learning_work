import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)


def plot_learning_curve_from_npy(
    steps_file, scores_file, filename, title="reinforcement learning algorithm returns"
):
    steps = np.load(steps_file)
    scores = np.load(scores_file)
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(steps, running_avg)
    plt.title("MAPPO average returns")
    plt.savefig(filename)
