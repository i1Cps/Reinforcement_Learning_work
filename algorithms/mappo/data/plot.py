import numpy as np
import matplotlib.pyplot as plt

mappo_scores = np.load("raw_data/mappo_scores.npy")
mappo_steps = np.load("raw_data/mappo_steps.npy")


def plot_learning_curve(x, scores, filename):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("MAPPO average returns")
    plt.savefig(filename)


def main():
    plot_learning_curve(
        x=mappo_steps,
        scores=(mappo_scores),
        filename="plots/mappo_plot",
    )


if __name__ == "__main__":
    main()
