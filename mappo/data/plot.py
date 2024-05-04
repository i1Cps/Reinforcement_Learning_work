import numpy as np
from utils import plot_learning_curve

mappo_scores = np.load("data/raw_data/mappo_scores.npy")
mappo_steps = np.load("data/raw_data/mappo_steps.npy")


plot_learning_curve(
    x=mappo_steps,
    scores=(mappo_scores),
    filename="plots/mappo_plot",
)
