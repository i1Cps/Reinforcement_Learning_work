import os
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np

os.environ["SET_NUM_THREADS"] = "4"


class ParallelEnv:
    def __init__(self, num_threads, env_id):
        thread_names = ["thread_" + str(i) for i in range(num_threads)]
        self.processes = [
            mp.Process(target=worker, args=(name, env_id)) for name in thread_names
        ]

        [p.start() for p in self.processes]
        [p.join() for p in self.processes]


def worker(name, env_id):
    env = gym.make(env_id)
    episode = 0
    max_episodes = 10
    score_history = []
    best_score = -np.inf
    while episode < max_episodes:
        obs, info = env.reset()
        trunc = False
        terminated = False
        score = 0
        while not (trunc or terminated):
            action = env.action_space.sample()
            obs_, reward, terminated, trunc, info = env.step(action)
            score += reward
            obs = obs_

        score_history.append(score)
        print("Episode {}, thread {}, score {:.2f}".format(episode, name, score))
        episode += 1

        if score > best_score:
            best_score = score


if __name__ == "__main__":
    mp.get_start_method("spawn")
    env = ParallelEnv(num_threads=4, env_id="CartPole-v1")
