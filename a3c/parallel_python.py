import os
import torch.multiprocessing as mp

os.environ["SET_NUM_THREADS"] = "1"


def worker(name):
    print("hello", name)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    process = mp.Process(target=worker, args=("Timi",))
    process.start()
    process.join()
