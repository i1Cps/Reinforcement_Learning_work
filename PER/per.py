from copy import deepcopy
from dataclasses import dataclass, field
import enum
from typing import List
import numpy as np


# Class for storting priorties and ranks for each memory
@dataclass
class MemoryCell:
    priority: float
    rank: int
    transition: List[np.array] = field(repr=False)

    def update_priority(self, new_priority: float):
        self.priotiy = new_priority

    def update_rank(self, new_rank: int):
        self.rank = new_rank

    # Add overload comparision function for memory cell datatype

    # Greater than
    def __gt__(self, other):
        return self.priority > other.priority

    # Greate than or equal
    def __ge__(self, other):
        return self.priotiy >= other.priority

    # Less than
    def __lt__(self, other):
        return self.priority < other.priority

    # Less than or equal too
    def __le__(self, other):
        return self.priority <= other.priority


class MaxHeap:
    def __init__(
        self,
        max_size: int = 1e6,
        n_batches: int = 32,
        alpha: float = 0.5,
        beta: float = 0,
        rebalance_iter: int = 32,
    ):
        self.array: List[MemoryCell] = []
        self.max_size: int = max_size
        self.memory_counter: int = 0
        self.n_batches = n_batches
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.alpha_start = alpha
        self.rebalance_iter = rebalance_iter

    # Takes in typical state, action , reward, new_state, done
    def store_transition(self, sarsd: List[np.array]):
        priority = 10
        rank = 1
        transition = MemoryCell(priority, rank, sarsd)
        self._insert(transition)

    # use '_' for private function ( clean function, professional code)
    def _insert(self, transition: MemoryCell):
        if self.memory_counter < self.max_size:
            self.array.append(transition)
        else:
            index = self.memory_counter % self.max_size
            self.array[index] = transition
        self.memory_counter += 1

    # Below is an implace sort of two arrays at the same time
    # We need to include a indices array, to index the memory cells after the sort
    # Sort on the basis of the 0th element of the pair which is the priority
    def _update_ranks(self):
        array = deepcopy(self.array)
        indices = [i for i in range(len(array))]
        sorted_array = [
            list(x)
            for x in zip(
                *sorted(zip(array, indices), key=lambda pair: pair[0], reverse=True)
            )
        ]
        for index, value in enumerate(sorted_array[1]):
            self.array[value].rank = index + 1

    def print_array(self, a=None):
        array = self.array if a is None else a
        for cell in array:
            print(cell)

    # Rebalances a heap branch given its array and a leaf
    def _max_heapify(
        self, array: List[MemoryCell], i: int, N: int = None
    ) -> List[MemoryCell]:
        N = len(array) if N is None else N
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i

        # Check if left child leaf exists and if its bigger than parent at index i
        if left < N and array[left] > array[i]:
            largest = left
        # Check if right child leaf exists and if its bigger than parent at index i
        if right < N and array[right] > array[largest]:
            largest = right
        # Swap largest and current index i if largest is bigger
        if i != largest:
            array[i], array[largest] = array[largest], array[i]
            self._max_heapify(array, largest, N)

        return array

    # Runs max heapify function to cover all branches within tree
    def _build_max_heap(self):
        array = deepcopy(self.array)
        N = len(array)
        for i in range(N // 2, -1, -1):  # range: [N/2, 0]
            array = self._max_heapify(array, i)
        return array

    def rebalance_heap(self):
        self.array = self._build_max_heap()

    def update_priorities(self, indices: List[int], priorities: List[float]):
        for idx, index in enumerate(indices):
            self.array[index].update_priority(priorities[idx])

    def ready(self):
        return self.memory_counter >= self.n_batches

    # 0 to 1 thoughout training episodes
    def anneal_beta(self, ep: int, ep_max: int):
        self.beta = self.beta_start + ep / ep_max * (1 - self.beta_start)

    # 1 to 0 throughout training episodes
    def anneal_alpha(self, ep: int, ep_max: int):
        self.alpha = self.alpha_start * (1 - ep / ep_max)

    def _precompute_indices(self):
        print("precomputing indices")
        self.indices = []
        n_batches = self.n_batches if self.rebalance_iter > 1 else self.rebalance_iter
        start = [i for i in range(n_batches, self.max_size + 1, n_batches)]
        for start_idx in start:
            bs = start_idx // n_batches
            indices = np.array(
                [[j * bs + k for k in range(bs)] for j in range(n_batches)],
                dtype=np.int16,  # 16 bit is -32000,320000, use 32 bit for bigger memory size
            )
            self.indices.append(indices)

    def compute_probs(self):
        self.probs = []
        n_batches = self.n_batches if self.rebalance_iter > 1 else self.rebalance_iter
        idx = min(self.memory_counter, self.max_size) // n_batches - 1  # Get row index
        for indices in self.indices[idx]:
            probs = []
            for index in indices:
                p = (
                    1 / (self.array[index].rank) ** self.alpha
                )  # Get probs based on rank and alpha
                probs.append(p)
            z = [p / sum(probs) for p in probs]  # Normalize
            self.probs.append(z)

    def _calculate_weights(self, probs: List):
        weights = np.array(
            [(1 / self.memory_counter * 1 / prob) ** self.beta for prob in probs]
        )
        weights *= 1 / (max(weights))  # Normalize ffafaf
        return weights

    def sample(self):
        n_batches = self.n_batches if self.rebalance_iter > 1 else self.rebalance_iter
        idx = min(self.memory_counter, self.max_size) // n_batches - 1  # Get row index

        # Use nummpy random choice to randomley select elements from the inner list for a given row
        if self.rebalance_iter != 1:
            samples = [
                np.random.choice(self.indices[idx][row], p=self.probs[row])
                for row in range(len(self.indices[idx]))
            ]
            p = [val for row in self.probs for val in row]  # LOL
            probs = [p[s] for s in samples]
        else:
            # For uniform sampling just get the 0th element because their wont be stacking in each inner list
            # help
            samples = np.random.choice(self.indices[idx][0], self.n_batches)
            probs = [1 / len(samples) for _ in range(len(samples))]
        weights = self._calculate_weights(probs)
        # Get the actual state, action, rewards etc using sampled index from sampels
        mems = np.array([self.array[s] for s in samples])
        sarsd = []
        for item in mems:
            row = []
            for i in range(len(item.transition)):
                row.append(np.array(item.transition[i]))
            sarsd.append(row)
        return sarsd, samples, weights
