from copy import deepcopy
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class MemoryCell:

    """
    Represents a single memory cell in the prioritized experience replay buffer.

    Attributes:
        - priority (float): Priority of the memory cell.
        - rank (int): Rank of the memory cell in the priority queue.
        - transition (List[np.array]): Transition stored in the memory cell.
    """

    priority: float
    rank: int
    transition: List[np.array] = field(repr=False)

    def update_priority(self, new_priority: float):
        """Update the priority of the memory cell."""
        self.priority = new_priority

    def update_rank(self, new_rank: int):
        """Update the rank of the memory cell."""
        self.rank = new_rank

    def __gt__(self, other):
        """Greater-than comparison based on priority."""
        return self.priority > other.priority

    def __ge__(self, other):
        """Greater-than-or-equal comparison based on priority."""
        return self.priority >= other.priority

    def __lt__(self, other):
        """Less-than comparison based on priority."""
        return self.priority < other.priority

    def __le__(self, other):
        """Less-than-or-equal comparison based on priority."""
        return self.priority < other.priority


class MaxHeap:
    """
    Priority queue implemented as a max heap for prioritized experience replay.

    Attributes:
        - max_size (int): Maximum size of the memory buffer.
        - mem_cntr (int): Current number of stored transitions in the buffer.
        - array (List[MemoryCell]): List representing the max heap.
        - n_batches (int): Number of batches for priority updates.
        - alpha (float): Priority exponent for sampling.
        - beta (float): Importance sampling exponent.
        - beta_start (float): Initial value of beta.
        - alpha_start (float): Initial value of alpha.
        - r_iter (int): Number of iterations for priority ranking updates.
        - indices (List): Precomputed indices for batching priority updates.
        - probs (List): Probabilities for sampling transitions.
    """

    def __init__(
        self,
        max_size: int = 1e6,
        n_batches: int = 32,
        alpha: float = 0.5,
        beta: float = 0,
        r_iter: int = 32,
    ):
        self.array: List[MemoryCell] = []
        self.max_size = max_size
        self.mem_cntr: int = 0
        self.n_batches = n_batches
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.alpha_start = alpha
        self.r_iter = r_iter
        self._precompute_indices()

    # Methods for storing and updating memory cells

    def store_transition(self, sarsd: List[np.array]):
        """Store a transition in the memory buffer."""
        priority = 10
        rank = 1
        transition = MemoryCell(priority, rank, sarsd)
        self._insert(transition)

    def _insert(self, transition: MemoryCell):
        """Insert a transition into the max heap."""
        if self.mem_cntr < self.max_size:
            self.array.append(transition)
        else:
            index = self.mem_cntr % self.max_size
            self.array[index] = transition
        self.mem_cntr += 1

    def _update_ranks(self):
        """Update the ranks of memory cells in the max heap."""
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

    # Methods for managing the max heap
    def print_array(self, a=None):
        array = self.array if a is None else a
        for cell in array:
            print(cell)
        print("\n")

    def _max_heapify(self, array: List[MemoryCell], i: int, N: int = None):
        """Heapify the array with a max-heap property."""
        N = len(array) if N is None else N
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        if left < N and array[left] > array[i]:
            largest = left
        if right < N and array[right] > array[largest]:
            largest = right
        if largest != i:
            array[i], array[largest] = array[largest], array[i]
            self._max_heapify(array, largest, N)
        return array

    def _build_max_heap(self):
        """Build a max heap from the array."""
        array = deepcopy(self.array)
        N = len(array)
        for i in range(N // 2, -1, -1):
            array = self._max_heapify(array, i)
        return array

    def rebalance_heap(self):
        """Rebalance the heap by reconstructing the max heap."""
        self.array = self._build_max_heap()

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities of memory cells based on indices."""
        for idx, index in enumerate(indices):
            self.array[index].update_priority(priorities[idx])

    def ready(self):
        """Check if enough transitions are stored for sampling."""
        return self.mem_cntr >= self.n_batches

    def anneal_beta(self, ep: int, ep_max: int):
        """Anneal beta for importance sampling."""
        self.beta = self.beta_start + ep / ep_max * (1 - self.beta_start)

    def anneal_alpha(self, ep: int, ep_max: int):
        """Anneal alpha for priority sampling."""
        self.alpha = self.alpha_start * (1 - ep / ep_max)

    def _precompute_indices(self):
        """Precompute indices for batching priority updates."""
        print("precomputing indices")
        self.indices = []
        n_batches = self.n_batches if self.r_iter > 1 else self.r_iter
        start = [i for i in range(n_batches, self.max_size + 1, n_batches)]
        for start_idx in start:
            bs = start_idx // n_batches
            indices = np.array(
                [[j * bs + k for k in range(bs)] for j in range(n_batches)],
                dtype=np.int16,
            )
            self.indices.append(indices)

    def compute_probs(self):
        """Compute probabilities for sampling."""
        self.probs = []
        n_batches = self.n_batches if self.r_iter > 1 else self.r_iter
        idx = min(self.mem_cntr, self.max_size) // n_batches - 1
        for indices in self.indices[idx]:
            probs = []
            for index in indices:
                p = 1 / (self.array[index].rank) ** self.alpha
                probs.append(p)
            z = [p / sum(probs) for p in probs]
            self.probs.append(z)

    def _calculate_weights(self, probs: List):
        """Calculate importance sampling weights."""
        weights = np.array(
            [(1 / self.mem_cntr * 1 / prob) ** self.beta for prob in probs]
        )
        weights *= 1 / (max(weights))
        return weights

    def sample(self):
        """Sample transitions from the buffer."""
        n_batches = self.n_batches if self.r_iter > 1 else self.r_iter
        idx = min(self.mem_cntr, self.max_size) // n_batches - 1
        if self.r_iter != 1:
            samples = [
                np.random.choice(self.indices[idx][row], p=self.probs[row])
                for row in range(len(self.indices[idx]))
            ]
            p = [val for row in self.probs for val in row]
            probs = [p[s] for s in samples]
        else:
            samples = np.random.choice(self.indices[idx][0], self.n_batches)
            probs = [1 / len(samples) for _ in range(len(samples))]
        weights = self._calculate_weights(probs)
        mems = np.array([self.array[s] for s in samples])
        sarsd = []
        for item in mems:
            row = []
            for i in range(len(item.transition)):
                row.append(np.array(item.transition[i]))
            sarsd.append(row)
        return sarsd, samples, weights
