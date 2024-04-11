import numpy as np


# Rebalances a heap branch given its array and a leaf
def max_heapify(array: list[float], i: int, N=None) -> list[float]:
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
        max_heapify(array, largest, N)

    return array


# Runs max heapify function to cover all branches within tree
def build_max_heap(array):
    N = len(array)
    for i in range(N // 2, -1, -1):  # range: [N/2, 0]
        array = max_heapify(array, i)
    return array


if __name__ == "__main__":
    np.random.seed(42)
    a = np.random.choice(np.arange(100), 21, replace=False)
    print("unsorted array: {}".format(a))
    a = build_max_heap(a)
    reference = np.array(
        [
            90.0,
            80.0,
            83.0,
            77.0,
            55.0,
            73.0,
            70.0,
            76.0,
            53.0,
            44.0,
            18.0,
            30.0,
            39.0,
            33.0,
            22.0,
            4.0,
            45.0,
            10.0,
            12.0,
            31.0,
            0,
        ]
    )
    print("max heap array: {}".format(a))
    assert (a == reference).all()
