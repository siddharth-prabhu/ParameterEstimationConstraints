import numpy as np

def process_multiple_trajectories(alist : list):

    return np.vstack(alist)


a = [np.random.uniform(1, 10, size = (2, 2)) for _ in range(3)]
b = process_multiple_trajectories(a)
print(np.shape(a))
print(np.shape(b))