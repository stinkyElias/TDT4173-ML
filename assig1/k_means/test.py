import numpy as np

x = np.array([1,2,3])
y = 2

print(np.linalg.norm(x-y, ord=2, axis=-1))