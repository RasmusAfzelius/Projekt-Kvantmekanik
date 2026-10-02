import numpy as np

array = np.array([[1, 0, 0], [2, 2, 0], [0, 0, 3]])
print(array)

print(np.linalg.eigh(array).eigenvectors)
