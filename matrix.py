import numpy as np
import matplotlib.pyplot as plt

N = 6
matrix = np.zeros((N,N)) #N x N zero-matrix

V = 5

for i in range(N):
    for j in range(N):
        if np.abs(i-j) == 1:
            matrix[i,j] = V #Set V on the two off-diagonals
            
print("Matrix:")
print(matrix)
plt.imshow(matrix, label="V")

for i in range(N):
    for j in range(N):
        if matrix[i,j] == V:

            plt.text(i, j, "V", ha="center", va="center", color="black")

plt.show()

eigenvalues, eigenvectors = np.linalg.eigh(matrix)

print("Eigenvalues:")
print(eigenvalues)
print("\n Eigenvectors:")
print(eigenvectors)