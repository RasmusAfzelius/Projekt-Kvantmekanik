import numpy as np
from matplotlib import pyplot as plt

V = 1

L = 9

ep_1, ep_2, ep_3 = 10, 20, 30

new_matrix_above = np.eye(L, k=1)
for x in range(new_matrix_above.shape[0] - 1):
    if x % 3 == 2:
        new_matrix_above[x, x + 1] = 0

new_matrix_below = np.eye(L, k=-1)
for x in range(new_matrix_below.shape[0] - 1):
    if x % 3 == 2:
        new_matrix_below[x, x - 1] = 0


H0 = V * (new_matrix_above + new_matrix_below)

epsilon_matrix = np.zeros((9, 9))

for row, first_epsilon in enumerate([ep_1, ep_2, ep_3]):

    for col, second_epsilon in enumerate([ep_1, ep_2, ep_3]):
        epsilon_matrix[row + col, row + col] = first_epsilon + second_epsilon

plt.imshow(epsilon_matrix)

plt.show()
