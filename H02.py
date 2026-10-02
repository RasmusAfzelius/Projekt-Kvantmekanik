import numpy as np
import matplotlib.pyplot as plt


N = 3
eps = np.array([5, 0,0,0,0,0]) 
U = np.array([0.0, 15,15,15,15,15]) 
H02 = np.zeros((N**2,N**2)) #N x N zero-matrix
V = -1

for i in range(N**2):
    n_up, n_dn = divmod(i, N)        # i = n_up*N + n_dn
    for j in range(N**2):
        m_up, m_dn = divmod(j, N)

        if i == j:                                         # diagonal
            H02[i, j] = eps[n_up] + eps[n_dn]
            if(n_up == n_dn):
                plt.text(i, j, f"2e{n_up+1} + U{n_up+1}", ha="center", va="center", color="black")
            else:
                plt.text(i, j, f"e{n_up+1} + e{n_dn+1}", ha="center", va="center", color="black")

            if n_up == n_dn:
                H02[i, j] += U[n_up]                       # both on same site
        elif n_up == m_up and abs(n_dn - m_dn) == 1:       # down electron hops
            H02[i, j] = V
            plt.text(i, j, "V", ha="center", va="center", color="black")

        elif n_dn == m_dn and abs(n_up - m_up) == 1:       # up electron hops
            H02[i, j] = V  
            plt.text(i, j, "V", ha="center", va="center", color="black")

plt.imshow(H02, label="V", cmap="viridis_r")



plt.show()

