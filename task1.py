import numpy as np
import matplotlib.pyplot as plt

L, V, eps1 = 6, -1.0, 2.0

H0 = V * (np.eye(L, k=1) + np.eye(L, k=-1)) 
H1 = H0.copy()
H1[0,0] = eps1


E0, psi_n_0 = np.linalg.eigh(H0)
E1, psi_n_lam = np.linalg.eigh(H1)

psi_nt = np.zeros([L], dtype=complex)
t = np.linspace(0,20,100)


def tempfunc(lam):
    sum = 0
    for n in range(L):
        sum += psi_n_0[0,n] * psi_n_lam[lam,n]
    return sum

def solution(t,n):
    psi_net = np.zeros(len(t), dtype=complex)
    for lam in range(L):
            psi_net += np.exp(-1j * E1[lam] * t) * psi_n_lam[lam, n] * tempfunc(lam)

    return psi_net




for i in range(L):
    psi_nt = solution(t,i)
    plt.plot(t,np.abs(psi_nt)**2, label=f"State {i+1}")

plt.legend()
plt.show()





