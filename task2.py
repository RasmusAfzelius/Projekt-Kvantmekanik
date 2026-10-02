import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 6
eps = np.array([-15, 0.0, 0.0, 0.0, 0.0, 0.0]) 
U = np.array([0.0, 15.0, 15.0, 15.0, 15.0, 15.0]) 
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

eps = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
U = np.array([0.0, 15.0, 15.0, 15.0, 15.0, 15.0]) 
H12 = np.zeros((N**2,N**2)) #N x N zero-matrix
V = -1

for i in range(N**2):
    n_up, n_dn = divmod(i, N)        # i = n_up*N + n_dn
    for j in range(N**2):
        m_up, m_dn = divmod(j, N)

        if i == j:                                         # diagonal
            H12[i, j] = eps[n_up] + eps[n_dn]
            if(n_up == n_dn):
                plt.text(i, j, f"2e{n_up+1} + U{n_up+1}", ha="center", va="center", color="black")
            else:
                plt.text(i, j, f"e{n_up+1} + e{n_dn+1}", ha="center", va="center", color="black")

            if n_up == n_dn:
                H12[i, j] += U[n_up]                       # both on same site
        elif n_up == m_up and abs(n_dn - m_dn) == 1:       # down electron hops
            H12[i, j] = V
            plt.text(i, j, "V", ha="center", va="center", color="black")

        elif n_dn == m_dn and abs(n_up - m_up) == 1:       # up electron hops
            H12[i, j] = V  
            plt.text(i, j, "V", ha="center", va="center", color="black")





t = np.linspace(0, 20, 1000)
L = 6



E0, psi_0_eigenvectors = np.linalg.eigh(
    H02
)  # H_0|psi(0)>=E_0|psi(0)> , each column [:,i] is the ith eigenvector of H0
E1, lambda_basis = np.linalg.eigh(
    H12
)  # H'|lambda>=E_lambda|lambda>, each column [:,i] is the ith eigenvector of H1

psi_0 = psi_0_eigenvectors[
    :, 0
]  # we only care about the first one because that  minimizes energy


def solution(t, n):  # calculates equation 16

    def inner_sum(lam):  # calculates the inner sum of equation 16

        eigenvector = lambda_basis[:, lam]
        sum = 0
        for n in range(L**2):
            sum += psi_0[n] * eigenvector[n]

        return sum

    psi_net = np.zeros(len(t), dtype=complex)
    for lam in range(L**2):

        eigenvector = lambda_basis[:, lam]

        psi_net += np.exp(-1j * E1[lam] * t) * eigenvector[n] * inner_sum(lam)

    return psi_net


def plot3d(dataset):  # plots the wavefunctsions along each other

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")

    for lane, data in enumerate(dataset):
        x = np.arange(len(data))
        y = np.full_like(x, lane)
        z = data

        ax.plot(x, y, z, linewidth=3, label=lane)

    ax.set_xlabel("time")
    ax.set_ylabel("potential well")
    ax.set_zlabel("Probability density")

    ax.view_init(elev=25, azim=-65)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_animated(dataset):  # animates the probability density in time across space

    fig, ax = plt.subplots(figsize=(10, 6))

    n_lanes = len(dataset)
    n_frames = len(dataset[0])

    lanes = np.arange(n_lanes)

    # Initial values
    densities = [data[0] for data in dataset]

    (line,) = ax.plot(lanes, densities, "o-", linewidth=2, markersize=7)

    ax.set_xlabel("Lane")
    ax.set_ylabel("Probability density")

    ax.set_xlim(-0.5, n_lanes - 0.5)
    ax.set_ylim(np.min(dataset), np.max(dataset))

    ax.set_xticks(lanes)

    def update(frame):
        densities = [data[frame] for data in dataset]

        # print(sum(densities)) double check that normalization is kept

        line.set_data(lanes, densities)
        ax.set_title(f"Time = {frame}")

        return (line,)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=5, blit=True)

    plt.tight_layout()
    plt.show()

    return ani


wave_function_dataset = [abs(solution(t, i)) ** 2 for i in range(L**2)]
rho_up = [sum(wave_function_dataset[L*m+k] for k in range(L)) for m in range(L)]
plot3d(rho_up)

#plot_animated(wave_function_dataset).save("wavefunctions.mp4", writer="ffmpeg", fps=30)