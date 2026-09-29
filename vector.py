import numpy as np
import matplotlib.pyplot as plt

#Length
L = 1
#Initialize vector as 0 with N elements
N = 30
vector = np.zeros(N)

x_axis = np.linspace(0,L,N) #Between 0 and L, with N points

plt.scatter(x_axis, vector) #x = x_axis, y = vector
plt.title("Zero vector")
plt.show()


vector[0] = 1 #Set a value on the first element of the vector

#Loops through the vector and sets every new element equal to the element before it
for n in range(1, N): #Index between 1 and N-1 (up to but not including N)
    vector[n] = vector[n-1]

plt.scatter(x_axis, vector)
plt.title("Every element = element before it")
plt.show()

### sin(x) example
vector[0] = 0 #sin(0) = 0
for n in range(1, N): #Index between 1 and N-1 (up to but not including N)
    vector[n] = np.sin(2*np.pi*L*n/N)

plt.scatter(x_axis, vector)
plt.title("sin(x)")
plt.show()


#Wavefunction example
psi = np.zeros(N)
psi[0] = psi[1] = 1
dx = L/N

k = 100
V = k/2 * x_axis**2 #Harmonic oscillator potential
E = 5 #Energy guess
for n in range(1,N-1):
        psi[n+1] = 2*psi[n]-psi[n-1]-2*dx**2*(E-V[n])*psi[n]

#Mirror around x = 0 to show -L to L. [:0:-1] reverses and skips index 0 so x = 0 isn't duplicated
x_full = np.concatenate((-x_axis[:0:-1], x_axis))
psi_full = np.concatenate((psi[:0:-1], psi)) #Even state: psi(-x) = psi(x)
V_full = np.concatenate((V[:0:-1], V)) #V(-x) = V(x)

fig, ax1 = plt.subplots()
ax1.scatter(x_full, psi_full**2, color="tab:blue", label="Probability")
ax1.set_xlabel("x")
ax1.set_ylabel("|psi|^2", color="tab:blue")

ax2 = ax1.twinx() #Second y-axis on the right, sharing the same x-axis
ax2.plot(x_full, V_full, color="tab:orange", label="Potential")
ax2.set_ylabel("V(x)", color="tab:orange")

fig.legend(loc="upper center")
plt.title("PSI wavefunction")
plt.show()
