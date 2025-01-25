import numpy as np
import matplotlib.pyplot as plt

from gates import pauli_x_gate, pauli_z_gate

def entanglement_entropy(state: np.ndarray) -> float:
    if state.shape != (4,):
        raise ValueError("Entanglement entropy only implemented for 2-qubit system.")
    
    rho = state.reshape(-1, 1) * state.reshape(1, -1)

    # Compute partial trace - we trace out the second qubit.
    rho_A = np.zeros((2, 2))
    rho_A[0, 0] = rho[0, 0] + rho[1, 1]
    rho_A[1, 1] = rho[2, 2] + rho[3, 3]
    rho_A[0, 1] = rho[0, 2] + rho[1, 3]
    rho_A[1, 0] = rho[2, 0] + rho[3, 1]
    
    eigvals = np.linalg.eigvalsh(rho_A)

    def safe_log_base_2(x):
        return np.log2(x) if x > 0 else 0

    log_eigvals = np.array([safe_log_base_2(x) for x in eigvals]) # We define x log x := 0.
    return - np.sum(eigvals * log_eigvals)


    
    


X = pauli_x_gate()
Z = pauli_z_gate()

Hx = 2
Hz = 3
H0_diagonal = [0, 2.5, 6.5, 7.0]


H0 = np.diag(H0_diagonal)
HI = Hx * np.kron(X, X) + Hz * np.kron(Z, Z)
# H = H0 + HI


n_lambda_values = 1001

lambdas = np.linspace(0, 1, n_lambda_values)

eigvals = np.zeros((4, n_lambda_values))
ground_states = np.zeros((4, n_lambda_values))

for n, lmbda in enumerate(lambdas):
    H_lambda = H0 + lmbda * HI
    vals, vecs = np.linalg.eigh(H_lambda)
    eigvals[:, n] = vals

    ground_states[:, n] = vecs[:, 0]


for i in range(4):
    plt.plot(lambdas, eigvals[i, :], c="g")


entropies = []
for n in range(n_lambda_values):
    S = entanglement_entropy(ground_states[:, n])
    entropies.append(S)


plt.title("Eigenvalues")
plt.ylabel("Eigenenergy")
plt.xlabel(r"$\lambda$")
plt.show()


plt.title("Entanglement Entropy")
plt.xlabel(r"$\lambda$")
plt.ylabel("Entropy")
plt.plot(lambdas, entropies)
plt.show()



