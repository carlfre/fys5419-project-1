import numpy as np
import matplotlib.pyplot as plt

from gates import multi_kron, pauli_x_gate, pauli_y_gate, pauli_z_gate, identity_gate

X = pauli_x_gate()
Y = pauli_y_gate()
Z = pauli_z_gate()
I = identity_gate()

def single_particle_term(eps: float, N: int) -> np.ndarray:
    ssum = 0
    list_of_identities = [I for _ in range(N)]
    for i in range(N):
        tensor_prod_list = list_of_identities.copy()
        tensor_prod_list[i] = Z
        ssum += multi_kron(*tensor_prod_list)
    return eps / 2 * ssum

def V_term(V: float, N: int) -> np.ndarray:
    ssum = 0
    list_of_identities = [I for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            tensor_prod_list = list_of_identities.copy()
            tensor_prod_list[i] = X
            tensor_prod_list[j] = X
            ssum += multi_kron(*tensor_prod_list)
    ssum2 = 0
    for i in range(N):
        for j in range(i+1, N):
            tensor_prod_list = list_of_identities.copy()
            tensor_prod_list[i] = Y
            tensor_prod_list[j] = Y
            ssum2 += multi_kron(*tensor_prod_list)
    return - V / 2 * (ssum - ssum2)

def W_term(W: float, N: int) -> np.ndarray:
    return 0


def lipkin_hamiltonian(eps: float, V: float, N: int) -> np.ndarray:
    return single_particle_term(eps, N) + V_term(V, N) + W_term(0, N)






def H_J_eq_1(eps: float, V: float) -> np.ndarray:
    term1 = eps / 2 * ( np.kron(Z, I) + np.kron(I, Z))
    
    term2 = - V/2 * ( np.kron(X, X) - np.kron(Y, Y))
    return term1 + term2

def H_J_eq_2(eps: float, V: float) -> np.ndarray:
    Z1 = multi_kron(Z, I, I, I)
    Z2 = multi_kron(I, Z, I, I)
    Z3 = multi_kron(I, I, Z, I)
    Z4 = multi_kron(I, I, I, Z)

    X12 = multi_kron(X, X, I, I)
    X13 = multi_kron(X, I, X, I)
    X14 = multi_kron(X, I, I, X)
    X23 = multi_kron(I, X, X, I)
    X24 = multi_kron(I, X, I, X)
    X34 = multi_kron(I, I, X, X)

    Y12 = multi_kron(Y, Y, I, I)
    Y13 = multi_kron(Y, I, Y, I)
    Y14 = multi_kron(Y, I, I, Y)
    Y23 = multi_kron(I, Y, Y, I)
    Y24 = multi_kron(I, Y, I, Y)
    Y34 = multi_kron(I, I, Y, Y)

    term1 = eps / 2 * (Z1 + Z2 + Z3 + Z4)
    term2 = -V / 2 * (X12 + X13 + X14 + X23 + X24 + X34)
    term3 = -V / 2 * (Y12 + Y13 + Y14 + Y23 + Y24 + Y34)
    return term1 + term2 - term3



def interaction_term(V: float) -> np.ndarray:
    term2 = - V/2 * (np.kron(X, X) + np.kron(Y, Y))
    return term2


V_vals = np.linspace(0, 1, 101)

all_eigvals = np.zeros((len(V_vals), 4))
for i, Vi in enumerate(V_vals):
    # Hi = H_J_eq_1(1, Vi)
    Hi = lipkin_hamiltonian(1, Vi, 2)
    eigvals = np.linalg.eigvalsh(Hi)
    all_eigvals[i] = eigvals
# for Vi in V_vals:
#     H_I = interaction_term(Vi)
#     eigvals = np.linalg.eigvalsh(H_I)
#     # print(f"V = {Vi}, eigvals = {eigvals}")
#     all_eigvals.append(eigvals)




plt.plot(V_vals, all_eigvals[:, 0], label="E_0")
plt.plot(V_vals, all_eigvals[:, 1], label="E_1")
plt.plot(V_vals, all_eigvals[:, 2], label="E_2")
plt.plot(V_vals, all_eigvals[:, 3], label="E_3")
plt.legend()
plt.show()


V_vals = np.linspace(0, 1, 101)

all_eigvals = np.zeros((len(V_vals), 16))
for i, Vi in enumerate(V_vals):
    # Hi = H_J_eq_2(1, Vi)
    Hi = lipkin_hamiltonian(1, Vi, 4)
    eigvals = np.linalg.eigvalsh(Hi)
    all_eigvals[i] = eigvals
# for Vi in V_vals:
#     H_I = interaction_term(Vi)
#     eigvals = np.linalg.eigvalsh(H_I)
#     # print(f"V = {Vi}, eigvals = {eigvals}")
#     all_eigvals.append(eigvals)



# for i in range(16):
plt.plot(V_vals, all_eigvals[:, 0], label=f"E_{i}")
plt.legend()
plt.show()





