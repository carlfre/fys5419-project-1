#%%
import multiprocessing
from typing import Callable

import numpy as np
import numba
from scipy.optimize import minimize

from ansatzes import hardware_efficient_2_qubit_ansatz, hardware_efficient_4_qubit_ansatz
from gates import identity_gate, pauli_x_gate, pauli_y_gate, pauli_z_gate, hadamard_gate, cnot_gate, multi_kron, SWAP_gate, CX_10_gate, phase_gate
from utils import write_to_csv

#%%


def measure_qubit_one(state: np.ndarray) -> int:
    dim = state.shape[0]
    if dim % 2 == 1:
        raise ValueError("State vector must have even dimension")
    prob_0 = sum(np.abs(state[:dim//2])**2)
    if np.random.rand() < prob_0:
        return 0
    else:
        return 1
    



I, X, Y, Z, H, SWAP, CX_10, S = identity_gate(), pauli_x_gate(), pauli_y_gate(), pauli_z_gate(), hadamard_gate(), SWAP_gate(), CX_10_gate(), phase_gate()


def Z_measurement_to_energy(measurement: int) -> float:
    if measurement == 0:
        return 1
    else:
        return -1


def easy_2_qubit(eps_00: float, eps_01: float, eps_10: float, eps_11: float, Hx: float, Hz: float, lmbda: float, n_shots=500) -> Callable:

    def expected_value(theta: list[float]) -> float:
        psi = hardware_efficient_2_qubit_ansatz(*theta)

        c_II = (eps_00 + eps_01 + eps_10 + eps_11) / 4
        c_IZ = (eps_00 - eps_01 + eps_10 - eps_11) / 4
        c_ZI = (eps_00 + eps_01 - eps_10 - eps_11) / 4
        c_ZZ = (eps_00 - eps_01 - eps_10 + eps_11) / 4

        energies = []
        
        for _ in range(n_shots):
            noninteracting_energy = 0
            interacting_energy = 0

            # non-interacting circuits
            # I ⨂ I
            noninteracting_energy += c_II

            # Z ⨂ I
            measurement = measure_qubit_one(psi)
            noninteracting_energy += c_ZI * Z_measurement_to_energy(measurement)

            # I ⨂ Z
            measurement_basis = SWAP @ psi
            measurement = measure_qubit_one(measurement_basis)
            noninteracting_energy += c_IZ * Z_measurement_to_energy(measurement)

            # Z ⨂ Z
            measurement_basis = CX_10 @ psi
            measurement = measure_qubit_one(measurement_basis)
            noninteracting_energy += c_ZZ * Z_measurement_to_energy(measurement)

            # interacting circuits

            # Hx term, X ⨂ X
            measurement_basis = CX_10 @ np.kron(H, H) @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += Hx * Z_measurement_to_energy(measurement)

            # Hz term, Z ⨂ Z
            measurement_basis = CX_10 @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += Hz * Z_measurement_to_energy(measurement)

            total_energy = noninteracting_energy + lmbda * interacting_energy
            energies.append(total_energy)
        return np.mean(energies)
    


    theta0 = np.random.rand(4) * 2 * np.pi
    res = minimize(expected_value, theta0, method='Powell')
    return res.fun
    

def lipkin_model_small(eps, V, n_shots=500):

    def expected_value(theta: list[float]) -> float:
        psi = hardware_efficient_2_qubit_ansatz(*theta)
        energies = []


        for _ in range(n_shots):
            non_interacting_energy = 0
            interacting_energy = 0

            # non-interacting terms

            # Z ⨂ I
            measurement = measure_qubit_one(psi)
            non_interacting_energy += eps / 2 * Z_measurement_to_energy(measurement)

            # I ⨂ Z
            measurement_basis = SWAP @ psi
            measurement = measure_qubit_one(measurement_basis)
            non_interacting_energy += eps / 2 * Z_measurement_to_energy(measurement)


            # Interacting terms

            # X ⨂ X
            measurement_basis = CX_10 @ np.kron(H, H) @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += - V / 2 * Z_measurement_to_energy(measurement)

            # Y ⨂ Y
            measurement_basis = CX_10 @ np.kron(H @ S.T.conj(), H @ S.T.conj()) @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += V / 2 * Z_measurement_to_energy(measurement)

            total_energy = non_interacting_energy + interacting_energy
            energies.append(total_energy)
        return np.mean(energies)
    
  
    theta0 = np.random.rand(4) * 2 * np.pi
    res = minimize(expected_value, theta0, method='powell')
    return res.fun


def lipkin_model_big(eps, V, n_shots=500):
    
    def expected_value(theta: list[float]) -> float:
        psi = hardware_efficient_4_qubit_ansatz(*theta)
        energies = []

        for _ in range(n_shots):
            non_interacting_energy = 0
            interacting_energy = 0

            # Non-interacting terms

            # Z ⨂ I ⨂ I ⨂ I
            measurement = measure_qubit_one(psi)
            non_interacting_energy += eps / 2 * Z_measurement_to_energy(measurement)

            # I ⨂ Z ⨂ I ⨂ I
            measurement_basis = multi_kron(SWAP, I, I) @ psi
            measurement = measure_qubit_one(measurement_basis)
            non_interacting_energy += eps / 2 * Z_measurement_to_energy(measurement)

            # I ⨂ I ⨂ Z ⨂ I
            measurement_basis = multi_kron(SWAP, I, I) @ multi_kron(I, SWAP, I) @ psi
            measurement = measure_qubit_one(measurement_basis)
            non_interacting_energy += eps / 2 * Z_measurement_to_energy(measurement)

            # I ⨂ I ⨂ I ⨂ Z
            measurement_basis = multi_kron(SWAP, I, I) @ multi_kron(I, SWAP, I) @ multi_kron(I, I, SWAP) @ psi
            measurement = measure_qubit_one(measurement_basis)
            non_interacting_energy += eps / 2 * Z_measurement_to_energy(measurement)


            # Interacting terms

            # X ⨂ X ⨂ I ⨂ I
            circuit = multi_kron(CX_10, I, I) @ multi_kron(H, H, I, I)
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += - V / 2 * Z_measurement_to_energy(measurement)


            # X ⨂ I ⨂ X ⨂ I
            circuit = multi_kron(CX_10, I, I) @ multi_kron(H, I, I, I) @ multi_kron(I, H, I, I) @ multi_kron(I, SWAP, I)
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += - V / 2 * Z_measurement_to_energy(measurement)

            # X ⨂ I ⨂ I ⨂ X
            circuit = (
                multi_kron(CX_10, I, I) @ multi_kron(H, I, I, I)
                @ multi_kron(I, SWAP, I)
                @ multi_kron(I, I, H, I) @ multi_kron(I, I, SWAP)
            )
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += - V / 2 * Z_measurement_to_energy(measurement)

            # I ⨂ X ⨂ X ⨂ I
            circuit = (
                multi_kron(SWAP, I, I)
                @ multi_kron(I, CX_10, I) @ multi_kron(I, H, H, I)
            )
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += - V / 2 * Z_measurement_to_energy(measurement)

            # I ⨂ X ⨂ I ⨂ X
            circuit = (
                multi_kron(SWAP, I, I)
                @ multi_kron(I, CX_10, I) @ multi_kron(I, H, I, I)
                @ multi_kron(I, I, H, I) @ multi_kron(I, I, SWAP)
            )
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += - V / 2 * Z_measurement_to_energy(measurement)

            # I ⨂ I ⨂ X ⨂ X
            circuit = (
                multi_kron(SWAP, I, I)
                @ multi_kron(I, SWAP, I)
                @ multi_kron(I, I, CX_10) @ multi_kron(I, I, H, H)
            )
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += - V / 2 * Z_measurement_to_energy(measurement)


            # Y ⨂ Y ⨂ I ⨂ I
            circuit = multi_kron(CX_10, I, I) @ multi_kron(H @ S.T.conj(), H @ S.T.conj(), I, I)
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += V / 2 * Z_measurement_to_energy(measurement)

            # Y ⨂ I ⨂ Y ⨂ I
            circuit = (
                multi_kron(CX_10, I, I) @ multi_kron(H @ S.T.conj(), I, I, I)
                @ multi_kron(I, np.kron(H @ S.T.conj(), I) @ SWAP, I)
            )
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += V / 2 * Z_measurement_to_energy(measurement)

            # Y ⨂ I ⨂ I ⨂ Y
            circuit = (
                multi_kron(CX_10, I, I) @ multi_kron(H @ S.T.conj(), I, I, I) #
                @ multi_kron(I, SWAP, I)
                @ multi_kron(I, I, np.kron(H @ S.T.conj(), I) @ SWAP)
            )
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += V / 2 * Z_measurement_to_energy(measurement)

            # I ⨂ Y ⨂ Y ⨂ I
            circuit = (
                multi_kron(SWAP, I, I)
                @ multi_kron(I, CX_10 @ np.kron(H @ S.T.conj(), H @ S.T.conj()), I)
            )
            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += V / 2 * Z_measurement_to_energy(measurement)


            # I ⨂ Y ⨂ I ⨂ Y
            circuit = (
                multi_kron(SWAP, I, I)
                @ multi_kron(I, CX_10 @ np.kron(H @ S.T.conj(), I), I)
                @ multi_kron(I, I, np.kron(H @ S.T.conj(), I) @ SWAP)
            )

            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += V / 2 * Z_measurement_to_energy(measurement)

            # I ⨂ I ⨂ Y ⨂ Y
            circuit = (
                multi_kron(SWAP, I, I)
                @ multi_kron(I, SWAP, I)
                @ multi_kron(I, I, CX_10 @ np.kron(H @ S.T.conj(), H @ S.T.conj()))
            )

            measurement_basis = circuit @ psi
            measurement = measure_qubit_one(measurement_basis)
            interacting_energy += V / 2 * Z_measurement_to_energy(measurement)

            total_energy = non_interacting_energy + interacting_energy
            energies.append(total_energy)
        return np.mean(energies)

    theta0 = np.random.rand(8) * 2 * np.pi
    res = minimize(expected_value, theta0, method='powell')
    return res.fun


def run_easy_2_qubit(lmbda: float):
    n_shots = 10_000

    eps_00 = 0
    eps_01 = 2.5
    eps_10 = 6.5
    eps_11 = 7
    Hx = 2
    Hz = 3
    return easy_2_qubit(eps_00, eps_01, eps_10, eps_11, Hx, Hz, lmbda, n_shots=n_shots)

def run_lipkin_model_small(V: float):
    n_shots = 10_000
    eps = 1
    return lipkin_model_small(eps, V, n_shots=n_shots)


def run_lipkin_model_big(V: float):
    n_shots = 10_000
    eps = 1
    return lipkin_model_big(eps, V, n_shots=n_shots)






#%%




# lmbda_grid = np.linspace(0, 1, 16)
# with multiprocessing.Pool(16) as pool:
#     vqe_estimates = pool.map(run_easy_2_qubit, lmbda_grid)

lmbda_grid
vqe_estimates = []
for lmbda in np.linspace(0, 1, 16):
    vqe_estimates.append(run_easy_2_qubit(lmbda))


write_to_csv([lmbda_grid, vqe_estimates], ['lambda', 'energy'], 'output/simple_2_qubit.csv')



#%%



V_grid = np.linspace(0, 1, 16)
with multiprocessing.Pool(16) as pool:
    vqe_estimates = pool.map(run_lipkin_model_small, V_grid)

write_to_csv([V_grid, vqe_estimates], ['V', 'energy'], 'output/lipkin_small.csv')


#%%


V_grid = np.linspace(0, 1, 16)
with multiprocessing.Pool(16) as pool:
    vqe_estimates = pool.map(run_lipkin_model_big, V_grid)

write_to_csv([V_grid, vqe_estimates], ['V', 'energy'], 'output/lipkin_big.csv')


#%%
import matplotlib.pyplot as plt
plt.plot(V_grid, vqe_estimates)

#%%


# lmbda_grid = np.linspace(0, 1, 10)
# vqe_estimates = np.zeros_like(lmbda_grid)

# for i, lmbda in enumerate(lmbda_grid):
#     print(f"Running for lambda = {lmbda}")
#     eps_00 = 0
#     eps_01 = 2.5
#     eps_10 = 6.5
#     eps_11 = 7
#     Hx = 2
#     Hz = 3
#     vqe_estimates[i] = easy_2_qubit(0, 2.5, 6.5, 7, 2, 3, lmbda=lmbda, n_shots=1_000)


# import matplotlib.pyplot as plt
# plt.plot(lmbda_grid, vqe_estimates)
# plt.xlabel(r"$\lambda$")
# plt.ylabel("Energy")
# plt.title("Energy vs. $\lambda$")
# plt.show()



lmbda_grid = np.linspace(0, 1, 11)
vqe_estimates = np.zeros_like(lmbda_grid)

for i, lmbda in enumerate(lmbda_grid):
    print(f"Running for lambda = {lmbda}")
    eps_00 = 0
    eps_01 = 2.5
    eps_10 = 6.5
    eps_11 = 7
    Hx = 2
    Hz = 3
    vqe_estimates[i] = lipkin_model_small(1, lmbda, n_shots=1_000)


import matplotlib.pyplot as plt
plt.plot(lmbda_grid, vqe_estimates)
plt.xlabel(r"$\lambda$")
plt.ylabel("Energy")
plt.title("lipkin small, Energy vs. $V/\epsilon$")
plt.show()




# res = easy_2_qubit(0, 2.5, 6.5, 7, 2, 3, lmbda=1, n_shots=1000)
# print(res)

    

    # min_energy = 99999999999
    # N_divisions = 5
    # theta_range = np.linspace(0, 2*np.pi, N_divisions)
    # for i in range(N_divisions):
    #     for j in range(N_divisions):
    #         for k in range(N_divisions):
    #             for l in range(N_divisions):
    #                 energy = 0
    #                 for _ in range(100):
    #                     theta = [theta_range[i], theta_range[j], theta_range[k], theta_range[l]]
    #                     energy += run_circuits(theta)
    #                 energy /= 100
    #                 print(energy)
    #                 # return
    #                 min_energy = min(energy, min_energy)

    # print(min_energy)
                    



    # theta_init = [0] * 4
    # theta, grad, H_theta_expected, full_data = vqe(run_circuit_once, theta_init, iters=500, lr=0.001, n_shots=500)

    # print(H_theta_expected)
    # return theta, grad, H_theta_expected, full_data


# theta, grad, H_theta_expected, full_data = easy_2_qubit(0, 2.5, 6.5, 7, 2, 3)


# %% 

# print("Minimized function value:", res.fun)


# # %%
# thetas, H_theta_expecteds, grads = full_data
# print(thetas)

# # %%
# # print(grads)
# print(H_theta_expecteds)
