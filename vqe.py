#%%
import numpy as np
from scipy.optimize import minimize

from problem_a import measure_first_qubit, measure_second_qubit
from ansatzes import hardware_efficient_2_qubit
from gates import identity_gate, pauli_x_gate, pauli_y_gate, pauli_z_gate, hadamard_gate, cnot_gate, multi_kron, SWAP_gate, CX_10_gate, phase_gate


#%%

from typing import Callable


I, X, Y, Z, H, SWAP, CX_10, S = identity_gate(), pauli_x_gate(), pauli_y_gate(), pauli_z_gate(), hadamard_gate(), SWAP_gate(), CX_10_gate(), phase_gate()


def Z_measurement_to_energy(measurement: int) -> float:
    if measurement == 0:
        return 1
    else:
        return -1


def easy_2_qubit(eps_00: float, eps_01: float, eps_10: float, eps_11: float, Hx: float, Hz: float, lmbda: float, n_shots=500) -> Callable:

    def run_circuits_once(theta: list[float]) -> float:
        
        noninteracting_energy = 0
        interacting_energy = 0


        psi = hardware_efficient_2_qubit(*theta)

        # non-interacting circuits
        
        c_II = (eps_00 + eps_01 + eps_10 + eps_11) / 4
        c_IZ = (eps_00 - eps_01 + eps_10 - eps_11) / 4
        c_ZI = (eps_00 + eps_01 - eps_10 - eps_11) / 4
        c_ZZ = (eps_00 - eps_01 - eps_10 + eps_11) / 4

        # I ⨂ I
        noninteracting_energy += c_II

        # Z ⨂ I
        measurement = measure_first_qubit(psi)[0]
        noninteracting_energy += c_ZI * Z_measurement_to_energy(measurement)

        # I ⨂ Z
        measurement_basis = SWAP @ psi
        measurement = measure_first_qubit(measurement_basis)[0]
        noninteracting_energy += c_IZ * Z_measurement_to_energy(measurement)

        # Z ⨂ Z
        measurement_basis = CX_10 @ psi
        measurement = measure_first_qubit(measurement_basis)[0]
        noninteracting_energy += c_ZZ * Z_measurement_to_energy(measurement)

        # interacting circuits

        # Hx term, X ⨂ X
        measurement_basis = CX_10 @ np.kron(H, H) @ psi
        measurement = measure_first_qubit(measurement_basis)[0]
        interacting_energy += Hx * Z_measurement_to_energy(measurement)

        # Hz term, Z ⨂ Z
        measurement_basis = CX_10 @ psi
        measurement = measure_first_qubit(measurement_basis)[0]
        interacting_energy += Hz * Z_measurement_to_energy(measurement)

        total_energy = noninteracting_energy + lmbda * interacting_energy
        return total_energy
    

    def function_to_minimize(theta: np.ndarray) -> float:
        total_energy = 0
        for _ in range(n_shots):
            total_energy += run_circuits_once(theta)
        return total_energy / n_shots
    
    res = minimize(function_to_minimize, [0, 0, 0, 0], method='powell')
    return res.fun
    

def lipkin_model_small(eps, V, n_shots=500):

    def run_circuits_once(theta: list[float]) -> float:
        psi = hardware_efficient_2_qubit(*theta)
        # non-interacting terms
        non_interacting_energy = 0
        interacting_energy = 0

        # non-interacting terms

        # Z ⨂ I
        measurement = measure_first_qubit(psi)[0]
        non_interacting_energy += eps / 2 * Z_measurement_to_energy(measurement)


        # Interacting terms

        # X ⨂ X
        measurement_basis = CX_10 @ np.kron(H, H) @ psi
        measurement = measure_first_qubit(measurement_basis)[0]
        interacting_energy += - V / 2 * Z_measurement_to_energy(measurement)

        # Y ⨂ Y
        measurement_basis = CX_10 @ np.kron(H @ S.T.conj(), H @ S.T.conj()) @ psi
        measurement = measure_first_qubit(measurement_basis)[0]
        interacting_energy += V / 2 * Z_measurement_to_energy(measurement)

        total_energy = non_interacting_energy + interacting_energy
        return total_energy
    
    def function_to_minimize(theta: np.ndarray) -> float:
        total_energy = 0
        for _ in range(n_shots):
            total_energy += run_circuits_once(theta)
        return total_energy / n_shots
    
    res = minimize(function_to_minimize, [0, 0, 0, 0], method='powell')
    return res.fun






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
#     vqe_estimates[i] = easy_2_qubit(0, 2.5, 6.5, 7, 2, 3, lmbda=lmbda, n_shots=500)


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
    vqe_estimates[i] = lipkin_model_small(1, lmbda, n_shots=1000)


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

print("Minimized function value:", res.fun)


# # %%
# thetas, H_theta_expecteds, grads = full_data
# print(thetas)

# # %%
# # print(grads)
# print(H_theta_expecteds)
