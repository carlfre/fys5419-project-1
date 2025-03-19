# Import necessary libraries
from qiskit_aer import Aer
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, SLSQP, COBYLA, NELDER_MEAD, L_BFGS_B, ADAM
from qiskit.primitives import Estimator
from qiskit.quantum_info import Operator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import matplotlib.pyplot as plt

# Define Pauli matrices and identity matrix
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.array([[1, 0], [0, 1]])
W = 0

# Updated Lipkin Hamiltonian using angular momentum operators
def lipkin_hamiltonian(J, V):
    epsilon = 0.5
    if J == 1:
        HJ1_rewritten = -epsilon * np.kron(sigma_z, identity) - V * np.kron(sigma_x, identity)
        return HJ1_rewritten 
    elif J == 2:
        HJ2_rewritten = (-2 * epsilon * np.kron(sigma_z, np.kron(identity, identity)) +
                  np.sqrt(6) * V * np.kron(sigma_x, np.kron(identity, identity)) +
                  3 * W * np.kron(identity, np.kron(sigma_z, identity)))
        return HJ2_rewritten
    
def SparseLipkinHamiltonian(J, V): 
    epsilon = 0.5
    #have to convert the Hamiltonian to a SparsePauliOp to use VQE.
    if J == 1:
        H_op = SparsePauliOp.from_operator(lipkin_hamiltonian(J, V))
    elif J == 2:
        H_op = SparsePauliOp.from_operator(lipkin_hamiltonian(J, V))
    return H_op 


estimator = Estimator()
backend = Aer.get_backend("statevector_simulator")
optimizer = ADAM(maxiter=5000)


def problem_g():
    # get numpy and our code data
    np_data_J_eq_1 = np.genfromtxt("output/np_lipkin_small.csv", delimiter=",", skip_header=1)
    np_data_J_eq_2 = np.genfromtxt("output/np_lipkin_big.csv", delimiter=",", skip_header=1)

    np_lambdas_J_eq_1 = np_data_J_eq_1[:, 0]
    np_lambdas_J_eq_2 = np_data_J_eq_2[:, 0]
    np_energy_J_eq_1 = np_data_J_eq_1[:, 1]
    np_energy_J_eq_2 = np_data_J_eq_2[:, 1]

    our_code_data_J_eq_1 = np.genfromtxt("output/lipkin_J_eq_1_alternate.csv", delimiter=",", skip_header=1)
    our_code_data_J_eq_2 = np.genfromtxt("output/lipkin_J_eq_2_alternate.csv", delimiter=",", skip_header=1)

    our_code_lambdas_J_eq_1 = our_code_data_J_eq_1[:, 0]
    our_code_lambdas_J_eq_2 = our_code_data_J_eq_2[:, 0]
    our_code_energy_J_eq_1 = our_code_data_J_eq_1[:, 1]
    our_code_energy_J_eq_2 = our_code_data_J_eq_2[:, 1]

    # make plot and place points on top
    plt.plot(np_lambdas_J_eq_1, np_energy_J_eq_1, label=r"$J = 1$ with NumPy", color="seagreen", zorder=1)
    plt.plot(np_lambdas_J_eq_1, np_energy_J_eq_2, label=r"$J = 2$ with NumPy", color="royalblue", zorder=2)
    
    plt.scatter(our_code_lambdas_J_eq_1, our_code_energy_J_eq_1, label=r"$J = 1$ with VQE (our code)", marker="x", c="coral", zorder=3)
    plt.scatter(our_code_lambdas_J_eq_2, our_code_energy_J_eq_2, label=r"$J = 2$ with VQE (our code)", marker="x", c="crimson", zorder=4)

    plt.title(r"Eigenenergy vs. $\lambda$ for Lipkin Hamiltonians with NumPy methods and VQE")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Eigenenergy")
    plt.legend()
    plt.savefig("images/problem_g_our_code_vs_numpy.png")
    plt.show()

    # previous version plot with Qiskit results only (uncomment if changes need to be made, takes time to run)
    # interaction_strengths = np.linspace(0, 1, 10)
    # J = [1, 2]
    # qbits = [2,4]
    
    # eigenval_J1 = []
    # eigenval_J2 = []

    # for j, k in zip(J, qbits): 

    #     # Define the ansatz
    #     ansatz = TwoLocal(num_qubits=k, rotation_blocks='ry', entanglement_blocks='cz', reps=1, entanglement='linear')
            
    #     for i in interaction_strengths:
    
    #         # Define the Hamiltonian 
    #         H_op = SparseLipkinHamiltonian(j, i)

    #         # Set up the VQE
    #         vqe = VQE(estimator, ansatz, optimizer)

    #         # Run the VQE
    #         result = vqe.compute_minimum_eigenvalue(H_op)

    #         if j == 1:
    #             eigenval_J1.append(result.eigenvalue)
    #         else:
    #             eigenval_J2.append(result.eigenvalue)


    # plt.plot(interaction_strengths, eigenval_J1, label="J = 1")
    # plt.plot(interaction_strengths, eigenval_J2, label="J = 2")

    # plt.title("Eigenenergy vs. $\lambda$ for Lipkin Hamiltonians with VQE")
    # plt.xlabel(r"$\lambda$")
    # plt.ylabel("Eigenenergy")
    # plt.legend()
    # plt.savefig("images/problem_g.png")
    # plt.show()

problem_g()