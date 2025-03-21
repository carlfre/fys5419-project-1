# Import necessary libraries
from qiskit_aer import Aer
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import ADAM
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import matplotlib.pyplot as plt

from gates import pauli_x_gate, pauli_y_gate, pauli_z_gate, identity_gate

X = pauli_x_gate()
Y = pauli_y_gate()
Z = pauli_z_gate()
I = identity_gate()

# Updated Lipkin Hamiltonian using angular momentum operators
def lipkin_hamiltonian(J, V):
    epsilon = 1
    W = 0
    if J == 1:
        # HJ1 as a Pauli string, as determined in problem f
        HJ1 = -epsilon/2 * (np.kron(Z, I) + np.kron(Z, Z)) - V/2 * (np.kron(X, I) + np.kron(X, Z))
        return HJ1
    
    elif J == 2:
        # zero padded HJ2 so dimension is a power of 2
        HJ2 = np.array([[-2*epsilon, 0, -np.sqrt(6)*V, 0, 0, 0, 0, 0],
              [0, -epsilon + 3*W, 0, -3*V, 0, 0, 0, 0],
              [-np.sqrt(6)*V, 0, 4*W, 0, -np.sqrt(6)*V, 0, 0, 0],
              [0, -3*V, 0, -epsilon + 3*W, 0, 0, 0, 0],
              [0, 0, -np.sqrt(6)*V, 0, 2*epsilon, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])
        return HJ2
    
def SparseLipkinHamiltonian(J, V): 
    #have to convert the Hamiltonian to a SparsePauliOp to use VQE
    if J == 1:
        H_op = SparsePauliOp.from_operator(lipkin_hamiltonian(J, V))
    elif J == 2:
        H_op = SparsePauliOp.from_operator(lipkin_hamiltonian(J, V))
    return H_op 


estimator = Estimator()
backend = Aer.get_backend("statevector_simulator")
optimizer = ADAM(maxiter=5000)


def problem_g():

    # Harware-efficient ansatz

    # get numpy and our code data
    np_data_J_eq_1 = np.genfromtxt("output/np_lipkin_J_eq_1.csv", delimiter=",", skip_header=1)
    np_data_J_eq_2 = np.genfromtxt("output/np_lipkin_J_eq_2.csv", delimiter=",", skip_header=1)

    np_lambdas_J_eq_1 = np_data_J_eq_1[:, 0]
    # np_lambdas_J_eq_2 = np_data_J_eq_2[:, 0]
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

    plt.title(r"Lipkin Ground State Energy vs VQE Estimate (HEA)")
    plt.ylabel(r"$E/\epsilon$")
    plt.xlabel(r"$V/\epsilon$")
    plt.legend()
    plt.savefig("images/problem_g_our_code_vs_numpy_hea.png")
    plt.show()


    # Complicated ansatz
    np_data_J_eq_2 = np.genfromtxt("output/np_lipkin_J_eq_2.csv", delimiter=",", skip_header=1)



    our_code_data_J_eq_2 = np.genfromtxt("output/lipkin_J_eq_2_alternate_complicated_ansatz.csv", delimiter=",", skip_header=1)

    our_code_lambdas_J_eq_2 = our_code_data_J_eq_2[:, 0]
    our_code_energy_J_eq_2 = our_code_data_J_eq_2[:, 1]

    # make plot and place points on top
    plt.plot(np_lambdas_J_eq_1, np_energy_J_eq_2, label=r"$J = 2$ with NumPy", color="royalblue", zorder=2)
    
    plt.scatter(our_code_lambdas_J_eq_2, our_code_energy_J_eq_2, label=r"$J = 2$ with VQE (our code)", marker="x", c="crimson", zorder=4)

    plt.title(r"Lipkin Ground State Energy vs VQE Estimate (Complicated Ansatz)")
    plt.ylabel(r"$E/\epsilon$")
    plt.xlabel(r"$V/\epsilon$")
    plt.legend()
    plt.savefig("images/problem_g_our_code_vs_numpy_complicated_ansatz.png")
    plt.show()


    # plot with Qiskit results only (uncomment if changes need to be made, takes time to run)
    interaction_strengths = np.linspace(0, 1, 10)
    J = [1, 2]
    qbits = [2,4]
    
    eigenval_J1 = []
    eigenval_J2 = []

    for j, k in zip(J, qbits): 
        ansatz = TwoLocal(num_qubits=k, rotation_blocks="ry", entanglement_blocks="cz", reps=1, entanglement="linear")
        # show ansatzes
        # ansatz.decompose().draw("mpl")
        # plt.show()
            
        for i in interaction_strengths:
            H_op = SparseLipkinHamiltonian(j, i)
            vqe = VQE(estimator, ansatz, optimizer)
            result = vqe.compute_minimum_eigenvalue(H_op)

            if j == 1:
                eigenval_J1.append(result.eigenvalue)
            else:
                eigenval_J2.append(result.eigenvalue)

    # plot of qiskit results
    plt.plot(interaction_strengths, eigenval_J1, label="J = 1", color="seagreen")
    plt.plot(interaction_strengths, eigenval_J2, label="J = 2", color="royalblue")
    plt.title(r"Lipkin Ground State Energy vs VQE Estimate, using Qiskit")
    plt.xlabel(r"$V/\epsilon$")
    plt.ylabel(r"$E/\epsilon$")
    plt.legend()
    plt.savefig("images/problem_g_qiskit.png")
    plt.show()

problem_g()