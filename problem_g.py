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
    #Have to convert the Hamiltonian to a SparsePauliOp to use VQE.
    if J == 1:
        # Create the SparsePauliOp
        H_op = SparsePauliOp.from_operator(lipkin_hamiltonian(J, V))
    elif J == 2:
        # Create the SparsePauliOp
        H_op = SparsePauliOp.from_operator(lipkin_hamiltonian(J, V))
    return H_op 

# Set up the VQE algorithm

# Parameters
estimator = Estimator()
backend = Aer.get_backend('statevector_simulator')
optimizer = ADAM(maxiter=5000)


def problem_g():

    interaction_strengths = np.linspace(0, 1, 10)
    J = [1, 2]
    qbits = [2,4]
    
    eigenval_J1 = []
    eigenval_J2 = []

    for j, k in zip(J, qbits): 

        # Define the ansatz
        ansatz = TwoLocal(num_qubits=k, rotation_blocks='ry', entanglement_blocks='cz', reps=1, entanglement='linear')
            
        for i in interaction_strengths:
    
            # Define the Hamiltonian 
            H_op = SparseLipkinHamiltonian(j, i)

            # Set up the VQE
            vqe = VQE(estimator, ansatz, optimizer)

            # Run the VQE
            result = vqe.compute_minimum_eigenvalue(H_op)

            if j == 1:
                eigenval_J1.append(result.eigenvalue)
            else:
                eigenval_J2.append(result.eigenvalue)


    plt.plot(interaction_strengths, eigenval_J1, label="J = 1")
    plt.plot(interaction_strengths, eigenval_J2, label="J = 2")

    plt.title("Eigenvalues vs Interaction strength with VQE solution for Lipkin Hamiltonian")
    plt.xlabel("Interaction strength")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.savefig("images/problem_g.png")
    plt.show()

problem_g()