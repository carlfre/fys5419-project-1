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

# Define the angular momentum operators

def Jz(J):
    m_values = np.arange(J, -J-1, -1)
    return np.diag(m_values)
X = Jz(1) 
print (X[0,0])

def J_plus(J):
    dim = int(2 * J + 1)
    J_plus_matrix = np.zeros((dim, dim))
    for m in range(dim - 1):
        value = (J - m) * (J + m + 1)
        if value > 0:
            J_plus_matrix[m, m + 1] = np.sqrt(value)
    return J_plus_matrix


def J_minus(J):
    return J_plus(J).T

# Updated Lipkin Hamiltonian using angular momentum operators
def lipkin_hamiltonian(J, V):
    epsilon = 0.5
    return Operator(- epsilon * Jz(J) + 0.5 * V * (np.dot(J_plus(J), J_plus(J)) + np.dot(J_minus(J), J_minus(J))))

def SparseLipkinHamiltonian(J, V): 
    epsilon = 0.5
    #Defining the coefficients for the Pauli terms. Have to convert the Hamiltonian to a SparsePauliOp to use VQE.
    if J == 1:
        coefficients = [
        ("XX", - epsilon * Jz(J)[0,0] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[0,0] + np.dot(J_minus(J), J_minus(J))[0,0]),
        ("XY", - epsilon * Jz(J)[0,1] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[0,1] + np.dot(J_minus(J), J_minus(J))[0,1]),
        ("XZ", - epsilon * Jz(J)[0,2] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[0,2] + np.dot(J_minus(J), J_minus(J))[0,2]),
        ("YX", - epsilon * Jz(J)[1,0] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[1,0] + np.dot(J_minus(J), J_minus(J))[1,0]),
        ("YY", - epsilon * Jz(J)[1,1] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[1,1] + np.dot(J_minus(J), J_minus(J))[1,1]),
        ("YZ", - epsilon * Jz(J)[1,2] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[1,2] + np.dot(J_minus(J), J_minus(J))[1,2]),
        ("ZX", - epsilon * Jz(J)[2,0] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[2,0] + np.dot(J_minus(J), J_minus(J))[2,0]),
        ("ZY", - epsilon * Jz(J)[2,1] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[2,1] + np.dot(J_minus(J), J_minus(J))[2,1]),
        ("ZZ", - epsilon * Jz(J)[2,2] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[2,2] + np.dot(J_minus(J), J_minus(J))[2,2]),
    ]
        # Create the SparsePauliOp
        H_op = SparsePauliOp.from_list(coefficients)

    elif J == 2:
        coefficients = [
        ("XX", - epsilon * Jz(J)[0,0] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[0,0] + np.dot(J_minus(J), J_minus(J))[0,0]),
        ("XY", - epsilon * Jz(J)[0,1] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[0,1] + np.dot(J_minus(J), J_minus(J))[0,1]),
        ("XZ", - epsilon * Jz(J)[0,2] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[0,2] + np.dot(J_minus(J), J_minus(J))[0,2]),
        ("XI", - epsilon * Jz(J)[0,3] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[0,3] + np.dot(J_minus(J), J_minus(J))[0,3]),
        ("XXX",- epsilon * Jz(J)[0,4] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[0,4] + np.dot(J_minus(J), J_minus(J))[0,4]),
        ("YX", - epsilon * Jz(J)[1,0] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[1,0] + np.dot(J_minus(J), J_minus(J))[1,0]),
        ("YY", - epsilon * Jz(J)[1,1] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[1,1] + np.dot(J_minus(J), J_minus(J))[1,1]),
        ("YZ", - epsilon * Jz(J)[1,2] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[1,2] + np.dot(J_minus(J), J_minus(J))[1,2]),
        ("YI", - epsilon * Jz(J)[1,3] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[1,3] + np.dot(J_minus(J), J_minus(J))[1,3]),
        ("YXX",- epsilon * Jz(J)[1,4] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[1,4] + np.dot(J_minus(J), J_minus(J))[1,4]),
        ("ZX", - epsilon * Jz(J)[2,0] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[2,0] + np.dot(J_minus(J), J_minus(J))[2,0]),
        ("ZY", - epsilon * Jz(J)[2,1] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[2,1] + np.dot(J_minus(J), J_minus(J))[2,1]),
        ("ZZ", - epsilon * Jz(J)[2,2] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[2,2] + np.dot(J_minus(J), J_minus(J))[2,2]),
        ("ZI", - epsilon * Jz(J)[2,3] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[2,3] + np.dot(J_minus(J), J_minus(J))[2,3]),
        ("ZXX",- epsilon * Jz(J)[2,4] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[2,4] + np.dot(J_minus(J), J_minus(J))[2,4]),
        ("IX", - epsilon * Jz(J)[3,0] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[3,0] + np.dot(J_minus(J), J_minus(J))[3,0]),
        ("IY", - epsilon * Jz(J)[3,1] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[3,1] + np.dot(J_minus(J), J_minus(J))[3,1]),
        ("IZ", - epsilon * Jz(J)[3,2] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[3,2] + np.dot(J_minus(J), J_minus(J))[3,2]),
        ("II", - epsilon * Jz(J)[3,3] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[3,3] + np.dot(J_minus(J), J_minus(J))[3,3]),
        ("IXX",- epsilon * Jz(J)[3,4] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[3,4] + np.dot(J_minus(J), J_minus(J))[3,4]),
        ("XXX",- epsilon * Jz(J)[4,0] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[4,0] + np.dot(J_minus(J), J_minus(J))[4,0]),
        ("XXY",- epsilon * Jz(J)[4,1] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[4,1] + np.dot(J_minus(J), J_minus(J))[4,1]),
        ("XXZ",- epsilon * Jz(J)[4,2] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[4,2] + np.dot(J_minus(J), J_minus(J))[4,2]),
        ("XXI",- epsilon * Jz(J)[4,3] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[4,3] + np.dot(J_minus(J), J_minus(J))[4,3]),
        ("XXXX",-epsilon * Jz(J)[4,4] + 0.5 + V * (np.dot(J_plus(J), J_plus(J)))[4,4] + np.dot(J_minus(J), J_minus(J))[4,4]),
    ]
        # Create the SparsePauliOp
        H_op = SparsePauliOp.from_list(coefficients)
    return H_op 

# Set up the VQE algorithm

# Parameters
num_qubits = 1
estimator = Estimator()
backend = Aer.get_backend('statevector_simulator')
optimizer = ADAM(maxiter=100)
ansatz = TwoLocal(num_qubits=num_qubits, rotation_blocks='ry', entanglement_blocks='cz', reps=1, entanglement='linear')

def problem_g():

    interaction_strengths = np.linspace(0, 1, 10)
    J = [1, 2]
    
    eigenval_J1 = []
    eigenval_J2 = []

    for i in interaction_strengths:
        for j in J: 

            # Define the Hamiltonian 
            H = SparseLipkinHamiltonian(j, i)


            # Set up the VQE
            vqe = VQE(estimator, ansatz, optimizer)

            # Run the VQE
            result = vqe.compute_minimum_eigenvalue(operator=H)

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