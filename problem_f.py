import numpy as np
import matplotlib.pyplot as plt


def Jz(J):
    m_values = np.arange(J, -J-1, -1)
    return np.diag(m_values)


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


def problem_f() -> None:

    interaction_strengths = np.linspace(0, 1, 10)
    epsilon = 0.5

    eigenval_J1 = []
    eigenval_J2 = []

    for i in interaction_strengths: 
        # Define the operators for J = 1
        J = 1

        # Hamiltonian for J = 1
        V = i  # Example interaction strength
        H = - epsilon * Jz(J) + 0.5 * V * (np.dot(J_plus(J), J_plus(J)) + np.dot(J_minus(J), J_minus(J)))

        # Calculate eigenvalues
        w, v = np.linalg.eigh(H)
        
        # Storing eigenvalues for J = 1
        eigenval_J1.append(np.min(w))

        # Define the operators for J = 2
        J2 = 2

        # Hamiltonian for J = 2
        H_J2 = - epsilon * Jz(J2) + 0.5 * V * (np.dot(J_plus(J2), J_plus(J2)) + np.dot(J_minus(J2), J_minus(J2)))

        # Calculate eigenvalues for J = 2
        w_J2, v_J2 = np.linalg.eigh(H_J2)

        # Storing eigenvalues for J = 2
        eigenval_J2.append(np.min(w_J2))

    plt.plot(interaction_strengths, eigenval_J1, label="J = 1")
    plt.plot(interaction_strengths, eigenval_J2, label="J = 2")

    plt.title("Eigenvalues vs interaction strengths for Lipkin Hamiltonian")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Eigenenergy")
    plt.legend()
    plt.savefig("images/problem_f.png")
    plt.show()

problem_f()