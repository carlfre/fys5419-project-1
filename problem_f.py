import numpy as np
import matplotlib.pyplot as plt

from utils import write_to_csv

# Define Pauli matrices and identity matrix
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.array([[1, 0], [0, 1]])


def problem_f() -> None:
    interaction_strengths = np.linspace(0, 1, 10)
    epsilon = 1

    eigenval_J1 = []
    eigenval_J2 = []

    for V in interaction_strengths: 
        W = 0

        HJ1 = np.array([[-epsilon, 0, -V],
              [0, 0, 0],
              [-V, 0, epsilon],])

        HJ2 = np.array([[-2*epsilon, 0, np.sqrt(6)*V, 0, 0],
              [0, -epsilon + 3*W, 0, 3*V, 0],
              [np.sqrt(6)*V, 0, 4*W, 0, np.sqrt(6)*V],
              [0, 3*V, 0, -epsilon + 3*W, 0],
              [0, 0, np.sqrt(6)*V, 0, 2*epsilon],])
        
        #HJ1_rewritten = -epsilon * np.kron(sigma_z, identity) - V * np.kron(sigma_x, identity)

        # HJ2_rewritten = (-2 * epsilon * np.kron(sigma_z, np.kron(identity, identity)) +
        #           np.sqrt(6) * V * np.kron(sigma_x, np.kron(identity, identity)) +
        #           3 * W * np.kron(identity, np.kron(sigma_z, identity)))


        w_J1, v_J1 = np.linalg.eigh(HJ1)
        w_J2, v_J2 = np.linalg.eigh(HJ2)
        
        eigenval_J1.append(np.min(w_J1))
        eigenval_J2.append(np.min(w_J2))

    plt.plot(interaction_strengths, eigenval_J1, label="J = 1")
    plt.plot(interaction_strengths, eigenval_J2, label="J = 2")

    plt.title(r"Eigenenergy vs. $\lambda$ for Lipkin Hamiltonians")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Eigenenergy")
    plt.legend()
    plt.savefig("images/problem_f.png")
    plt.show()

    write_to_csv([interaction_strengths, eigenval_J1], ["lambdas", "J1"], "output/np_lipkin_small.csv")
    write_to_csv([interaction_strengths, eigenval_J2], ["lambdas", "J2"], "output/np_lipkin_big.csv")

problem_f()