import numpy as np
import matplotlib.pyplot as plt

from gates import pauli_x_gate, pauli_y_gate, pauli_z_gate
from utils import write_to_csv


sigma_x = pauli_x_gate()
sigma_y = pauli_y_gate()
sigma_z = pauli_z_gate()

E1 = 0
E2 = 4

V11 = 3
V22 = -3
V12 = V21 = 0.2

H0 = np.array([[E1, 0],
               [0, E2]])

HI = np.array([[V11, V12],
              [V21, V22]])


def problem_b() -> None:
    lambdas = np.linspace(0, 1, 101)

    lower_eigs = []
    upper_eigs = []

    for lmbda in lambdas:
        H = H0 + lmbda * HI
        eigvals = np.linalg.eigvals(H)
        lower_eigs.append(np.min(eigvals))
        upper_eigs.append(np.max(eigvals))
        print(eigvals)

    plt.plot(lambdas, lower_eigs, label="Lower eigenvalue")
    plt.plot(lambdas, upper_eigs, label="Upper eigenvalue")

    plt.axvline(x=2/3, color='r', linestyle='--', label=r"$\lambda = \frac{2}{3}$")

    plt.title("Eigenenergy vs. $\\lambda$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Eigenenergy")
    plt.legend()
    plt.savefig("images/problem_b.png")
    plt.show()

    # Take existing data lists and choose every sixteenth data point
    lambdas = [lambdas[min(round(i * (len(lambdas) / 15)), len(lambdas) - 1)] for i in range(16)]
    lower_eigs = [lower_eigs[min(round(i * (len(lower_eigs) / 15)), len(lower_eigs) - 1)] for i in range(16)]

    assert len(lambdas) == 16, "wrong length"
    assert len(lower_eigs) == 16, "wrong length"

    write_to_csv([lambdas, lower_eigs], ["lambdas", "lower_eigs"], "output/np_1_qubit.csv")    
       
problem_b()