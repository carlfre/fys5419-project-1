import numpy as np
import matplotlib.pyplot as plt
from utils import write_to_csv

# defining Hamiltonian
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
    component_0 = []
    component_1 = []

    for lmbda in lambdas:
        H = H0 + lmbda * HI
        eigvals, eigvecs = np.linalg.eig(H)

        lower_eigs.append(np.min(eigvals))
        upper_eigs.append(np.max(eigvals))
        
        min_eigenvalue_index = np.argmin(eigvals)
        min_eigenvector = eigvecs[:, min_eigenvalue_index]
    
        component_0.append(min_eigenvector[0]) 
        component_1.append(min_eigenvector[1])

    # getting rid of degree of freedom from choice of phase
    component_0[0] *= -1

    # upper and lower eigenvalues
    plt.plot(lambdas, upper_eigs, label="Upper eigenvalue", color="teal")
    plt.plot(lambdas, lower_eigs, label="Lower eigenvalue", color="mediumaquamarine")
    plt.axvline(x=2/3, color='r', linestyle='--', label=r"$\lambda = \frac{2}{3}$")
    plt.title("Eigenenergy vs. $\\lambda$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Eigenenergy")
    plt.legend()
    plt.savefig("images/problem_b.png")
    plt.show()

    # save eigenvalue data for plotting in problem c
    write_to_csv([lambdas, lower_eigs], ["lambdas", "lower_eigs"], "output/np_simple_1_qubit.csv")    
    
    # eigenvector component plot
    plt.plot(lambdas, component_0, label="Contribution of |0⟩", color="plum")
    plt.plot(lambdas, component_1, label="Contribution of |1⟩", color="darkmagenta")
    plt.axvline(x=2/3, color='r', linestyle='--', label=r"$\lambda = \frac{2}{3}$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Component Value")
    plt.title(r"Eigenvector Components with Varying $\lambda$")
    plt.legend()
    plt.savefig("images/problem_b_eigenvectors.png")
    plt.show()

problem_b()