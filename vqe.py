import multiprocessing
from time import time


import numpy as np
from scipy.optimize import minimize

from gates import multi_kron, identity_gate, pauli_x_gate, pauli_y_gate, pauli_z_gate, hadamard_gate, SWAP_gate, CX_10_gate, phase_gate
from ansatzes import one_qubit_ansatz, hardware_efficient_2_qubit_ansatz, hardware_efficient_4_qubit_ansatz, repeated_hae_gate_4_qubit_ansatz
from utils import write_to_csv

I, X, Y, Z, H, SWAP, CX_10, S = identity_gate(), pauli_x_gate(), pauli_y_gate(), pauli_z_gate(), hadamard_gate(), SWAP_gate(), CX_10_gate(), phase_gate()


def measure_first_qubit(ket: np.ndarray, n_shots: int) -> np.ndarray:
    """Measures the first qubit of a collection of states

    """
    dim = ket.shape[0]
    probs = np.abs(ket)**2
    first_qubit_eq_0_prob = np.sum(probs[:dim//2])
    uniform_samples = np.random.rand(n_shots)
    return np.where(uniform_samples < first_qubit_eq_0_prob, 0, 1)

def measurement_to_energy(measurements: np.ndarray):
    return -2* measurements.mean() + 1



def estimate_pauli_expval(psi_initial: np.ndarray, U: np.ndarray, n_shots: int) -> float:
    measurements = measure_first_qubit(U @ psi_initial, n_shots)
    return measurement_to_energy(measurements)


def vqe_simple_1_qubit_hamiltonian(
    E1: float,
    E2: float,
    V11: float,
    V22: float,
    V_offdiag: float,
    lmbda: float,
    n_shots: int = 10_000,
) -> float:
    
    Eps = (E1 + E2) / 2
    omega = (E1 - E2) / 2

    c = (V11 + V22) / 2
    omega_z = (V11 - V22) / 2
    omega_x = V_offdiag

    U_Z = I
    U_X = H

    def expected_value(theta: np.ndarray, n_shots: int) -> float:
        non_interacting_energy = 0
        interacting_energy = 0

        ket = one_qubit_ansatz(*theta)

        non_interacting_energy += Eps
        non_interacting_energy += omega * estimate_pauli_expval(ket, U_Z, n_shots)

        interacting_energy += c
        interacting_energy += omega_x * estimate_pauli_expval(ket, U_X, n_shots)
        interacting_energy += omega_z * estimate_pauli_expval(ket, U_Z, n_shots)

        return non_interacting_energy + lmbda * interacting_energy
    
    theta0 = np.random.rand(2) * (2 * np.pi) # Random initial guess
    res = minimize(expected_value, theta0, args=(n_shots), method="Powell")
    theta_opt = res.x

    return expected_value(theta_opt, n_shots)

    




def vqe_simple_2_qubit_hamiltonian(
        eps_00: float,
        eps_01: float,
        eps_10: float,
        eps_11: float,
        Hx: float,
        Hz: float,
        lmbda: float,
        n_shots=10_000,
    ) -> float:

    c_II = (eps_00 + eps_01 + eps_10 + eps_11) / 4
    c_IZ = (eps_00 - eps_01 + eps_10 - eps_11) / 4
    c_ZI = (eps_00 + eps_01 - eps_10 - eps_11) / 4
    c_ZZ = (eps_00 - eps_01 - eps_10 + eps_11) / 4

    U_IZ = SWAP
    U_ZI = np.kron(I, I)
    U_ZZ = CX_10

    U_XX = CX_10 @ np.kron(H, H)
    U_ZZ = CX_10

    def expected_value(theta: np.ndarray, n_shots: int) -> float:
        non_interacting_energy = 0
        interacting_energy = 0


        ket = hardware_efficient_2_qubit_ansatz(*theta)

        # Non-interacting Hamiltonian
        non_interacting_energy += c_II
        non_interacting_energy += c_IZ * estimate_pauli_expval(ket, U_IZ, n_shots)
        non_interacting_energy += c_ZI * estimate_pauli_expval(ket, U_ZI, n_shots)
        non_interacting_energy += c_ZZ * estimate_pauli_expval(ket, U_ZZ, n_shots)

        # Interacting Hamiltonian
        interacting_energy += Hx * estimate_pauli_expval(ket, U_XX, n_shots)
        interacting_energy += Hz * estimate_pauli_expval(ket, U_ZZ, n_shots)

        return non_interacting_energy + lmbda * interacting_energy


    theta0 = np.random.rand(4) * (2 * np.pi) # Random initial guess
    res = minimize(expected_value, theta0, args=(n_shots), method="Powell")
    theta_opt = res.x

    return expected_value(theta_opt, n_shots)


def vqe_lipkin_J_eq_1(
        eps: float,
        V: float,
        n_shots: int = 10_000,
    ) -> float:

    U_ZI = np.kron(I, I)
    U_IZ = SWAP

    U_XX = CX_10 @ np.kron(H, H)
    U_YY = CX_10 @ np.kron(H @ S.T.conj(), H @ S.T.conj())

    def expected_value(theta: np.ndarray, n_shots: int) -> float:
        term0 = 0
        term1 = 0

        ket = hardware_efficient_2_qubit_ansatz(*theta)

        term0 += estimate_pauli_expval(ket, U_ZI, n_shots)
        term0 += estimate_pauli_expval(ket, U_IZ, n_shots)

        term1 += estimate_pauli_expval(ket, U_XX, n_shots)
        term1 -= estimate_pauli_expval(ket, U_YY, n_shots)

        return eps/2 * term0 - V/2 * term1
    
    theta0 = np.random.rand(4) * (2 * np.pi) # Random initial guess
    res = minimize(expected_value, theta0, args=(n_shots), method="Powell")
    theta_opt = res.x

    return expected_value(theta_opt, n_shots)

def vqe_lipkin_J_eq_2(eps: float, V: float, n_shots: int = 10000, use_hea: bool = True) -> float:

    U_ZIII = multi_kron(I, I, I, I)
    U_IZII = multi_kron(SWAP, I, I)
    U_IIZI =  multi_kron(SWAP, I, I) @ multi_kron(I, SWAP, I)
    U_IIIZ = multi_kron(SWAP, I, I) @ multi_kron(I, SWAP, I) @ multi_kron(I, I, SWAP)

    U_XXII = multi_kron(CX_10, I, I) @ multi_kron(H, H, I, I)
    U_XIXI = multi_kron(CX_10, I, I) @ multi_kron(H, I, I, I) @ multi_kron(I, H, I, I) @ multi_kron(I, SWAP, I)
    U_XIIX = multi_kron(CX_10, I, I) @ multi_kron(H, I, I, I) @ multi_kron(I, SWAP, I) @ multi_kron(I, I, H, I) @ multi_kron(I, I, SWAP)
    U_IXXI = multi_kron(SWAP, I, I) @ multi_kron(I, CX_10, I) @ multi_kron(I, H, H, I)
    U_IXIX = multi_kron(SWAP, I, I) @ multi_kron(I, CX_10, I) @ multi_kron(I, H, I, I) @ multi_kron(I, I, H, I) @ multi_kron(I, I, SWAP)
    U_IIXX = multi_kron(SWAP, I, I) @ multi_kron(I, SWAP, I) @ multi_kron(I, I, CX_10) @ multi_kron(I, I, H, H)

    U_YYII = multi_kron(CX_10, I, I) @ multi_kron(H @ S.T.conj(), H @ S.T.conj(), I, I)
    U_YIYI = multi_kron(CX_10, I, I) @ multi_kron(H @ S.T.conj(), I, I, I) @ multi_kron(I, np.kron(H @ S.T.conj(), I) @ SWAP, I)
    U_YIIY = multi_kron(CX_10, I, I) @ multi_kron(H @ S.T.conj(), I, I, I) @ multi_kron(I, SWAP, I) \
                @ multi_kron(I, I, np.kron(H @ S.T.conj(), I) @ SWAP)
    U_IYYI = multi_kron(SWAP, I, I) @ multi_kron(I, CX_10 @ np.kron(H @ S.T.conj(), H @ S.T.conj()), I)
    U_IYIY = multi_kron(SWAP, I, I) @ multi_kron(I, CX_10 @ np.kron(H @ S.T.conj(), I), I) \
                @ multi_kron(I, I, np.kron(H @ S.T.conj(), I) @ SWAP)
    U_IIYY = multi_kron(SWAP, I, I) @ multi_kron(I, SWAP, I) @ multi_kron(I, I, CX_10 @ np.kron(H @ S.T.conj(), H @ S.T.conj()))

    def expected_value(theta: np.ndarray, n_shots: int) -> float:
        term0 = 0
        term1 = 0

        if use_hea:
            ket = hardware_efficient_4_qubit_ansatz(*theta)
        else:
            ket = repeated_hae_gate_4_qubit_ansatz(*theta)

        term0 += estimate_pauli_expval(ket, U_ZIII, n_shots)
        term0 += estimate_pauli_expval(ket, U_IZII, n_shots)
        term0 += estimate_pauli_expval(ket, U_IIZI, n_shots)
        term0 += estimate_pauli_expval(ket, U_IIIZ, n_shots)

        term1 += estimate_pauli_expval(ket, U_XXII, n_shots)
        term1 += estimate_pauli_expval(ket, U_XIXI, n_shots)
        term1 += estimate_pauli_expval(ket, U_XIIX, n_shots)
        term1 += estimate_pauli_expval(ket, U_IXXI, n_shots)
        term1 += estimate_pauli_expval(ket, U_IXIX, n_shots)
        term1 += estimate_pauli_expval(ket, U_IIXX, n_shots)

        term1 -= estimate_pauli_expval(ket, U_YYII, n_shots)
        term1 -= estimate_pauli_expval(ket, U_YIYI, n_shots)
        term1 -= estimate_pauli_expval(ket, U_YIIY, n_shots)
        term1 -= estimate_pauli_expval(ket, U_IYYI, n_shots)
        term1 -= estimate_pauli_expval(ket, U_IYIY, n_shots)
        term1 -= estimate_pauli_expval(ket, U_IIYY, n_shots)

        return eps/2 * term0 - V/2 * term1
    
    if use_hea:
        theta0 = np.random.rand(8) * (2 * np.pi) # Random initial guess
    else:
        theta0 = np.random.rand(16) * (2 * np.pi)
    res = minimize(expected_value, theta0, args=(n_shots), method="Powell")
    theta_opt = res.x

    return expected_value(theta_opt, n_shots)



def vqe_lipkin_J_eq_1_alternate(eps: float, V: float, n_shots: int = 10_000) -> float:
    U_Z = I
    U_X = H

    def expected_value(theta: np.ndarray, n_shots: int) -> float:
        psi_initial = one_qubit_ansatz(*theta)

        energy = 0
        energy += - eps * estimate_pauli_expval(psi_initial, U_Z, n_shots)
        energy += - V * estimate_pauli_expval(psi_initial, U_X, n_shots)

        return energy
    
    theta0 = np.random.rand(2) * (2 * np.pi)
    res = minimize(expected_value, theta0, args=(n_shots), method="Powell")
    theta_opt = res.x

    return expected_value(theta_opt, n_shots)


def vqe_lipkin_J_eq_2_alternate(eps: float, V: float, W: float, n_shots: int = 10_000) -> float:

    U_IZ = SWAP
    U_ZI = np.kron(I, I)
    U_ZZ = CX_10
    U_ZZ = CX_10
    U_IX = np.kron(H, I) @ SWAP
    U_XX = CX_10 @ np.kron(H, H)
    U_YY = CX_10 @ np.kron(H @ S.T.conj(), H @ S.T.conj())
    U_ZX = CX_10 @ np.kron(I, H)


    def expected_value_block_1(theta: np.ndarray, n_shots: int) -> float:
        psi_initial = hardware_efficient_2_qubit_ansatz(*theta)

        eps_term = 0
        W_term = 0
        V_term = 0

        eps_term += estimate_pauli_expval(psi_initial, U_ZI, n_shots)
        eps_term += estimate_pauli_expval(psi_initial, U_ZZ, n_shots)

        W_term += 1
        W_term += estimate_pauli_expval(psi_initial, U_ZI, n_shots)
        W_term += - estimate_pauli_expval(psi_initial, U_IZ, n_shots)
        W_term += - estimate_pauli_expval(psi_initial, U_ZZ, n_shots)

        V_term += estimate_pauli_expval(psi_initial, U_IX, n_shots)
        V_term += estimate_pauli_expval(psi_initial, U_ZX, n_shots)
        V_term += estimate_pauli_expval(psi_initial, U_XX, n_shots)
        V_term += estimate_pauli_expval(psi_initial, U_YY, n_shots)


        return -eps * eps_term + W * W_term - np.sqrt(6) * V / 2 * V_term
    

    U_Z = I
    U_X = H
    def expected_value_block_2(theta: np.ndarray, n_shots: int) -> float:
        psi_initial = one_qubit_ansatz(*theta)

        energy = 0
        energy += 3 * W
        energy += - eps * estimate_pauli_expval(psi_initial, U_Z, n_shots)
        energy += 3 * V * estimate_pauli_expval(psi_initial, U_X, n_shots)

        return energy
    

    theta0_block1 = np.random.rand(4) * (2 * np.pi)
    theta0_block2 = np.random.rand(2) * (2 * np.pi)

    res_block1 = minimize(expected_value_block_1, theta0_block1, args=(n_shots), method="Powell")
    res_block2 = minimize(expected_value_block_2, theta0_block2, args=(n_shots), method="Powell")

    theta_opt_block1 = res_block1.x
    theta_opt_block2 = res_block2.x

    min_energy_block1 = expected_value_block_1(theta_opt_block1, n_shots)
    min_energy_block2 = expected_value_block_2(theta_opt_block2, n_shots)

    return min(min_energy_block1, min_energy_block2)



def run_vqe_simple_1_qubit_hamiltonian(lmbda: float, n_shots: int = 10_000):
    E1 = 0
    E2 = 4
    V11 = 3
    V22 = -3
    V_offdiag = 0.2

    energy = vqe_simple_1_qubit_hamiltonian(
        E1,
        E2,
        V11,
        V22,
        V_offdiag,
        lmbda,
        n_shots=n_shots,
    )
    return energy



def run_vqe_simple_2_qubit_hamiltonian(lmbda: float, n_shots: int = 10_000) -> float:
    eps_00 = 0
    eps_01 = 2.5
    eps_10 = 6.5
    eps_11 = 7
    Hx = 2
    Hz = 3

    energy = vqe_simple_2_qubit_hamiltonian(
        eps_00,
        eps_01,
        eps_10,
        eps_11,
        Hx,
        Hz,
        lmbda,
        n_shots=n_shots,
    )
    return energy




def run_vqes():
    n_shots=100_000

    t_init = time()
    lambdas = np.linspace(0, 1, 20)
    energies_simple_1_qubit = [run_vqe_simple_1_qubit_hamiltonian(lmbda, n_shots=n_shots) for lmbda in lambdas]
    write_to_csv([lambdas, energies_simple_1_qubit], ["lmbda", "energy"], "output/simple_1_qubit_new.csv")
    print("Done with simple 1 qubit. Time taken: ", time() - t_init)

    t_init = time()
    lambdas = np.linspace(0, 1, 20)
    energies_simple_2_qubit = [run_vqe_simple_2_qubit_hamiltonian(lmbda, n_shots=n_shots) for lmbda in lambdas]
    write_to_csv([lambdas, energies_simple_2_qubit], ["lmbda", "energy"], "output/simple_2_qubit_new.csv")
    print("Done with simple 2 qubit. Time taken: ", time() - t_init)


    t_init = time()
    V_over_eps = np.linspace(0, 1, 20)
    energies_J_eq_1 = [vqe_lipkin_J_eq_1(1, V, n_shots=n_shots) for V in V_over_eps]
    write_to_csv([V_over_eps, energies_J_eq_1], ["V_over_eps", "energy_over_eps"], "output/lipkin_J_eq_1_new.csv")
    print("Done with Lipkin, J=1. Time taken: ", time() - t_init)

    t_init = time()
    V_over_eps = np.linspace(0, 1, 20)

    def apply_vqe_lipkin_J_eq_2(V):
        return vqe_lipkin_J_eq_2(1, V, n_shots=n_shots, use_hea=True)

    with multiprocessing.Pool(processes=16) as pool:
        energies_J_eq_2 = pool.map(apply_vqe_lipkin_J_eq_2, V_over_eps)

    write_to_csv([V_over_eps, energies_J_eq_2], ["V_over_eps", "energy_over_eps"], "output/lipkin_J_eq_2_new.csv")
    print("Done with Lipkin, J=2. Time taken: ", time() - t_init)




def run_vqes_alternate():
    n_shots = 100_000

    t_init = time()
    V_over_eps = np.linspace(0, 1, 20)
    energies_J_eq_1 = [vqe_lipkin_J_eq_1_alternate(1, V, n_shots=n_shots) for V in V_over_eps]
    write_to_csv([V_over_eps, energies_J_eq_1], ["V_over_eps", "energy_over_eps"], "output/lipkin_J_eq_1_alternate.csv")
    print("Done with Lipkin, J=1. Time taken: ", time() - t_init)

    t_init = time()
    V_over_eps = np.linspace(0, 1, 20)
    energies_J_eq_2 = [vqe_lipkin_J_eq_2_alternate(1, V, 0, n_shots=n_shots) for V in V_over_eps]
    write_to_csv([V_over_eps, energies_J_eq_2], ["V_over_eps", "energy_over_eps"], "output/lipkin_J_eq_2_alternate.csv")
    print("Done with Lipkin, J=2. Time taken: ", time() - t_init)





if __name__ == "__main__":
    run_vqes_alternate()










