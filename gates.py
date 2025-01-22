import numpy as np

def identity_gate() -> np.ndarray:
    return np.array([[1, 0], [0, 1]])


def pauli_x_gate() -> np.ndarray:
    return np.array([[0, 1], [1, 0]])


def pauli_y_gate() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]])


def pauli_z_gate() -> np.ndarray:
    return np.array([[1, 0], [0, -1]])


def hadamard_gate() -> np.ndarray:
    return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])


def phase_gate(phi: float) -> np.ndarray:
    #TODO: check which one we should use
    # return np.array([[1, 0], [0, np.exp(1j * phi)]])
    return np.array([[1, 0], [0, 1j]])


def cnot_gate() -> np.ndarray:
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])