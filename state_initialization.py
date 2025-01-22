import numpy as np


def one_qubit_basis() -> list[np.ndarray]:
    return [np.array([1, 0]), np.array([0, 1])]


def bell_phi_plus() -> np.ndarray:
    """1/sqrt(2) * (|00> +|11>)"""
    return 1 / np.sqrt(2) * np.array([1, 0, 0, 1])

def bell_phi_minus() -> np.ndarray:
    """1/sqrt(2) * (|00> -|11>)"""
    return 1 / np.sqrt(2) * np.array([1, 0, 0, -1])

def bell_psi_plus() -> np.ndarray:
    """1/sqrt(2) * (|10> + |01>)"""
    return 1 / np.sqrt(2) * np.array([0, 1, 1, 0])

def bell_psi_minus() -> np.ndarray:
    """1/sqrt(2) * (|10> - |01>)"""
    return 1 / np.sqrt(2) * np.array([0, 1, -1, 0])


def bell_states() -> list[np.ndarray]:
    return [
        bell_phi_plus(),
        bell_phi_minus(),
        bell_psi_plus(),
        bell_psi_minus()
    ]