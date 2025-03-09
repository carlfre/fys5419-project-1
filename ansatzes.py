import numpy as np

from gates import RX_gate, RY_gate, cnot_gate, multi_kron, identity_gate


def hardware_efficient_2_qubit(theta00: float, theta01: float, theta10: float, theta11: float) -> np.ndarray:
    start_ket = np.array([1, 0, 0, 0])

    Rx = np.kron(RX_gate(theta00), RX_gate(theta10))
    Ry = np.kron(RY_gate(theta01), RY_gate(theta11))
    CNOT = cnot_gate(first_is_control=True)

    circuit = CNOT @ Ry @ Rx

    return circuit @ start_ket


def hardware_efficient_4_qubit(theta00: float, theta01: float, theta10: float, theta11: float, theta20: float, theta21: float, theta30: float, theta31: float) -> np.ndarray:
    start_ket = np.zeros(2**4)
    start_ket[0] = 1

    Rx = multi_kron(RX_gate(theta00), RX_gate(theta10), RX_gate(theta20), RX_gate(theta30))
    Ry = multi_kron(RY_gate(theta01), RY_gate(theta11), RY_gate(theta21), RY_gate(theta31))

    I = identity_gate()
    CNOT = cnot_gate(first_is_control=True)

    CNOT_1 = multi_kron(CNOT, I, I)
    CNOT_2 = multi_kron(I, CNOT, I)
    CNOT_3 = multi_kron(I, I, CNOT)

    circuit = CNOT_3 @ CNOT_2 @ CNOT_1 @ Ry @ Rx
    return circuit @ start_ket

