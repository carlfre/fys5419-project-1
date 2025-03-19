import numpy as np

from gates import RX_gate, RY_gate, cnot_gate, multi_kron, identity_gate


def one_qubit_ansatz(theta0: float, theta1) -> np.ndarray:
    start_ket = np.array([1, 0])
    Rx = RX_gate(theta0)
    Ry = RY_gate(theta1)
    return Ry @ Rx @ start_ket


def hardware_efficient_2_qubit_ansatz(theta00: float, theta01: float, theta10: float, theta11: float) -> np.ndarray:
    start_ket = np.array([1, 0, 0, 0])

    Rx = np.kron(RX_gate(theta00), RX_gate(theta10))
    Ry = np.kron(RY_gate(theta01), RY_gate(theta11))
    CNOT = cnot_gate(first_is_control=True)

    circuit = CNOT @ Ry @ Rx

    return circuit @ start_ket


def hardware_efficient_4_qubit_ansatz(theta00: float, theta01: float, theta10: float, theta11: float, theta20: float, theta21: float, theta30: float, theta31: float) -> np.ndarray:
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


def repeated_hae_gate_4_qubit_ansatz(theta00: float, theta01: float, theta10: float, theta11: float, theta20: float, theta21: float, theta30: float, theta31: float, 
                      phi00: float, phi01: float, phi10: float, phi11: float, phi20: float, phi21: float, phi30: float, phi31: float):
    
    start_ket = np.zeros(2**4)
    start_ket[0] = 1

    Rx1 = multi_kron(RX_gate(theta00), RX_gate(theta10), RX_gate(theta20), RX_gate(theta30))
    Ry1 = multi_kron(RY_gate(theta01), RY_gate(theta11), RY_gate(theta21), RY_gate(theta31))

    I = identity_gate()
    CNOT = cnot_gate(first_is_control=True)

    CNOT_1 = multi_kron(CNOT, I, I)
    CNOT_2 = multi_kron(I, CNOT, I)
    CNOT_3 = multi_kron(I, I, CNOT)

    CNOTS = CNOT_3 @ CNOT_2 @ CNOT_1


    Rx2 = multi_kron(RX_gate(phi00), RX_gate(phi10), RX_gate(phi20), RX_gate(phi30))
    Ry2 = multi_kron(RY_gate(phi01), RY_gate(phi11), RY_gate(phi21), RY_gate(phi31))

    circuit = CNOTS @ Ry2 @ Rx2 @ CNOTS @ Ry1 @ Rx1

    return circuit @ start_ket

