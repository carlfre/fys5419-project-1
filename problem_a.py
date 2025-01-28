import numpy as np
import matplotlib.pyplot as plt
import qiskit as qk


from state_initialization import one_qubit_basis, bell_psi_plus
from gates import identity_gate, pauli_x_gate, pauli_y_gate, pauli_z_gate, hadamard_gate, cnot_gate

def measure_first_qubit(state: np.ndarray) -> tuple[int, np.ndarray]:
    alpha, beta, gamma, delta = state

    prob_0 = np.abs(alpha)**2 + np.abs(beta)**2

    if np.random.rand() < prob_0:
        new_state = np.array([alpha, beta, 0, 0])
        return 0, new_state / np.linalg.norm(new_state)
    else:
        new_state = np.array([0, 0, gamma, delta])
        return 1, new_state / np.linalg.norm(new_state)
    

def measure_second_qubit(state: np.ndarray) -> tuple[int, np.ndarray]:
    alpha, beta, gamma, delta = state

    prob_0 = np.abs(alpha)**2 + np.abs(gamma)**2

    if np.random.rand() < prob_0:
        new_state = np.array([alpha, 0, gamma, 0])
        return 0, new_state / np.linalg.norm(new_state)
    else:
        new_state = np.array([0, beta, 0, delta])
        return 1, new_state / np.linalg.norm(new_state)


def problem_a() -> None:
    np.random.seed(973)
    
    zero, one = one_qubit_basis()
    X = pauli_x_gate()
    Y = pauli_y_gate()
    Z = pauli_z_gate()


    for ket_string, state in zip(["|0>", "|1>"], [zero, one]):
        for gate_name, gate in zip(["X", "Y", "Z"], [X, Y, Z]):
            after_gate = gate @ state
            print(f"{gate_name}{ket_string} = transpose({after_gate})")

    psi_plus = bell_psi_plus()
    H = hadamard_gate()
    CNOT = cnot_gate()

    result = CNOT @ np.kron(H, H) @ psi_plus

    N = 1000

    measurements = {
        (0, 0): 0,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 0
    }

    for i in range(N):
        first_qubit, after_first_measurement = measure_first_qubit(result)
        second_qubit, after_second_measurement = measure_second_qubit(after_first_measurement)
        measurements[(first_qubit, second_qubit)] += 1

    # Convert the tuple keys into string labels for better display in the plot
    labels = [str(key) for key in measurements.keys()]
    counts = list(measurements.values())

    # Create the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color='skyblue')

    # Adding text labels on top of each bar for clarity
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{int(yval)}', ha='center', va='bottom')

    # Add title and axis labels
    plt.title("Measurement Results Histogram")
    plt.xlabel("Measurement Outcome (First Qubit, Second Qubit)")
    plt.ylabel("Counts")
    plt.ylim(0, max(counts) * 1.2)  # Add some headroom for text labels
    plt.savefig("images/problem_a.png")
    # Show the plot
    plt.show()

    



    # print(f"final state = {after_second_measurement}")

    






    





problem_a()

# def bell_states()
