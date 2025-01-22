#From chatgpt

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a Quantum Circuit with 1 qubit and 1 classical bit
qc = QuantumCircuit(1, 1)

# Apply a Hadamard gate to put the qubit into superposition
qc.h(0)

qc.x(0)

# Measure the qubit
qc.measure(0, 0)

# Draw the circuit
print(qc.draw())

# # Use Aer's qasm_simulator
# simulator = Aer.get_backend('qasm_simulator')

# # Transpile the circuit for the simulator
# compiled_circuit = transpile(qc, simulator)

# # Assemble the circuit into a Qobj that can be run
# qobj = assemble(compiled_circuit)

# # Execute the circuit on the qasm simulator
# result = simulator.run(qobj).result()

# # Get the counts (results)
# counts = result.get_counts(qc)
# print("\nTotal count for 0 and 1 are:", counts)

# # Plot a histogram of results
# plot_histogram(counts)
# plt.show()