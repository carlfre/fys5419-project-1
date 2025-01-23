#From chatgpt

from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import matplotlib.pyplot as plt

# create 1 qubit and 1 classical bit for measurement
qc = QuantumCircuit(1, 1)

# apply gate (.h .x .y .z)
qc.x(0)

# measure qubit and store result in classical bit 0
qc.measure(0, 0)

# choose a backend 
# "Aer is a simulator that runs the circuit without real quantum hardware"
# ^dette får meg til å tro at vi kan gjøre det med quantum hardware også???
backend = Aer.get_backend('qasm_simulator')

# run circuit with chosen backend
transpiled_qc = transpile(qc, backend)
job = backend.run(transpiled_qc, backend, shots=2000) # shots is number of trials
#results = job.result()[0].data["c"z].get_counts()
results = job.result() #<--- denne linja er problemet
counts = results.get_counts(qc)

# make histogram
plot_histogram(counts)
plt.show()
