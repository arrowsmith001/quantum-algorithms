from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from qutip import Bloch

### VARIABLES (EDITABLE) ###

# Target state, which is a 2D array of arbitrary complex numbers
# e.g. "+" state, the state after applying a Hadamard gate to ∣0⟩: 1/sqrt(2)∣0⟩ + 1/sqrt(2)∣1⟩
s = np.array([1 / sqrt(2), 1 / sqrt(2)])

# Number of trials when running the simulator. The more trials, the closer the probabilities will be to the true probabilities.
n = 10000


### SETTING AN ARBITRARY 1-QUBIT STATE ###

# The state should be normal to satisfy |a|^2 + |b|^2 = 1. In case it isn't already, we will normalise it here.
s = s / np.linalg.norm(s)

# a and b are the probability amplitudes
a, b = s

# Create the unitary matrix
U = np.matrix([[a, b], [-np.conj(b), np.conj(a)]])

# Construct the circuit
qr = QuantumRegister(1)
cr = ClassicalRegister(1)
qc = QuantumCircuit(qr, cr)

q0 = qc.qubits[0]
qc.unitary(U, q0, "U")

qc_measured = qc.measure_all(inplace=False)
assert qc_measured is not None

# Sample and compare theory with the actual results

# Theoretical results
p_a = abs(np.square(a))
exp_a = p_a * n
p_b = abs(np.square(b))
exp_b = p_b * n
print(f"\nExpected counts:\n")
print(f"\t\t0 = {round(np.real(exp_a))} ({round(p_a*100, ndigits=1)}%)")
print(f"\t\t1 = {round(np.real(exp_b))} ({round(p_b*100, ndigits=1)}%)")

# Actual results
sampler = StatevectorSampler()
job = sampler.run([qc_measured], shots=n)
result = job.result()
counts = result[0].data["meas"].get_counts()

count_0 = counts.get("0", 0)
count_1 = counts.get("1", 0)
p_0 = count_0 / n
p_1 = count_1 / n
print(f"\nActual counts:\n")
print(f"\t\t0 = {count_0} ({round(p_0*100, ndigits=1)}%)")
print(f"\t\t1 = {count_1} ({round(p_1*100, ndigits=1)}%)")

qc.draw(output="mpl")

sphere = Bloch()
x = 2*np.real(a * b)
y = 2*np.imag(a * b)
z = np.abs(a)**2 - np.abs(b)**2
sphere.add_vectors([x, y, z])
sphere.show()

plt.show()