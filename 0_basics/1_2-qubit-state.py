from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

### VARIABLES (EDITABLE) ###

# Target state, which is a 4D array of arbitrary complex numbers
# e.g. Bell state (maximum entanglement): 1/sqrt(2)∣00⟩ + 1/sqrt(2)∣11⟩
s = np.array([1 / sqrt(2), 0, 0, 1 / sqrt(2)])

# Number of trials when running the simulator. The more trials, the closer the probabilities will be to the true probabilities.
n = 10000


### SETTING AN ARBITRARY 2-QUBIT STATE ###

# Normalise
s = s / np.linalg.norm(s)

# Probability amplitudes
a0, a1, a2, a3 = s

# Our goal is to create a circuit that would produce this state from 00.
# (Note: This can be done with qiskit easily with qc.initialize(Statevector(target_state)), but that's no fun)
#
# The way this can be done is computing the matrices that would take us from this state *to* the 00 state.
# Then we can invert those matrices to get the target state starting from 00, since all quantum circuit operations are reversible.
# The steps to compute and apply those matrices are outlined here: https://www.youtube.com/watch?v=LIdYSs-rE-o

def unitary(x, y):
    norm = np.linalg.norm([x, y])
    m = np.matrix([[x, y], [-np.conj(y), np.conj(x)]], dtype=complex)
    return (1.0 / norm) * m

# Part 1: s0 -> s1
A1 = np.array([a0, a1])
A2 = np.array([a2, a3])

A1Norm = np.linalg.norm(A1)
A2Norm = np.linalg.norm(A2)
A1dotA2 = np.dot(np.conj(A1), A2)

if A1dotA2 == 0:
    k = A2Norm / A1Norm
else:
    k1 = -A2Norm / A1Norm
    k2 = A1dotA2 / np.abs(A1dotA2)
    k = k1 * k2

W1 = unitary(a3 - k * a1, np.conj(a2 - k * a0)).T

# Controlled-Z matrix
CZ = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)

W1xI = np.kron(
    np.identity(2),
    W1,
)

s1 = CZ @ W1xI @ s

n0, n1, n2, n3 = [s1[0, 0], s1[0, 1], s1[0, 2], s1[0, 3]]
s1 = np.array([n0, n1, n2, n3]) # Quirk of complex numbers with numpy - must redefine the array to avoid errors

# Part 2: s1 -> s2
W2 = unitary(np.conj(n1), np.conj(n3)).T

W2xI = np.kron(W2.T, np.identity(2))

s2 = W2xI @ s1

# Part 3: s2 -> 00
g0, g1, g2, g3 = [s2[0, 0], s2[0, 1], s2[0, 2], s2[0, 3]]
s2 = np.array([g0, g1, g2, g3])

W3 = unitary(np.conj(g0), -np.conj(g1)).T

IxW3 = np.kron(np.identity(2), W3)

s3 = IxW3 @ s2

s3 = np.array([s3[0, 0], s3[0, 1], s3[0, 2], s3[0, 3]])

# Confirm that s3 is indeed the zero state
test_array = s3 - np.array([1, 0, 0, 0])
assert np.allclose(test_array, np.zeros((1, 4)), atol=1e-10)

# Now we have the generalised steps from our target state -> 00, as follows:
#
# 00 = IxW3 @ W2xI @ CZ @ W1xI @ s0
#
# Simply need to invert the actual matrices we've made to go the other way:
#
# s0 = W1xI^-1 @ CZ^1 @ (W2xI)^-1 @ (IxW3)^-1 @ 00

IxW3_inv = np.linalg.inv(IxW3)
W2xI_inv = np.linalg.inv(W2xI)
CZ_inv = np.conj(np.linalg.inv(CZ))
W1xI_inv = np.linalg.inv(W1xI)

# This is the final unitary matrix that is required to get straight to the target state *from* 00
U = W1xI_inv @ CZ_inv @ W2xI_inv @ IxW3_inv

# Let us use this and measure the outcomes when compared with the predicted outcomes
# For example, the Bell State should be evenly split between 00 and 11
qc = QuantumCircuit(QuantumRegister(2), ClassicalRegister(2))

qc.unitary(U, [0, 1], "U")
qc.measure_all(inplace=False)

qc_measured = qc.measure_all(inplace=False)
assert qc_measured is not None

# Theoretical results
p_00 = abs(np.square(a0))
exp_00 = abs(p_00) * n
p_01 = abs(np.square(a1))
exp_01 = abs(p_01) * n
p_10 = abs(np.square(a2))
exp_10 = abs(p_10) * n
p_11 = abs(np.square(a3))
exp_11 = abs(p_11) * n

print(f"\nExpected counts:\n")
print(f"\t\t00 = {round(np.real(exp_00))}\t({round(p_00*100, ndigits=1)}%)")
print(f"\t\t01 = {round(np.real(exp_01))}\t({round(p_01*100, ndigits=1)}%)")
print(f"\t\t10 = {round(np.real(exp_10))}\t({round(p_10*100, ndigits=1)}%)")
print(f"\t\t11 = {round(np.real(exp_11))}\t({round(p_11*100, ndigits=1)}%)")

# Actual results
sampler = StatevectorSampler()
job = sampler.run([qc_measured], shots=n)
result = job.result()
counts = result[0].data["meas"].get_counts()

count_00 = counts.get("00", 0)
count_01 = counts.get("01", 0)
count_10 = counts.get("10", 0)
count_11 = counts.get("11", 0)
p_00 = count_00 / n
p_01 = count_01 / n
p_10 = count_10 / n
p_11 = count_11 / n

print(f"\nActual counts:\n")
print(f"\t\t00 = {count_00}\t({round(p_00*100, ndigits=1)}%)")
print(f"\t\t01 = {count_01}\t({round(p_01*100, ndigits=1)}%)")
print(f"\t\t10 = {count_10}\t({round(p_10*100, ndigits=1)}%)")
print(f"\t\t11 = {count_11}\t({round(p_11*100, ndigits=1)}%)")

qc.draw(output="mpl")
plt.show()