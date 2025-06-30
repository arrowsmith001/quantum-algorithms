from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city, plot_state_qsphere, plot_bloch_multivector
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from qutip import Bloch

### DEUTSCH's ALGORITHM ###

# This is the most basic problem for which a quantum computer demonstrably outperforms a classical computer.

# The problem is: given a function f:{0,1} -> {0,1}, determine whether f is:
#
#   - constant (always returns the same value for all inputs), or 
#   - balanced (returns 0 for half of its inputs and 1 for the other half of the inputs).
#
# Given the only possible inputs are 0 and 1, there are 4 possibilities for f:
#
# f_0(x) => 0 for all x           = constant 0 function           => constant
# f_1(x) => 0 if x=0, 1 if x=1    = identity function             => balanced         
# f_2(x) => 1 if x=0, 0 if x=1    = "not" function aka bit-flip   => balanced
# f_3(x) = 1 for all x            = constant 1 function           => constant

# For the quantum implementations, we refer to the standard oracle construction:
#         _____
# |x⟩ -> |     | -> |x⟩
#        | U_f |
# |y⟩ -> |_____| -> |(y + f(x)) % 2⟩
#
# The top x is a control bit - it goes unchanged. The bottom y typically enters as |0⟩, so it will be transformed into |f(x)⟩, however it must nonetheless be included so that the unitary is reversible.
#
# With a little work, given the form of the oracle and the functions f_n, you can convince yourself that U_fn obeys the following table:
#
# Input ->  |00⟩    |01⟩    |10⟩    |11⟩
# U_f0:     |00⟩    |01⟩    |10⟩    |11⟩    = identity (no change)
# U_f1:     |00⟩    |01⟩    |11⟩    |10⟩    = controlled not (i.e. if left bit is 0, do nothing, if left bit is 1, flip right bit)
# U_f2:     |01⟩    |00⟩    |10⟩    |11⟩    = controlled not, then flip right bit (compare to U_f1)
# U_f3:     |01⟩    |00⟩    |11⟩    |10⟩    = flip right bit
#
# Note: bit flipping is also known as the "not" operation
# Note: the right bit is the least significant bit, so in the below circuits indexing is done right-to-left (the right bit is the 1st bit, and the left bit is the 2nd bit).

def U_f0():
    subcircuit = QuantumCircuit(QuantumRegister(2))
    subcircuit.id(0)
    subcircuit.id(1)
    return subcircuit.to_gate(label='U_f0')

def U_f1():
    subcircuit = QuantumCircuit(QuantumRegister(2))
    subcircuit.cx(0, 1)
    return subcircuit.to_gate(label='U_f1')

def U_f2():
    subcircuit = QuantumCircuit(QuantumRegister(2))
    subcircuit.cx(0, 1)
    subcircuit.x(1)
    return subcircuit.to_gate(label='U_f2')

def U_f3():
    subcircuit = QuantumCircuit(QuantumRegister(2))
    subcircuit.x(1, "U_f3")
    return subcircuit.to_gate(label='U_f3')

### VARIABLES (EDITABLE) ###

# The 2-qubit unitary representing the chosen function f. Change to U_fn to change the function.
U_f = U_f1

enable_visualisations = True

### THE ALGORITHM - at each step I will also visualise the effect of each transformation on the Bloch sphere.

def visualise(qc: QuantumCircuit, title=""):
    if not enable_visualisations:
        return
    sv = Statevector.from_instruction(qc)
    fig = plot_state_qsphere(sv)
    fig.suptitle(title)
    plt.show()

# We only intend to measure the first qubit, so only 1 classical register is required
qc = QuantumCircuit(QuantumRegister(2), ClassicalRegister(1))
visualise(qc, "Initial state is |00⟩")

qc.x(1)
visualise(qc, "Left bit is flipped, changing the state to |10⟩")

qc.barrier()
qc.h(0)
qc.h(1)
visualise(qc, "Places system in a superposition of all 4 base vector states.\n|00⟩ and |01⟩ are in phase, and |10⟩ and |11⟩ are out of phase.\nThe colours relate to the phase.\nVectors that are out-of-phase correspond to negative probability amplitudes.")

# TODO: Explain this better
qc.append(U_f(), [0, 1])
visualise(qc, "U_f is also known as a \"phase oracle\".\nNotice that if f is constant there is no change in phase.\nIf f is balanced then the states associated with q0=0 flip phase.\nThis is known as \"phase kickback\").")

qc.h(0)
visualise(qc, "Hadamard gate combines the amplitudes to just two states.\nThey are in a superposition, but we're only measuring the 1st qubit,\nwhich is either 0 or 1 regardless of the 2nd qubit.")

q0 = qc.qubits[0]
qc.measure(q0, 0)

simulator = AerSimulator()
compiled = transpile(qc, simulator)
result = simulator.run(compiled).result()
counts = result.get_counts()

# The simulator does 1024 runs by default. Depending on the function f given, they should either all be 0 or all be 1.
#
#       - 0 => function is constant (f_0 or f_3)
#       - 1 => function is balanced (f_1 or f_2) 

print('Counts: ', counts)
result = list(counts.keys())[0]
print(f'Result: the function is {'balanced' if result == '1' else 'constant'}.')

qc.draw(output="mpl")
plt.show()

# This algorithm outperforms a classical computer since the oracle is only consulted once, whereas a classical computer would need to evaluate f twice (once for each input) to reach the same conclusion. However, Deutsch's algorithm does not tell us exactly which function the oracle represents - only that it is either constant or balanced.