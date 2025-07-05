from tokenize import String
from typing import Literal
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from typing import Literal
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_qsphere

### SUPERDENSE CODING ###

# Alice and Bob each possess a qubit. These qubits are maximally entangled (i.e. are in the Bell state), and Alice and Bob (and their qubits) are far apart.

# Ordinarily, Alice could only transfer 1 bit of information via a single bit or a qubit. In the case of a qubit, she would prepare the state to either definitely be 0 or 1.

# However, because of the entanglement, Alice can send Bob 2 bits of information by sending him just 1 qubit. This is "superdense coding".


### VARIABLES (EDITABLE) ############################################################################################################

# What bits should Alice send to Bob?
target : Literal['00', '01', '10', '11'] = '01'

enable_visualisations = True

#####################################################################################################################################

# THE ALGORITHM

def visualise(qc: QuantumCircuit, title=""):
    if not enable_visualisations:
        return
    sv = Statevector.from_instruction(qc)
    fig = plot_state_qsphere(sv)
    fig.suptitle(title)
    plt.show()
    
qc = QuantumCircuit(QuantumRegister(2))

q_alice = qc.qubits[0]
q_bob = qc.qubits[1]

# Standard way to prepare qubits in a Bell state
qc.h(q_alice)
qc.cx(q_alice, q_bob)
qc.barrier()

# Confirm that state is Bell state (this would not be possible on actual quantum hardware)
current_state = Statevector.from_instruction(qc)
bell_state = np.array([1/sqrt(2), 0, 0, 1/sqrt(2)])
assert(np.allclose(current_state, bell_state, atol=10e-16))

visualise(qc, f"Bell state")

# Now Alice and Bob travel far apart.

# Alice decides what 2 bits of information she wants to send to Bob. Based on that she applies a gate to her qubit.

first_bit = int(target[0])
second_bit = int(target[1])

if(first_bit == 0):
    if(second_bit == 0):    # 00
        qc.id(q_alice)
    elif(second_bit == 1):  # 01
        qc.z(q_alice)
elif(first_bit == 1):
    if(second_bit == 0):    # 10
        qc.x(q_alice)
    elif(second_bit == 1):  # 11
        qc.x(q_alice)
        qc.z(q_alice)
        
qc.barrier()
visualise(qc, f"The state Alice has prepared")

# Alice sends her qubit to Bob.

# Bob applies the following gates to both qubits now:

qc.cx(q_alice, q_bob)
qc.h(q_alice)

visualise(qc, f"State {first_bit}{second_bit}")

# Bob now measures the two qubits.

qc_measured = qc.measure_all(inplace=False)
assert qc_measured is not None

simulator = AerSimulator()
compiled = transpile(qc_measured, simulator)
result = simulator.run(compiled).result()
counts = result.get_counts()

print('Counts: ', counts)

qc_measured.draw(output="mpl")
plt.show()