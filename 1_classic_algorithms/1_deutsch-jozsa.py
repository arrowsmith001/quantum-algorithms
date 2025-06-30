import random
from typing import Literal
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import MCXGate, IGate, HGate, XGate
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_qsphere
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

### DEUTSCH-JOZSA ALGORITHM ###

# This is exactly the same as Deutsch's algorithm, just generalised to an n-dimensional input space.
#
# The function "f" now takes n inputs. It still only returns either 0 or 1 i.e. f:{0,1}^(2^n) -> {0,1}
#             _____
# |x_0⟩   -> |     | -> |x_0⟩
# |x_1⟩   -> |     | -> |x_1⟩
# |x_2⟩   -> |     | -> |x_2⟩
# |x_3⟩   -> | U_f | -> |x_3⟩
# ...     -> |     | -> ...
# |x_n-1⟩ -> |     | -> |x_n-1⟩
# |y⟩     -> |_____| -> |(y + f(x_0,...,x_n-1)) % 2⟩
#
# As before, it is constant if it returns the same value for all inputs. It is balanced if it returns 0 for half its inputs and 1 for the other half.

# It is a bit trickier to construct U_f now for arbitrary n, so I provide some parameters below to generate U_f...


### VARIABLES (EDITABLE) ############################################################################################################

n = 4  # Number of inputs for f (setting this to 1 will reduce it to the original Deutsch algorithm!)

# U_f generation:
is_constant = False  # True: f will be constant, False: f will be balanced
value_if_constant : Literal[0, 1] = 0  # Change between 0 or 1 if you want to test each constant case
seed_if_balanced = 1234  # Can be any number, string, bytes, etc. This is just to seed our balanced function generator to make results reproducable.

enable_sense_check = True
enable_visualisations = True

#####################################################################################################################################


# All n-bit string representations
def all_nbit_strings(n):
    return [np.binary_repr(i).zfill(n) for i in range(2**n)]


# This randomly generates an array that is half of all possible nbit strings.
# This will be our inputs that a balanced function will return '1' for.
# The other half by omission will be the '0' inputs.
def random_half_of_nbit_strings(seed):
    random.seed(seed)
    # Generate array of all possible binary strings
    all = all_nbit_strings(n)
    # Randomly split the array into 2 equal halves
    random.shuffle(all)
    return all[: len(all) // 2]


# The idea in constructing the oracle is to use multi-controlled X gates, which will flip the target bit (the output) to "1" only if all input bits are 1.
# Since each controlled gate will only trigger once at most, the output will be 1 if the input is in our set, otherwise it will be 0.
# You can visualise this subcircuit if you have enable_visualisations=True and is_constant=False
# Note: if I was being efficient I could remove bit flip pairs, and even choose an order that would maximise these cancellable pairs, but I'll leave this as is for clarity.
def balanced_oracle(n, ones, barriers=False):
    subcircuit = QuantumCircuit(n + 1)
    qubits = subcircuit.qubits
    for x in ones:
        # We flip only the 0s, so the state will be "111..." if a string matches.
        # e.g. if "001" is in our set, we should bit flip according to the pattern "110", which will produce "111"
        for i in range(0, n):
            if x[n - i - 1] == "0": # Indexing reversed because the rightmost bit is least significant
                subcircuit.x(qubits[i])

        # Multi-controlled X, which will flip the target bit only if the state is "111..." when applied
        subcircuit.append(MCXGate(n), qubits[0:n+1])

        # We then "reset" by flipping the bits back so we can do the same as above for the next element of the set, and so on.
        for i in range(0, n):
            if x[n - i - 1] == "0":
                subcircuit.x(qubits[i])
                
        if(barriers):
            subcircuit.barrier() # for visuals
    
    return subcircuit

# The constant oracle is far simpler. If 0, then it is the identity - no action is necessary. If 1, then we directly bit flip "y".
def constant_oracle(n, value):
    subcircuit = QuantumCircuit(n + 1)
    if value != 0:
        subcircuit.x([n])
    return subcircuit

ones_array = random_half_of_nbit_strings(seed_if_balanced)
U_f : QuantumCircuit = constant_oracle(n, value_if_constant) if is_constant else balanced_oracle(n, ones_array)

if not is_constant:
    print(f'\nThese are the input values for which f should return 1:\n\n\t', ones_array)

# Let's take a moment to convince ourselves that this oracle encodes f
if enable_sense_check:
    print('\nSense check...\n')
    for s in all_nbit_strings(n):
        _qc = QuantumCircuit(n+1)
        _qc.append(U_f, _qc.qubits[0:n+1])
        
        state = Statevector.from_label('0' + s) # The extra '0' is for the target bit "y"
        state = state.evolve(_qc) # Evolve initial state
        prb = state.probabilities_dict(decimals=0)
        result_bit = next(iter(prb.keys()))[0]
        
        expected_value = value_if_constant if is_constant else (1 if s in ones_array else 0)
        
        print(f"\t{s}: -> {result_bit}, expected: {expected_value}")
        assert(int(result_bit) == expected_value)

# Visualise the balanced oracle
if enable_visualisations and not is_constant:
    Uf_circuit = balanced_oracle(n, ones_array, True)
    Uf_circuit.draw(output='mpl')
    plt.suptitle('Circuit for the balanced Deutsch-Jozsa Oracle', fontsize=16)
    plt.title(f'Encodes (from Left to Right): {ones_array}', fontsize=10, y=1)
    plt.show()


# THE ALGORITHM

def visualise(qc: QuantumCircuit, title=""):
    if not enable_visualisations:
        return
    sv = Statevector.from_instruction(qc)
    fig = plot_state_qsphere(sv)
    fig.suptitle(title)
    plt.show()
    
qc = QuantumCircuit(QuantumRegister(n+1), ClassicalRegister(n))
visualise(qc, f"Initial state is |{'0'*(n+1)}⟩")

qc.x(qc.qubits[n])
visualise(qc, f"Output bit is flipped, changing the state to |1{'0'*n}⟩")

qc.barrier()
qc.h(range(n+1))
visualise(qc, f"Places system in an equal superposition of all {2**(n+1)} base states")

# TODO: Explain this better
qc.append(U_f.to_gate(label='U_f'), range(n+1))
visualise(qc, "Applies phase oracle, changing some phases if f is balanced.")

qc.h(range(n))
visualise(qc, f"If constant, Hadamard gate will collapse to a superposition of 2 states,\nboth of which are |{'0'*(n)}⟩ in the measured bits.\nIf balanced, the state will be in a superposition of states\n all containing '1' within the first (rightmost) n bits.")

qc.measure(range(n), range(n))

simulator = AerSimulator()
compiled = transpile(qc, simulator)
result = simulator.run(compiled).result()
counts = result.get_counts()

# The simulator does 1024 runs by default. This time we're measuring the first n qubits only (the input qubits):
#
#       - If ALL of the first n qubits are 0 (i.e. |000...0⟩), then f is constant
#       - If ANY of the first n qubits are 1, then f is balanced

print('\nCounts: ', counts)

# get keys of counts
keys = list(counts.keys())
if len(keys) == 1 and keys[0] == '0'*n:
    print('\nThe function is constant')
else:
    print('\nThe function is balanced')

qc.draw(output='mpl')
plt.show()

# Again, this algorithm only consults the oracle once, whereas this time a classical computer would need to evaluate f ((2^n)/2)+1 times in the worst case (1 + half of the number of all possible inputs), which is exponential.
