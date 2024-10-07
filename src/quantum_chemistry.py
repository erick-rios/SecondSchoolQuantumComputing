# ICTP second quantum computing school 

import pennylane as qml
from pennylane import numpy as pnp 

# import your favourite libraries 

symbols = ["H","H"]

# choose the position of the nuclei of the molecule

coordinates = pnp.array([[0.0,0.0,-0.6614], [0.0,0.0,0.6614]])

# Build the molecular hamiltonian

# Build the molecular Hamiltonian (in second-quantized form and mapped to qubits via Wordan-Wigner transformation)
molecule = qml.qchem.Molecule(symbols, coordinates)
H, qubits = qml.qchem.molecular_hamiltonian(molecule)

print("Number of qubits: {:}".format(qubits))
print("Qubit Hamiltonian: ")
print(H)

# Print the len of the molecule
len(H)

# The variational Quantum Eigensolver (VQE) algorithm

# Create the Hartree-Fock initial state for the molecule

electrons = 2
orbitals = qubits

hf = qml.qchem.hf_state(electrons, orbitals)
print(hf)

# Thi should print [ 1 1 0 0 ]

# Define variational curcuit that preapares anstaze

def circuit_H2_VQE( param, wires)-> None:
    # Prepares HF basis state in the circuit to  initialize the circuit
    
    qml.BasisState(hf, wires = wires)
    
    # Parametrize coupling between HF state
    
    qml.DoubleExcitation(param, wires = [0,1,2,3])
    
    
# define device to use

dev = qml.device("default.qubit", wires = qubits)

@qml.qnode(dev)
def cost_fn(param) -> None:
    
    circuit_H2_VQE(param, wires=range(qubits))
    
    
    return qml.expval(H)

# Draw circuit

qml.draw_mpl(cost_fn)(0-2)

# Choose an optimier

opt = qml.GradientDescentOptimizer(stepsize = 0.4)

# Choose an initial parametre for the variation cicuit (angle of Givens Rotation)
theta = pnp.array(2.0, requires_grad = True)

cost_fn(theta)

# store the vaues of the cost function

energy = [cost_fn(theta)]

# store the values of the circuit parameter

angle = [theta]

max_iterations = 40
conv_tol = 1e-06



# Repetition of optimization until convergence

for n in range(max_iterations):
    theta, prev_energy = opt.step_and_cost(cost_fn, theta)
    
    energy.append(cost_fn(theta))
    angle.append(theta)

    conv = pnp.abs(energy[-1] - prev_energy)
    
    if n%2==0:
        print(f"Nrep = {n}, Energy = {energy}")
    
    if conv <= conv_tol:
        break
    
# print the results for the las state

