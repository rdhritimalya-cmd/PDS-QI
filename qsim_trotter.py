"""
Trotterized time evolution e^{-iHt} for the encoded sd-IBM Hamiltonian.
Demonstrates first- and second-order Trotter error vs number of steps,
benchmarked against exact matrix exponentiation. This is the 'tool' half:
time evolution appears ONLY here, not in any physics result.
"""
import numpy as np, math
from scipy.linalg import expm
import ibm_pds as m
from qiskit.quantum_info import SparsePauliOp, Operator

def embed(mat):
    d=mat.shape[0]; n=max(1,math.ceil(math.log2(d))); D=2**n
    big=np.zeros((D,D),complex); big[:d,:d]=mat
    sh=np.linalg.eigvalsh(mat).real.max()+10.0
    for k in range(d,D): big[k,k]=sh
    return big,n,d,sh

def pauli_H(N):
    spN=m.IBMSpace(N); spN2=m.IBMSpace(N-2)
    H=m.build_H(spN,spN2,1.0,1.0,0.0).toarray().real
    big,n,d,sh=embed(H)
    op=SparsePauliOp.from_operator(Operator(big))
    return op,big,n,d

def trotter1(terms, coeffs, t, steps):
    """First-order Trotter: prod_k exp(-i c_k P_k t/steps), repeated steps times."""
    dt=t/steps
    U=np.eye(terms[0].shape[0],dtype=complex)
    layer=np.eye(terms[0].shape[0],dtype=complex)
    for P,c in zip(terms,coeffs):
        layer=expm(-1j*c*P*dt)@layer
    Us=np.linalg.matrix_power(layer,steps)
    return Us

def trotter2(terms,coeffs,t,steps):
    """Second-order (Strang) Trotter."""
    dt=t/steps
    half=np.eye(terms[0].shape[0],dtype=complex)
    for P,c in zip(terms,coeffs): half=expm(-1j*c*P*dt/2)@half
    for P,c in zip(reversed(terms),reversed(coeffs)): half=expm(-1j*c*P*dt/2)@half
    return np.linalg.matrix_power(half,steps)

if __name__=="__main__":
    N=2
    op,big,n,d=pauli_H(N)
    terms=[Operator(SparsePauliOp(pl)).data for pl in op.paulis]
    coeffs=np.real(op.coeffs)
    t=1.0
    Uexact=expm(-1j*big*t)
    psi0=np.zeros(big.shape[0],complex); psi0[0]=1.0  # a fixed reference in the qubit register
    print(f"N={N}, qubits={n}, #Pauli terms={len(coeffs)}, t={t}")
    print(" steps |  1st-order err   2nd-order err   (state infidelity)")
    for steps in [1,2,4,8,16,32]:
        U1=trotter1(terms,coeffs,t,steps)
        U2=trotter2(terms,coeffs,t,steps)
        f1=1-abs(np.vdot(Uexact@psi0, U1@psi0))**2
        f2=1-abs(np.vdot(Uexact@psi0, U2@psi0))**2
        print(f"  {steps:4d} |   {f1:.3e}      {f2:.3e}")
