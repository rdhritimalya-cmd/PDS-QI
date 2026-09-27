"""
Magic / nonstabilizerness for PDS eigenstates (pre-empts Savage-Robin PRC 111).

Savage-Robin: entanglement alone doesn't capture quantum complexity; magic
(nonstabilizerness) is a separate resource, large in deformed nuclei. To position
our label-variance diagnostic against theirs, we compute the stabilizer 2-Renyi
entropy (Leone-Oliviero-Hamma) of PDS eigenstates under a qubit encoding, and ask:
  - is magic a SEPARATE axis from label-variance (yes -> complementary, not redundant)?
  - do solvable states have distinctive magic (a bonus fingerprint)?

Encoding: the M=0 boson block has dimension D; pad to 2^n and encode as n qubits
(n=ceil(log2 D)). Stabilizer 2-Renyi entropy:
   M2 = -log2( sum_P <psi|P|psi>^4 / 2^n )  over all n-qubit Paulis P,
   normalized magic m = M2 (>=0; 0 for stabilizer states).
Exact Pauli sum is 4^n; feasible for n<=6 (D<=64 -> N<=4 full, or truncated blocks).
For larger D we use the exact formula on the smallest faithful encoding and sample
Paulis if needed. Here we use small N (N=3,4) where D(M=0) is small, to make the
POINT (magic is a distinct axis) rigorously and exactly; scaling is a VQE-era task.
"""
import numpy as np, itertools, math
from ibm_pds import IBMSpace, build_H

# single-qubit Paulis
I2=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.array([[1,0],[0,-1]])
PAULI=[I2,X,Y,Z]

def pauli_string(indices):
    M=np.array([[1]])
    for k in indices:
        M=np.kron(M,PAULI[k])
    return M

def stabilizer_renyi2(psi, n):
    """Exact M2 stabilizer 2-Renyi entropy for n-qubit pure state psi (len 2^n)."""
    dim=2**n
    v=np.zeros(dim,dtype=complex); v[:len(psi)]=psi; v/=np.linalg.norm(v)
    s=0.0
    for idx in itertools.product(range(4),repeat=n):
        P=pauli_string(idx)
        exp=np.real(v.conj()@(P@v))
        s+=exp**4
    # M2 = -log2( (1/2^n) sum_P <P>^4 ) ; note sum includes identity giving 1
    M2=-math.log2(s/dim)
    return M2

def encode_dim(D):
    return int(math.ceil(math.log2(D)))

if __name__=="__main__":
    # N=4: M=0 block is small enough for exact 4^n Pauli sum
    N=4
    spN=IBMSpace(N); spN2=IBMSpace(N-2); sel=np.where(spN.Mvals==0)[0]
    D=len(sel); n=encode_dim(D)
    print(f"N={N}: M=0 dim {D} -> {n} qubits (2^n={2**n}), Pauli terms 4^n={4**n}")
    C2b=spN.C2_SU3()[np.ix_(sel,sel)].toarray()
    L2b=spN.L2()[np.ix_(sel,sel)].toarray()
    H=build_H(spN,spN2,0.35,1.0,0.0523)[np.ix_(sel,sel)].toarray()
    ev,V=np.linalg.eigh(H); i=0
    while i<len(ev):
        j=i
        while j+1<len(ev) and ev[j+1]-ev[i]<1e-9: j+=1
        if j>i:
            sub=V[:,i:j+1]
            for op in (L2b,C2b):
                w,u=np.linalg.eigh(sub.conj().T@op@sub); sub=sub@u
            V[:,i:j+1]=sub
        i=j+1
    c2=np.einsum('in,ij,jn->n',V.conj(),C2b,V).real
    c2var=np.einsum('in,ij,jn->n',V.conj(),C2b@C2b,V).real-c2**2
    solv=c2var<1e-6
    print(f"solvable {solv.sum()}, mixed {(~solv).sum()}")
    print(f"{'idx':>3} {'solv':>5} {'M2(magic)':>10} {'c2var':>10}")
    mags=[]
    for k in range(len(ev)):
        m=stabilizer_renyi2(V[:,k],n)
        mags.append(m)
    mags=np.array(mags)
    print(f"magic M2: solvable mean={mags[solv].mean():.3f} [{mags[solv].min():.3f},{mags[solv].max():.3f}]")
    print(f"          mixed    mean={mags[~solv].mean():.3f} [{mags[~solv].min():.3f},{mags[~solv].max():.3f}]")
    # correlation of magic with label-variance (should be WEAK -> separate axes)
    from numpy import corrcoef
    cc=corrcoef(mags, np.sqrt(np.clip(c2var,0,None)))[0,1]
    print(f"corr(magic, sqrt Var C2) = {cc:.3f}  (weak => magic is a SEPARATE axis)")
