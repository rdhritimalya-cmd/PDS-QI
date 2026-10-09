"""
Boson-to-qubit encodings of the sd-IBM, built as SparsePauliOps so we never
materialize a 2^(6q) matrix. Three encodings:
  - 'binary' : each of 6 modes stored in q=ceil(log2(N+1)) qubits (binary occupation)
  - 'unary'  : each of 6 modes stored in q=(N+1) qubits (one-hot occupation)
Per-mode a, adag, n are built as small (2^q x 2^q) matrices and decomposed to
Pauli strings; full-register operators are tensor products / sums / products of these.
Physics check: project H onto the physical Fock subspace and compare eigenvalues
to the exact Fock-space spectrum.
(The 'dense' symmetry-reduced encoding is handled separately via the existing code.)
"""
import numpy as np, math, itertools
from functools import reduce
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli
from ibm_pds import IBMSpace, cg, D_MS, build_H, f2
import critical as cr

MODES = ['s','d-2','d-1','d0','d1','d2']   # 6 modes; d index m = -2..2
def mode_m(name):  # magnetic quantum number for d modes, None for s
    return None if name=='s' else int(name[1:])

def qubits_per_mode(N, enc):
    if enc=='binary': return max(1, math.ceil(math.log2(N+1)))
    if enc=='unary':  return N+1
    raise ValueError

def enc_index(k, q, enc):
    """computational-basis integer for occupation k."""
    if enc=='binary': return k                     # binary number
    if enc=='unary':  return (1<<k)                # one-hot: bit k set
    raise ValueError

def local_ops(N, q, enc):
    """small 2^q x 2^q matrices for a, adag, n on one mode (occupations 0..N)."""
    D=2**q
    a=np.zeros((D,D)); n=np.zeros((D,D))
    for k in range(0,N+1):
        ik=enc_index(k,q,enc); n[ik,ik]=k
        if k>=1:
            ikm=enc_index(k-1,q,enc)
            a[ikm,ik]=math.sqrt(k)                 # a|k> = sqrt(k)|k-1>
    adag=a.T.copy()
    return a,adag,n

def to_spo(mat):
    return SparsePauliOp.from_operator(Operator(mat)).simplify()

def pad(op_spo, pos, q, nmodes=6):
    """tensor a q-qubit SparsePauliOp into full register at mode position `pos`."""
    I_before = SparsePauliOp('I'*(q*pos)) if pos>0 else None
    I_after  = SparsePauliOp('I'*(q*(nmodes-1-pos))) if pos<nmodes-1 else None
    # mode 0 must sit on the LOW qubits (consistent with fock_to_bits bit-packing):
    # layout high->low = I_after(high modes) | op(mode pos) | I_before(low modes)
    out=op_spo
    if I_before is not None: out=out.tensor(I_before)   # op ⊗ I_before  (low modes on low qubits)
    if I_after  is not None: out=I_after.tensor(out)    # I_after ⊗ out  (high modes on high qubits)
    return out.simplify()

class Encoded:
    def __init__(self, N, enc):
        self.N=N; self.enc=enc; self.q=qubits_per_mode(N,enc); self.nq=6*self.q
        a,adag,n=local_ops(N,self.q,enc)
        self.a_spo=to_spo(a); self.adag_spo=to_spo(adag); self.n_spo=to_spo(n)
        self.A   =[pad(self.a_spo,p,self.q) for p in range(6)]     # a_mode
        self.Ad  =[pad(self.adag_spo,p,self.q) for p in range(6)]  # adag_mode
        self.Nm  =[pad(self.n_spo,p,self.q) for p in range(6)]     # n_mode
        self.id  =SparsePauliOp('I'*self.nq)
    def midx(self,name): return MODES.index(name)
    def nd(self):
        return sum((self.Nm[self.midx(nm)] for nm in MODES[1:]), SparsePauliOp('I'*self.nq)*0).simplify()
    def adag_of(self,name): return self.Ad[self.midx(name)]
    def a_of(self,name):    return self.A[self.midx(name)]

def dmode(m): return f'd{m}' if m!=0 else 'd0'

def build_pair_P0(E):
    """P0^dag = sum_m (-1)^m d_m^dag d_{-m}^dag - 2 (s^dag)^2 . Returns SparsePauliOp."""
    nq=E.nq; out=SparsePauliOp('I'*nq)*0
    for m in D_MS:
        term=E.adag_of(dmode(m)).dot(E.adag_of(dmode(-m)))
        out=out + ((-1)**m)*term
    out=out - 2.0*E.adag_of('s').dot(E.adag_of('s'))
    return out.simplify()

def build_pair_P2(E, mu, beta0=None):
    """P2mu^dag. If beta0 given, critical form; else PRL-77 SU(3) form (coef 2, sqrt7)."""
    nq=E.nq; out=SparsePauliOp('I'*nq)*0
    c_sd = (beta0 if beta0 is not None else 2.0)
    c_dd = (math.sqrt(3.5) if beta0 is not None else math.sqrt(7.0))
    out=out + c_sd*E.adag_of('s').dot(E.adag_of(dmode(mu)))
    for m1 in D_MS:
        m2=mu-m1
        if -2<=m2<=2:
            c=cg(2,m1,2,m2,2,mu)
            if c!=0.0:
                out=out + c_dd*c*E.adag_of(dmode(m1)).dot(E.adag_of(dmode(m2)))
    return out.simplify()

def build_dd_tensor(E, k, q):
    """(d^dag dtil)^(k)_q = sum_m1 cg(2,m1,2,m2,k,q)*(-1)^m2 adag_{d,m1} a_{d,-m2}, m2=q-m1."""
    nq=E.nq
    out=SparsePauliOp('I'*nq)*0
    for m1 in D_MS:
        m2=q-m1
        if -2<=m2<=2:
            c=cg(2,m1,2,m2,k,q)
            if c!=0.0:
                out=out + c*((-1)**m2)*E.adag_of(dmode(m1)).dot(E.a_of(dmode(-m2)))
    return out.simplify()

def build_L2(E):
    nq=E.nq; out=SparsePauliOp('I'*nq)*0
    for q in (-1,0,1):
        Lq=math.sqrt(10.0)*build_dd_tensor(E,1,q)
        Lmq=math.sqrt(10.0)*build_dd_tensor(E,1,-q)
        out=out + ((-1)**q)*Lq.dot(Lmq)
    return out.simplify()

def build_H_encoded(E, h0, h2, C, beta0=None):
    """Stable SU(3) H = h0 P0^dP0 + h2 sum_mu P2^d P2 + C L^2, or critical (beta0)."""
    nq=E.nq
    H=SparsePauliOp('I'*nq)*0
    if beta0 is None:
        P0=build_pair_P0(E)
        H=H + h0*P0.dot(P0.adjoint())
        for mu in range(-2,3):
            P2=build_pair_P2(E,mu)
            H=H + h2*P2.dot(P2.adjoint())
        if C!=0.0:
            H=H + C*build_L2(E)
    else:
        for mu in range(-2,3):
            P2=build_pair_P2(E,mu,beta0=beta0)
            H=H + h2*P2.dot(P2.adjoint())
    return H.simplify()

# ---- physical-subspace verification (no full matrix) ----
def fock_to_bits(state, q, enc):
    """Fock occupation tuple (ns,n d-2..d2) -> computational basis integer on 6q qubits."""
    idx=0
    for p,occ in enumerate(state):
        local=enc_index(occ,q,enc)
        idx |= (local << (q*p))
    return idx

def pauli_on_basis(label, bitint, nq):
    """Apply a Pauli string (label, qubit 0 = rightmost char) to |bitint>; return (coef, newbit)."""
    coef=1+0j; newbit=bitint
    for qi,ch in enumerate(reversed(label)):
        bit=(bitint>>qi)&1
        if ch=='I': continue
        if ch=='Z':
            if bit: coef*=-1
        elif ch=='X':
            newbit^=(1<<qi)
        elif ch=='Y':
            newbit^=(1<<qi)
            coef*= (1j if bit==0 else -1j)
    return coef,newbit

def verify_spectrum(E, Hspo, h0,h2,C,beta0=None, tol=1e-8):
    """Project Hspo onto physical Fock states, compare eigenvalues to exact Fock H."""
    spN=IBMSpace(E.N); spN2=IBMSpace(E.N-2)
    if beta0 is None:
        Hex=build_H(spN,spN2,h0,h2,C).toarray()
    else:
        Hex=cr.H_first_order(spN,spN2,beta0,h2).toarray()
    states=spN.states
    bits=[fock_to_bits(st,E.q,E.enc) for st in states]
    bitpos={b:i for i,b in enumerate(bits)}
    dimP=len(states)
    Hp=np.zeros((dimP,dimP),complex)
    labels=[p.to_label() for p in Hspo.paulis]; coeffs=Hspo.coeffs
    for j,b in enumerate(bits):
        for lab,cf in zip(labels,coeffs):
            c,nb=pauli_on_basis(lab,b,E.nq)
            i=bitpos.get(nb)
            if i is not None:
                Hp[i,j]+=cf*c
    Hp=0.5*(Hp+Hp.conj().T)
    ev_enc=np.linalg.eigvalsh(Hp); ev_ex=np.linalg.eigvalsh(0.5*(Hex+Hex.conj().T))
    maxdiff=np.max(np.abs(np.sort(ev_enc)-np.sort(ev_ex)))
    # also check H is block-diagonal wrt physical subspace (no leakage): row-sum of |offdiag to unphysical|
    return maxdiff, ev_ex[:4], ev_enc[:4]
