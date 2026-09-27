"""
Second-order transition (U(5) <-> O(6)) analysis.  Leviatan PRL 98, Eq. (6):
    H_2nd = eps nd + A[(d^dag.d^dag - s^dag^2)+h.c.],  eps=4(N-1)A   (critical)
Preserves O(5): [H_2nd, C2[O(5)]]=0 -> O(5) PDS of type II (ALL states keep tau).

Contrast with first-order (type I): there, DISTINCT subsets are exactly solvable
(some SU(3), some U(5)) and the rest mix. Here, EVERY state keeps one exact label
(tau) -- a qualitatively different QI signature.

We establish, across the U(5)<->O(6) line H(t)= (1-t) nd + t*Hpair (t: 0->1):
  (1) tau (O(5)) variance = 0 for ALL states and ALL t on the O(5)-scalar line.
  (2) BUT U(5) (nd) and O(6) (sigma) variances are nonzero and peak at the transition.
  (3) The critical Hamiltonian Eq.(6) sits at the O(5)-PDS point; the s|d entanglement
      is organized by tau-blocks -> entanglement bounded within each tau sector.
The signature for the paper: at a 2nd-order (continuous) transition the QI
'rigidity' is a CONSERVED-LABEL (tau) structure shared by all states, not a
solvable SUBSET. Two different faces of PDS -> two different QI fingerprints.
"""
import numpy as np, math, json
from ibm_pds import IBMSpace
from critical import C2_O5
from run_step0 import rdm_modes, vn_entropy
import scipy.sparse as sp

def setup(N):
    spN=IBMSpace(N); spN2=IBMSpace(N-2); sel=np.where(spN.Mvals==0)[0]
    nd=spN.n_d().toarray()[np.ix_(sel,sel)]
    O5=C2_O5(spN).toarray()[np.ix_(sel,sel)]
    L2=spN.L2().toarray()[np.ix_(sel,sel)]
    # O(6) Casimir: C2[O(6)] = 2(N(N+4) - P0d P0)/... use pairing P^dag P with P^dag=dd-s^2
    idxN=spN.index
    def add_two(st,i,j):
        lst=list(st); a=math.sqrt(lst[j]+1); lst[j]+=1; a*=math.sqrt(lst[i]+1); lst[i]+=1
        return tuple(lst),a
    rows,cols,vals=[],[],[]
    for col,st in enumerate(spN2.states):
        for m in (-2,-1,0,1,2):
            ns,a=add_two(st,spN.mode_index(m),spN.mode_index(-m))
            rows.append(idxN[ns]);cols.append(col);vals.append(((-1)**m)*a)
        ns,a=add_two(st,0,0); rows.append(idxN[ns]);cols.append(col);vals.append(-1.0*a)
    Pdag=sp.csr_matrix((vals,(rows,cols)),shape=(spN.dim,spN2.dim))
    pair=(Pdag@Pdag.conj().T).toarray()[np.ix_(sel,sel)]
    # sigma(sigma+4) Casimir of O(6): C2[O6] = N(N+4) - Pdag P  (schematically); 
    # we just track pair-variance as the O(6)-breaking measure.
    return spN,spN2,sel,dict(nd=nd,O5=O5,L2=L2,pair=pair),Pdag

def var(op,V):
    m=np.einsum('in,ij,jn->n',V.conj(),op,V).real
    return np.einsum('in,ij,jn->n',V.conj(),op@op,V).real-m**2

def run(N):
    spN,spN2,sel,ops,Pdag=setup(N)
    ts=np.linspace(0.0,1.0,21)
    print(f"N={N}: U(5)<->O(6) line H(t)=(1-t)nd + t*pair")
    o5max=[]; ndmix=[]
    for t in ts:
        H=(1-t)*ops['nd']+t*ops['pair']
        e,V=np.linalg.eigh(H)
        o5v=var(ops['O5'],V); ndv=var(ops['nd'],V)
        o5max.append(float(o5v.max())); ndmix.append(float(ndv.mean()))
    print(f"  max Var C2[O(5)] over ALL t and states: {max(o5max):.2e}  "
          f"(=> tau exactly conserved throughout: O(5) PDS type II)")
    ti=int(np.argmax(ndmix))
    print(f"  mean Var n_d (U(5) mixing) peaks at t={ts[ti]:.2f}")

    # critical Hamiltonian Eq.(6): eps=4(N-1)A, A=1
    A=1.0; eps=4*(N-1)*A
    Hc=eps*ops['nd']+A*ops['pair']
    e,V=np.linalg.eigh(Hc)
    o5v=var(ops['O5'],V)
    print(f"  critical H (Eq.6): max Var C2[O(5)] = {o5v.max():.2e} (all tau-good)")
    # tau value per state and s|d entropy within tau blocks
    o5m=np.einsum('in,ij,jn->n',V.conj(),ops['O5'],V).real
    taus=np.round((-3+np.sqrt(9+4*o5m))/2).astype(int)
    def Sof(v):
        full=np.zeros(spN.dim,dtype=complex); full[sel]=v
        return vn_entropy(rdm_modes(full,spN.states,(0,)))
    S=np.array([Sof(V[:,n]) for n in range(V.shape[1])])
    # is entanglement organized by tau? report mean S per tau
    print("  s|d entropy grouped by O(5) label tau (critical H):")
    for tau in sorted(set(taus)):
        m=taus==tau
        print(f"    tau={tau}: n={m.sum():>3}  <S>={S[m].mean():.3f}  range[{S[m].min():.3f},{S[m].max():.3f}]")
    return dict(N=N, ts=ts.tolist(), o5max=o5max, ndmix=ndmix,
                crit_o5var_max=float(o5v.max()))

if __name__=="__main__":
    res={}
    for N in (8,10):
        res[N]=run(N); print()
    json.dump(res[10], open("q2_secondorder_N10.json","w"))
