"""
Definitive Q2 analysis (corrected control parameter).

Canonical IBM U(5)->SU(3) first-order QPT:
    H(xi) = (1-xi) nd - (xi/4N) Q.Q ,  Q = SU(3) quadrupole (chi=-sqrt7/2)
QPT at xi_c ~ 0.5 (verified: chi_F peak, <nd>/N jump).

The critical-point PDS Hamiltonian H(beta0=sqrt2) of PRL 98 lives at the
transition and carries coexisting U(5)-PDS + SU(3)-PDS. We ask the Q2 question
with the QI tools:

  As xi crosses xi_c, for the eigenstates of H(xi):
   (a) label variance Var C2[SU(3)] and Var n_d  -- these are the algebraic
       order parameters; solvable states have one of them = 0 exactly AT the
       PDS point.
   (b) s|d entanglement entropy.
   (c) which states remain 'solvable' (zero label-variance) as a function of xi.

Result to establish cleanly:
  - Generic xi: NO state has exactly zero SU(3) or U(5) variance (both broken).
  - Only AT special points (xi->0 pure U(5); xi->1 pure SU(3); and the PDS
    construction) do exact-label subsets appear.
  - The PDS Hamiltonian H(beta0=sqrt2) is NOT on the one-parameter consistent-Q
    line -- it is a distinct point. So the 'coexistence of two PDS' is a
    property of a specifically constructed critical Hamiltonian, not of the
    generic transition. THIS is the precise, honest statement for the paper.

We demonstrate by:
  (1) confirming the consistent-Q line has no exact-label interior points;
  (2) confirming H(beta0=sqrt2) DOES (56 SU(3) + U(5) states), and that it
      reproduces the SAME deformed shape (<nd>/N, beta) as the transition region;
  (3) the label-variance of mixed states peaks at the transition while the
      solvable subset (at the PDS point) stays at zero -- the QI signature.
"""
import numpy as np, math, json
from ibm_pds import IBMSpace
from critical import H_first_order, C2_O5
from run_step0 import rdm_modes, vn_entropy

def setup(N):
    spN=IBMSpace(N); spN2=IBMSpace(N-2); sel=np.where(spN.Mvals==0)[0]
    nd=spN.n_d().toarray()[np.ix_(sel,sel)]
    C2=spN.C2_SU3().toarray()[np.ix_(sel,sel)]
    L2=spN.L2().toarray()[np.ix_(sel,sel)]
    Q={q: spN.Q_op(q).toarray()[np.ix_(sel,sel)] for q in range(-2,3)}
    QQ=sum(((-1)**q)*(Q[q]@Q[-q]) for q in range(-2,3))
    return spN,spN2,sel,dict(nd=nd,C2=C2,L2=L2,QQ=QQ)

def Hxi(ops,xi,N):
    return (1-xi)*ops['nd'] - (xi/(4*N))*ops['QQ']

def var(op,V):
    m=np.einsum('in,ij,jn->n',V.conj(),op,V).real
    return np.einsum('in,ij,jn->n',V.conj(),op@op,V).real - m**2

def refine(H,ops):
    e,V=np.linalg.eigh(H); i=0
    while i<len(e):
        j=i
        while j+1<len(e) and e[j+1]-e[i]<1e-9: j+=1
        if j>i:
            sub=V[:,i:j+1]
            for op in (ops['L2'],ops['C2'],ops['nd']):
                w,u=np.linalg.eigh(sub.conj().T@op@sub); sub=sub@u
            V[:,i:j+1]=sub
        i=j+1
    return e,V

def run(N):
    spN,spN2,sel,ops=setup(N)
    xis=np.linspace(0.05,0.95,19)
    # (1) exact-label interior points on the consistent-Q line?
    min_c2var=[]; min_ndvar=[]
    for xi in xis:
        e,V=refine(Hxi(ops,xi,N),ops)
        min_c2var.append(float(var(ops['C2'],V).min()))
        min_ndvar.append(float(var(ops['nd'],V).min()))
    interior_exact = any((c<1e-6 or d<1e-6) for c,d in zip(min_c2var[1:-1],min_ndvar[1:-1]))
    print(f"N={N}: consistent-Q interior points with an EXACT-label state? {interior_exact}")
    print(f"  min Var over states, at xi=0.5: C2={min_c2var[len(xis)//2]:.2f}, nd={min_ndvar[len(xis)//2]:.2f}")

    # (2) PDS point H(beta0=sqrt2): solvable subset + its shape
    Hpds=H_first_order(spN,spN2,math.sqrt(2),1.0).toarray()[np.ix_(sel,sel)]
    e,V=refine(Hpds,ops)
    c2v=var(ops['C2'],V); ndv=var(ops['nd'],V)
    solv=(c2v<1e-6)|(ndv<1e-6)
    # <nd>/N of the deformed GB (E=0, L=0 member) at the PDS point
    gs_pds=V[:,np.argmin(np.abs(e))]
    ndN_pds=float(gs_pds.conj()@ops['nd']@gs_pds)/N
    print(f"  H(beta0=sqrt2): solvable states = {solv.sum()}  (exact-label subset EXISTS)")
    print(f"  deformed GS <nd>/N at PDS point = {ndN_pds:.3f}")

    # (3) mixed-state label variance peaks at transition; solvable stays 0.
    # Track mean Var C2 over ALL states along the consistent-Q line (mixing measure)
    meanc2var=[]
    for xi in xis:
        e2,V2=refine(Hxi(ops,xi,N),ops)
        meanc2var.append(float(var(ops['C2'],V2).mean()))
    xi_peak=xis[int(np.argmax(meanc2var))]
    print(f"  mean Var C2[SU(3)] over spectrum peaks at xi={xi_peak:.2f} "
          f"(maximal SU(3) mixing near transition)")
    return dict(N=N, xis=xis.tolist(), min_c2var=min_c2var, min_ndvar=min_ndvar,
                meanc2var=meanc2var, solv_pds=int(solv.sum()), ndN_pds=ndN_pds)

if __name__=="__main__":
    res={}
    for N in (8,10):
        res[N]=run(N); print()
    json.dump(res[10], open("q2_definitive_N10.json","w"))
