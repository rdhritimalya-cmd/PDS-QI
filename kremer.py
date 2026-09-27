"""
Kremer et al. reproduction (PRC 89, 041302; arXiv:1404.3826).

ECQF Hamiltonian (their Eq. 1):
    H = eps*[ (1-xi) n_d  -  (xi/(4N)) Q^chi . Q^chi ]
    Q^chi_mu = (s^dag d~ + d^dag s)_mu + chi (d^dag d~)^(2)_mu
Their example: O(6)-PDS + SU(3)-QDS along the low-Delta-sigma valley.

Goal: reproduce their published algebraic quantities (NOT QI) to validate our
machinery, then compute the QI quantities (which they did NOT) on the same states.

Published anchors to hit:
  Fig. 1: Delta_sigma_gs saturation ~2.47 (U(5), xi=0), ~1.25 (SU(3), xi=1),
          valley minimum ~1e-2 for the O(6)-like trajectory.
  Table I (N=14 examples), Delta_sigma_0 (ground state O(6) fluctuation):
     160Gd (xi=0.84, chi=-0.53): Delta_sigma_0=0.19, f_{sigma=N}=99.1%
     162Dy: Delta_sigma_0=0.07, f=99.9%
Delta_sigma = sqrt(<C2[O6]> - <sqrt(C2[O6])-ish>)... they define via sigma-fluctuation:
   Delta_sigma_Psi = sqrt( sum_i a_i^2 sigma_i^2 - (sum_i a_i^2 sigma_i)^2 )
where a_i^2 are probabilities of O(6) sigma-irreps. We compute sigma-decomposition
by projecting eigenstates onto C2[O(6)] eigenspaces (sigma(sigma+4)).
"""
import numpy as np, math
from ibm_pds import IBMSpace
import scipy.sparse as sp

def build(N):
    spN=IBMSpace(N); sel=np.where(spN.Mvals==0)[0]
    nd=spN.n_d()
    # Q^chi generators
    def Qchi(chi):
        out={}
        for q in range(-2,3):
            # (s^dag d~ + d^dag s)_q  : d~_q=(-1)^q d_{-q}
            t1=((-1)**q)*spN.hop(0,spN.mode_index(-q))   # s^dag d_{-q}
            t2=spN.hop(spN.mode_index(q),0)               # d_q^dag s
            t3=spN.dd_tensor(2,q)                          # (d^dag d~)^(2)_q
            out[q]=(t1+t2+chi*t3)
        return out
    return spN,sel,nd,Qchi

def C2_O6(spN):
    """C2[O(6)] = 2(N(N+4)) - ... use pairing form: C2[O6] with eigenvalue sigma(sigma+4).
    Standard: C2[O(6)] = 2[ N(N+4) - P^dag P ] with P^dag=(1/2)(d^dag d^dag - s^dag s^dag)?
    We build from generators to be safe: O(6) generators are L (from d) and the
    T^(3) octupole and the (s^dag d~ + d^dag s) dipole+... Actually O(6) has 15 gen:
    (d^dag d~)^(k) k=1,3 [O(5)] plus (s^dag d~ + d^dag s)^(2) [the pair coupling].
    C2[O(6)] = C2[O(5)] + sum_q (s d~ + d s)^(2)_q (s d~ + d s)^(2)_{-q}(-1)^q *norm.
    Verify by eigenvalue = sigma(sigma+4)."""
    from critical import C2_O5
    out=C2_O5(spN).tocsr() if hasattr(C2_O5(spN),'tocsr') else sp.csr_matrix(C2_O5(spN))
    out=sp.csr_matrix(C2_O5(spN))
    # add the O(6)/O(5) coset generator squared: R_q = (s^dag d~ + d^dag s)^(2)_q
    R={}
    for q in range(-2,3):
        t1=((-1)**q)*spN.hop(0,spN.mode_index(-q))
        t2=spN.hop(spN.mode_index(q),0)
        R[q]=(t1+t2)
    add=sp.csr_matrix((spN.dim,spN.dim))
    for q in range(-2,3):
        add=add+((-1)**q)*(R[q]@R[-q])
    return (out+add).tocsr()

if __name__=="__main__":
    N=14
    spN,sel,nd,Qchi=build(N)
    C2o6=C2_O6(spN).toarray()
    ev=np.linalg.eigvalsh(C2o6)
    sig=(-2+np.sqrt(4+ev))  # sigma(sigma+4)=ev -> sigma=-2+sqrt(4+ev)
    dev=np.abs(sig-np.round(sig))
    print(f"C2[O(6)] eigenvalue -> sigma integer? max dev = {dev.max():.2e}")
    print(f"  sigma range: {sig.min():.2f} .. {sig.max():.2f} (expect 0..N={N})")
    # allowed sigma for [N] symmetric: sigma=N,N-2,...,0 or 1
    print(f"  distinct sigma present: {sorted(set(np.round(sig).astype(int)))}")
