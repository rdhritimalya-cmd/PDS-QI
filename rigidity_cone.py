"""
Rigidity-cone test (tautology-killer).

Claim to test: solvable states are rigid along the WHOLE PDS-preserving cone
{h0, h4, h5} but a generic eigenstate is not; and rigidity FAILS along an
SU(3)-breaking direction (n_d, i.e. a U(5) Casimir term) for the states that
are only "accidentally" pure.

Directions (all built from operators already in code):
  D_h0 : P0^dag P0                      (PDS-preserving)
  D_h4 : P0^dag (s^dag s) P0            (PDS-preserving)
  D_nd : n_d                            (SU(3)-breaking; a real DS-mixing term)

Protocol: fix base H_PDS(h0=0.35,h2=1,C=0.0523). For each direction D and a
small step t, form H+tD, re-diagonalize, match states, measure per-state drift
of S_{s|d}. A state is 'rigid along D' if its wavefunction (hence every QI
measure) is stationary to the perturbation, i.e. D|psi> is parallel to |psi>
(D adds only a phase/energy shift) OR annihilates it.

The cleaner, step-free diagnostic (used here as primary): for eigenstate |psi>,
rigidity along D  <=>  the off-diagonal coupling  ||(1-P_psi) D |psi>|| = 0,
i.e. first-order perturbation theory moves the state by
   |delta psi> = sum_{m!=n} <m|D|n>/(E_n-E_m) |m|.
We report R_D(psi) = || Q D |psi> ||  with Q = 1 - |psi><psi| projected out of
the FULL degenerate eigenspace at E_n (so trivial in-space rotations don't count).
R_D = 0  => exactly rigid along D to first order (and, if D|psi> in-space, to all).
"""
import numpy as np
from ibm_pds import IBMSpace, build_H

def analyze(N, h0=0.35, h2=1.0, C=0.0523):
    spN = IBMSpace(N); spN2 = IBMSpace(N - 2)
    sel = np.where(spN.Mvals == 0)[0]
    H = build_H(spN, spN2, h0, h2, C)[np.ix_(sel, sel)].toarray()
    C2 = spN.C2_SU3()[np.ix_(sel, sel)].toarray()
    L2 = spN.L2()[np.ix_(sel, sel)].toarray()

    # PDS-preserving directions
    P0d, _ = spN.pair_ops(spN2); P0 = P0d.conj().T
    D_h0 = (P0d @ P0)[np.ix_(sel, sel)].toarray()
    D_h4 = (P0d @ (spN2.n_s() @ P0))[np.ix_(sel, sel)].toarray()
    # SU(3)-breaking direction: n_d (U(5) linear Casimir)
    D_nd = spN.n_d()[np.ix_(sel, sel)].toarray()

    ev, V = np.linalg.eigh(H)
    # degeneracy-aware rotation H->L2->C2 so states carry good L and (when pure) good irrep
    def refine(ev, V):
        V = V.copy()
        i = 0
        while i < len(ev):
            j = i
            while j + 1 < len(ev) and ev[j + 1] - ev[i] < 1e-9:
                j += 1
            if j > i:
                sub = V[:, i:j + 1]
                w, u = np.linalg.eigh(sub.conj().T @ L2 @ sub); sub = sub @ u
                k = 0
                while k <= j - i:
                    l = k
                    while l + 1 <= j - i and w[l + 1] - w[k] < 1e-9:
                        l += 1
                    if l > k:
                        s2 = sub[:, k:l + 1]
                        _, u2 = np.linalg.eigh(s2.conj().T @ C2 @ s2)
                        sub[:, k:l + 1] = s2 @ u2
                    k = l + 1
                V[:, i:j + 1] = sub
            i = j + 1
        return V
    V = refine(ev, V)

    # degenerate-subspace projector at each state's energy
    groups = []
    i = 0
    while i < len(ev):
        j = i
        while j + 1 < len(ev) and ev[j + 1] - ev[i] < 1e-9:
            j += 1
        groups.append(list(range(i, j + 1))); i = j + 1
    grp_of = {}
    for g in groups:
        for idx in g:
            grp_of[idx] = g

    def rigidity(D):
        R = np.zeros(len(ev))
        for n in range(len(ev)):
            g = grp_of[n]
            Pg = V[:, g]                      # ON basis of the degenerate eigenspace
            v = V[:, n]
            Dv = D @ v
            # project OUT the whole degenerate eigenspace
            coeffs = Pg.conj().T @ Dv
            Dv_out = Dv - Pg @ coeffs
            R[n] = np.linalg.norm(Dv_out)
        return R

    R_h0 = rigidity(D_h0); R_h4 = rigidity(D_h4); R_nd = rigidity(D_nd)
    c2 = np.einsum('in,ij,jn->n', V.conj(), C2, V).real
    c2var = np.einsum('in,ij,jn->n', V.conj(), C2 @ C2, V).real - c2 ** 2
    l2 = np.einsum('in,ij,jn->n', V.conj(), L2, V).real
    Lv = (-1 + np.sqrt(1 + 4 * np.clip(l2, 0, None))) / 2

    pure = c2var < 1e-6

    # classify: 'listed solvable' vs 'forced-solvable' vs 'mixed'
    # forced-solvable = pure but not annihilated by BOTH pair operators in listed family.
    # Operationally: pure AND rigid along h0 AND rigid along h4 -> true PDS cone member.
    tol = 1e-8
    cone = pure & (R_h0 < tol) & (R_h4 < tol)
    print(f"\n=== N={N} ===  states={len(ev)}  pure={pure.sum()}  cone(h0&h4 rigid)={cone.sum()}")
    print(f"{'class':>16} {'n':>4} {'R_h0':>10} {'R_h4':>10} {'R_nd':>10}")
    def stat(mask, name):
        if mask.sum() == 0:
            print(f"{name:>16} {0:>4}"); return
        print(f"{name:>16} {mask.sum():>4} "
              f"{R_h0[mask].max():>10.2e} {R_h4[mask].max():>10.2e} {R_nd[mask].max():>10.2e}"
              f"   (max over class)")
    stat(cone, "PDS-cone")
    stat(pure & ~cone, "pure non-cone")
    stat(~pure, "mixed")
    # KEY CONTRAST: does the SU(3)-breaking direction distinguish cone vs pure-non-cone?
    print(f"\n  R_nd on PDS-cone states:      max={R_nd[cone].max():.3e}  "
          f"mean={R_nd[cone].mean():.3e}")
    if (pure & ~cone).sum():
        m = pure & ~cone
        print(f"  R_nd on pure-non-cone states: max={R_nd[m].max():.3e}  mean={R_nd[m].mean():.3e}")
    print(f"  R_nd on mixed states:         max={R_nd[~pure].max():.3e}  "
          f"mean={R_nd[~pure].mean():.3e}")
    # Interpretation guide:
    #   Cone states rigid along h0 & h4 (PDS directions) but NOT along n_d  => the
    #   rigidity is DIRECTION-SELECTIVE, not a tautology. A tautological 'rigidity'
    #   would make them rigid along every direction.
    return dict(R_h0=R_h0, R_h4=R_h4, R_nd=R_nd, cone=cone, pure=pure, ev=ev, Lv=Lv)

if __name__ == "__main__":
    for N in (8, 10):
        analyze(N)
