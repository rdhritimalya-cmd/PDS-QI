"""Convention arbitration BEFORE any physics: 
(1) theta2 identity (2010.10951 Eq. 27a): P0dP0 + P2d.P2 = -C2[SU(3)] + 2N(2N+3)
(2) L2 spectrum must be L(L+1), integer L
(3) C2[SU(3)] spectrum must match f2(lam,mu) for allowed irreps
(4) solvable-state energies E_k = 6 h2 k (2N-2k+1) + C L(L+1) must appear exactly
"""
import numpy as np
import scipy.sparse as sp
from ibm_pds import IBMSpace, build_H, f2, cg
import math

def test_N(N, h0=0.35, h2=1.0, C=0.0523):
    print(f"\n================ N = {N} ================")
    spN = IBMSpace(N)
    spN2 = IBMSpace(N - 2)
    print(f"dim[N] = {spN.dim}")

    P0d, P2d = spN.pair_ops(spN2)
    lhs = (P0d @ P0d.conj().T).toarray()
    for mu in range(-2, 3):
        lhs += (P2d[mu] @ P2d[mu].conj().T).toarray()
    C2 = spN.C2_SU3().toarray()
    Nhat = N * np.eye(spN.dim)
    rhs = -C2 + 2 * N * (2 * N + 3) * np.eye(spN.dim)
    err = np.max(np.abs(lhs - rhs))
    print(f"[CHECK 1] theta2 identity max|LHS-RHS| = {err:.3e}")

    L2 = spN.L2().toarray()
    evalsL2 = np.linalg.eigvalsh(L2)
    Ls = (-1 + np.sqrt(1 + 4 * evalsL2)) / 2
    print(f"[CHECK 2] L2 -> L values: max deviation from integer = "
          f"{np.max(np.abs(Ls - np.round(Ls))):.3e}; L range {Ls.min():.2f}..{Ls.max():.2f}")

    evalsC2 = np.linalg.eigvalsh(C2)
    # match each distinct eigenvalue to some f2(lam,mu)
    distinct = np.unique(np.round(evalsC2, 6))
    allowed = {}
    for lam in range(0, 2 * N + 1):
        for mu in range(0, 2 * N + 1):
            allowed[f2(lam, mu)] = (lam, mu)
    bad = [v for v in distinct if not any(abs(v - k) < 1e-6 for k in allowed)]
    print(f"[CHECK 3] C2 distinct eigenvalues: {len(distinct)}; unmatched to f2(l,m): {len(bad)}")
    if len(bad):
        print("   unmatched:", bad[:10])
    matched = sorted({allowed[k] for v in distinct for k in allowed if abs(v - k) < 1e-6},
                     key=lambda t: -f2(*t))
    print(f"   irreps present: {matched}")

    # ---- Hamiltonian & solvable states
    H = build_H(spN, spN2, h0, h2, C).toarray()
    evals, evecs = np.linalg.eigh(H)
    # predicted solvable energies
    pred = []
    for k in range(0, N // 2 + 1):
        lam, mu = 2 * N - 4 * k, 2 * k
        Lmin = 2 * k if k > 0 else 0
        Lmax = 2 * N - 2 * k
        Lstep_list = (range(Lmin, Lmax + 1) if k > 0
                      else range(0, Lmax + 1, 2))
        for L in Lstep_list:
            E = 6 * h2 * k * (2 * N - 2 * k + 1) + C * L * (L + 1)
            pred.append((k, L, E, lam, mu))
    found = 0
    for (k, L, E, lam, mu) in pred:
        i = np.argmin(np.abs(evals - E))
        if abs(evals[i] - E) < 1e-8:
            # verify purity: <C2> and Var(C2)
            v = evecs[:, i]
            # careful: if degenerate, need the right combination; check dimension of match
            found += 1
    print(f"[CHECK 4] solvable energies found in spectrum: {found}/{len(pred)} "
          f"(naive nearest-eigenvalue match; degeneracy-aware check in full run)")
    return err

if __name__ == "__main__":
    for N in (3, 4, 5):
        test_N(N)
