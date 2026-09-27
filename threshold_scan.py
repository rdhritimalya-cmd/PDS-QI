"""
threshold_scan.py -- robustness of the pure/mixed classification (Table I).

At a stable SU(3)-PDS point (N=10, 203 states in the M=0 block) the number of
states with Var C2[SU(3)] below a cut is constant over eight decades, dropping
only at 1e-10 (the numerical noise floor of the degeneracy refinement). This
reproduces Table I of the manuscript and shows the 1e-6 threshold is not fine
tuned: the gap between the largest exact-label variance (~6e-10) and the
smallest nonzero one (~24) spans ten orders of magnitude.
"""
import numpy as np
from ibm_pds import IBMSpace, build_H


def _refine(ev, V, ops, tol=1e-9):
    V = V.copy(); n = len(ev)

    def rec(sub, ops):
        if not ops or sub.shape[1] == 1:
            return sub
        w, u = np.linalg.eigh(sub.T @ ops[0] @ sub); sub = sub @ u
        out = []; k = 0
        while k < len(w):
            l = k
            while l + 1 < len(w) and w[l + 1] - w[k] < tol * max(1, abs(w[k])):
                l += 1
            out.append(rec(sub[:, k:l + 1], ops[1:])); k = l + 1
        return np.hstack(out)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and ev[j + 1] - ev[i] < tol:
            j += 1
        if j > i:
            V[:, i:j + 1] = rec(V[:, i:j + 1], ops)
        i = j + 1
    return V


def scan(N=10, h0=0.35, h2=1.0, C=0.0523):
    s = IBMSpace(N); s2 = IBMSpace(N - 2); sel = np.where(s.Mvals == 0)[0]
    R = lambda X: X.tocsr()[sel][:, sel].toarray()
    H = R(build_H(s, s2, h0, h2, C)); C2 = R(s.C2_SU3())
    L2 = R(s.L2()); nd = R(s.n_d())
    ev, V = np.linalg.eigh(H); W = _refine(ev, V, [L2, C2, nd])
    AV = C2 @ W
    v2 = np.einsum('ij,ij->j', AV, AV) - np.einsum('ij,ij->j', W, AV) ** 2
    thr = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    print(f"Threshold scan at a stable SU(3)-PDS point (N={N}, {len(sel)} states):")
    print("  threshold  : " + "  ".join(f"{t:.0e}" for t in thr))
    print("  #exact-label: " + "  ".join(f"{int((v2 < t).sum()):5d}" for t in thr))
    print(f"  largest exact-label variance = {np.sort(v2[v2 < 1e-6])[-1]:.1e}")
    print(f"  smallest nonzero variance    = {np.sort(v2[v2 > 1e-6])[0]:.1f}")


if __name__ == "__main__":
    scan()
