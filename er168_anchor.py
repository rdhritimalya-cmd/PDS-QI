"""
er168_anchor.py -- physical anchor at Leviatan's fitted 168Er parameters.

Uses Leviatan's PRL 77, 818 (1996) fit: h0=0.008, h2=0.004, lambda=0.013 MeV,
N=16. Reproduces, directly from the diagonalized eigenstates (no fitting here):

  (1) the exact-label counts (VarC2[SU(3)]=0) and the closed-form solvable
      subset, for N=12,14,16;
  (2) the ground-band rigid-rotor energies E(2+),E(4+),E(6+) and the
      gamma-band head E(2+_gamma) = 6 h2 (2N-1) + 6 lambda;
  (3) the SU(3) admixtures of the beta-band head via block probabilities
      P_lambda = <psi|Pi_lambda|psi>, reproducing Leviatan's stated
      10% (26,0) and 3% (24,4) into the dominant (28,2);
  (4) the parameter-free gamma->g B(E2) ratios from the general one-body E2
      operator T(E2) = alpha Q + theta (d^dag s + s^dag dtil) with the fitted
      theta/alpha = 4.261, reproducing Leviatan's Table I PDS column
      (2+_gamma -> 0+_g : 2+_g : 4+_g = 64.3 : 100 : 6.3).

All B(E2) ratios below are computed from wave functions only; the single input
from data is the ratio theta/alpha, exactly as in Leviatan's analysis.
"""
import numpy as np
from ibm_pds import IBMSpace, build_H, f2, su3_irreps_in_UN, cg

H0, H2, LAM = 0.008, 0.004, 0.013
THETA_OVER_ALPHA = 4.261  # Leviatan's fitted E2-operator ratio for 168Er


def _refine(ev, V, ops, tol=1e-10):
    V = V.copy()
    n = len(ev)

    def rec(sub, ops):
        if not ops or sub.shape[1] == 1:
            return sub
        A = ops[0]
        w, u = np.linalg.eigh(sub.T @ A @ sub)
        sub = sub @ u
        out = []
        k = 0
        while k < len(w):
            l = k
            while l + 1 < len(w) and w[l + 1] - w[k] < tol * max(1, abs(w[k])):
                l += 1
            out.append(rec(sub[:, k:l + 1], ops[1:]))
            k = l + 1
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


def _var(V, A):
    AV = A @ V
    return np.einsum('ij,ij->j', AV, AV) - np.einsum('ij,ij->j', V, AV) ** 2


def counts():
    print("Exact-label vs closed-form solvable counts (168Er parameters):")
    for N in (12, 14, 16):
        s = IBMSpace(N); s2 = IBMSpace(N - 2); sel = np.where(s.Mvals == 0)[0]
        R = lambda X: X.tocsr()[sel][:, sel].toarray()
        H = R(build_H(s, s2, H0, H2, LAM)); C2 = R(s.C2_SU3())
        L2 = R(s.L2()); nd = R(s.n_d())
        ev, V = np.linalg.eigh(H); W = _refine(ev, V, [L2, C2, nd]); v2 = _var(W, C2)
        c2 = np.einsum('ij,ij->j', W, C2 @ W)
        l2 = np.einsum('ij,ij->j', W, L2 @ W)
        Ls = np.round((-1 + np.sqrt(1 + 4 * l2)) / 2).astype(int)
        irr = {int(f2(2 * N - 4 * k, 2 * k)): k for k in range(N // 2 + 1)}
        pure = np.where(v2 < 1e-6)[0]; cf = 0
        for i in pure:
            c = int(round(c2[i])); k = irr.get(c); L = Ls[i]
            if k is not None and abs(ev[i] - (6 * H2 * k * (2 * N - 2 * k + 1) + LAM * L * (L + 1))) < 1e-9:
                cf += 1
        print(f"  N={N}: exact-label(VarC2=0) = {len(pure)}, "
              f"closed-form solvable = {cf}, multiplicity partners = {len(pure) - cf}")


def spectrum_and_admixtures(N=16):
    s = IBMSpace(N); s2 = IBMSpace(N - 2); sel = np.where(s.Mvals == 0)[0]
    R = lambda X: X.tocsr()[sel][:, sel].toarray()
    H = R(build_H(s, s2, H0, H2, LAM)); C2 = R(s.C2_SU3()); L2 = R(s.L2())
    ev, V = np.linalg.eigh(H)
    l2 = np.einsum('ij,ij->j', V, L2 @ V)
    Ls = np.round((-1 + np.sqrt(1 + 4 * l2)) / 2).astype(int)

    print(f"\nGround-band rigid-rotor energies (N={N}):")
    for L in (0, 2, 4, 6):
        i = [j for j in np.argsort(ev) if Ls[j] == L][0]
        print(f"  E({L}+_g) = {ev[i]:.3f} MeV   (lambda*L(L+1) = {LAM * L * (L + 1):.3f})")
    ig2 = [j for j in np.argsort(ev) if Ls[j] == 2]
    print(f"  E(2+_gamma) = {ev[ig2[1]]:.3f} MeV   "
          f"(6 h2 (2N-1) + 6 lambda = {6 * H2 * (2 * N - 1) + 6 * LAM:.3f})")

    # block probabilities of the beta-band head (first excited 0+)
    w, U = np.linalg.eigh(C2)
    irr = {int(f2(*r)): r for r in su3_irreps_in_UN(N)}
    vals = np.round(w).astype(int)

    def probs(v):
        c = U.T @ v; out = {}
        for val in np.unique(vals):
            p = float(np.sum(c[vals == val] ** 2))
            if p > 1e-4:
                out[irr.get(val, val)] = round(p, 4)
        return out

    i0 = [j for j in np.argsort(ev) if Ls[j] == 0]
    beta = V[:, i0[1]]
    print(f"\nBeta-band head (0+, E={ev[i0[1]]:.2f} MeV) SU(3) admixtures:")
    print(f"  P_lambda = {probs(beta)}")
    print("  (Leviatan: dominant (28,2) with ~10% (26,0) and ~3% (24,4))")
    return s, sel, ev, V, Ls, R


def be2_ratios(N=16):
    s, sel, ev, V, Ls, R = spectrum_and_admixtures(N)
    Q0 = R(s.Q_op(0))
    # (d^dag s + s^dag dtil)_0 ; dtil_0 = d_0
    sd0 = R(s.hop(s.mode_index(0), 0) + s.hop(0, s.mode_index(0)))

    def state(L, n):
        idx = [j for j in np.argsort(ev) if Ls[j] == L]
        return V[:, idx[n]]

    gam2 = state(2, 1)                     # 2+ gamma band
    g = {L: state(L, 0) for L in (0, 2, 4)}  # ground band
    T = Q0 + THETA_OVER_ALPHA * sd0
    B = {}
    for Lf in (0, 2, 4):
        me = g[Lf] @ T @ gam2
        c = cg(2, 0, 2, 0, Lf, 0)          # <2 0 2 0 | Lf 0>
        B[Lf] = (2 * Lf + 1) * me ** 2 / c ** 2
    ratios = {Lf: round(100 * B[Lf] / B[2], 1) for Lf in (0, 2, 4)}
    print(f"\ngamma->g B(E2) ratios (theta/alpha = {THETA_OVER_ALPHA}):")
    print(f"  2+_gamma -> (0+_g : 2+_g : 4+_g) = "
          f"{ratios[0]} : {ratios[2]} : {ratios[4]}")
    print("  (Leviatan PRL 77 Table I, PDS column: 64.3 : 100 : 6.3)")


if __name__ == "__main__":
    counts()
    be2_ratios(16)
