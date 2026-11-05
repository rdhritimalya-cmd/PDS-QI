"""
revision_common.py -- shared helpers for the revision checks.

Degenerate levels are resolved by the NESTED refinement described in Appendix A of the
manuscript: diagonalize L^2 inside each energy cluster, then the next operator inside each
L^2 eigenspace, and so on. The order of the operators decides which symmetry is displayed
when the operators do not commute on the cluster (first-order critical point, Sec. IV B).
"""
import numpy as np
from ibm_pds import IBMSpace, f2

# stable SU(3)-PDS point used throughout (units with h2 = 1)
H0, H2, CC = 0.35, 1.0, 0.0523
TOL_E = 1e-7


def nested_refine(e, V, ops, tol=TOL_E, optol=1e-6):
    """Resolve energy clusters (width < tol) by diagonalizing ops[0], then ops[1] inside
    each ops[0] eigenspace, etc."""
    V = V.copy()

    def rec(sub, ops):
        if not ops or sub.shape[1] == 1:
            return sub
        w, u = np.linalg.eigh(sub.T @ ops[0] @ sub)
        sub = sub @ u
        out, k = [], 0
        while k < len(w):
            l = k
            while l + 1 < len(w) and w[l + 1] - w[k] < optol * max(1.0, abs(w[k])):
                l += 1
            out.append(rec(sub[:, k:l + 1], ops[1:]))
            k = l + 1
        return np.hstack(out)

    i = 0
    while i < len(e):
        j = i
        while j + 1 < len(e) and e[j + 1] - e[i] < tol:
            j += 1
        if j > i:
            V[:, i:j + 1] = rec(V[:, i:j + 1], ops)
        i = j + 1
    return V


def var(A, V):
    """Per-column variance and mean of operator A on the columns of V."""
    AV = A @ V
    m = np.einsum('ij,ij->j', V, AV)
    return np.einsum('ij,ij->j', AV, AV) - m ** 2, m


class Block:
    """M = 0 block of the fixed-N sd-IBM space with C2[SU(3)], L^2 and n_d."""

    def __init__(self, N):
        self.N = N
        self.s, self.s2 = IBMSpace(N), IBMSpace(N - 2)
        self.sel = np.where(self.s.Mvals == 0)[0]
        self.C2 = self.R(self.s.C2_SU3())
        self.L2 = self.R(self.s.L2())
        self.nd = self.R(self.s.n_d())

    def R(self, X):
        X = X.tocsr()[self.sel][:, self.sel].toarray().real
        return 0.5 * (X + X.T)

    def solve(self, H, ops=None):
        e, V = np.linalg.eigh(H)
        V = nested_refine(e, V, ops if ops is not None else [self.L2, self.C2, self.nd])
        return e, V

    def Ls(self, V):
        l2 = np.einsum('ij,ij->j', V, self.L2 @ V)
        return np.round((-1 + np.sqrt(1 + 4 * l2)) / 2).astype(int)


def tower_E(N, h2, C, L, k):
    """Closed-form energy of the SU(3) solvable tower member (k, L), or None."""
    ok = (k == 0 and L % 2 == 0 and L <= 2 * N) or (k > 0 and 2 * k <= L <= 2 * N - 2 * k)
    return 6 * h2 * k * (2 * N - 2 * k + 1) + C * L * (L + 1) if ok else None


def classify(b, e, V, h2, C):
    """Return (VarC2, pure mask, closed-form-solvable mask, L values)."""
    vc, _ = var(b.C2, V)
    Ls = b.Ls(V)
    pure = vc < 1e-6
    sol = np.zeros(len(e), bool)
    for i in np.where(pure)[0]:
        for k in range(b.N // 2 + 1):
            Et = tower_E(b.N, h2, C, Ls[i], k)
            if (Et is not None and abs(e[i] - Et) < 1e-7
                    and abs(V[:, i] @ b.C2 @ V[:, i] - f2(2 * b.N - 4 * k, 2 * k)) < 1e-5):
                sol[i] = True
    return vc, pure, sol, Ls
