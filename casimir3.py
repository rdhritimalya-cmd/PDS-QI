"""
casimir3.py -- cubic SU(3) Casimir and the conjugate-irrep injectivity check.

The quadratic Casimir C2[SU(3)] with eigenvalue
    f2(l,m) = l^2 + m^2 + l m + 3(l+m)
is symmetric under (l,m) <-> (m,l), so conjugate irreps are C2-degenerate.
In the symmetric U(6) irrep [N] such pairs occur whenever N is a multiple of 3
(within the range used in the manuscript: N = 6, 9, 12, 15). At those N a
vanishing Var C2[SU(3)] certifies support in one C2 eigenspace, which may hold
two conjugate irreps rather than one.

The independent invariant that separates conjugate irreps is the cubic Casimir
C3[SU(3)], whose eigenvalue
    g3(l,m) = (l-m)(2l+m+3)(l+2m+3)
is antisymmetric under conjugation and therefore nonzero on every conjugate
pair. We build C3 as a scalar (rank-0) coupling of the SU(3) generators Q and L,
fixing the single free coefficient by demanding [C3, Q] = 0, then verify:

  (i) C3 commutes with all SU(3) generators to ~1e-12;
  (ii) on each C2-degenerate eigenspace, C3 splits the conjugate irreps
       (two distinct eigenvalues +/- g3);
  (iii) every state with Var C2[SU(3)] = 0 also has Var C3[SU(3)] = 0, at both
        the stable and first-order-critical points, so no conjugate-irrep
        mixture is ever mistaken for an exact-label state and no count changes.

Reproduces the check reported in Sec. III of the manuscript.
"""
import numpy as np
import scipy.sparse as sp
from ibm_pds import IBMSpace, build_H, cg, f2
from critical import H_first_order


def g3(lam, mu):
    """Cubic-Casimir eigenvalue (antisymmetric under (l,m)<->(m,l))."""
    return (lam - mu) * (2 * lam + mu + 3) * (lam + 2 * mu + 3)


def _couple(A, B, ka, kb, k, q):
    """Rank-k, component-q spherical-tensor coupling of tensors A(ka), B(kb)."""
    out = None
    for m1 in range(-ka, ka + 1):
        m2 = q - m1
        if abs(m2) > kb:
            continue
        c = cg(ka, m1, kb, m2, k, q)
        if c == 0.0:
            continue
        t = c * (A[m1] @ B[m2])
        out = t if out is None else out + t
    return out


def build_C3(space):
    """Construct the cubic SU(3) Casimir on the full space of `space`.

    Returns (C3_sparse, b), where b is the fitted L-Q-L admixture coefficient.
    C3 = [Q x Q]^(2) . Q  +  b * [L x Q]^(1) . L , with b fixed by [C3, Q] = 0.
    """
    Q = {q: space.Q_op(q) for q in range(-2, 3)}
    L = {q: space.L_op(q) for q in range(-1, 2)}
    QQ2 = {q: _couple(Q, Q, 2, 2, 2, q) for q in range(-2, 3)}
    QQQ = _couple(QQ2, Q, 2, 2, 0, 0)
    LQ1 = {q: _couple(L, Q, 1, 2, 1, q) for q in range(-1, 2)}
    LQL = _couple(LQ1, L, 1, 1, 0, 0)
    # fix b by least-squares nulling of [QQQ + b LQL, Q_+1]
    A1 = (QQQ @ Q[1] - Q[1] @ QQQ).toarray().ravel()
    A2 = (LQL @ Q[1] - Q[1] @ LQL).toarray().ravel()
    b = -np.dot(A1, A2) / np.dot(A2, A2)
    C3 = (QQQ + b * LQL).tocsr()
    return C3, b


def _var(V, A):
    AV = A @ V
    m = np.einsum('ij,ij->j', V, AV)
    m2 = np.einsum('ij,ij->j', AV, AV)
    return m2 - m * m


def _refine(ev, V, ops, tol=1e-9):
    """Order-fixed degeneracy refinement (diagonalize ops in sequence)."""
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


def check(N, kind='stable', h0=0.35, h2=1.0, C=0.0523):
    s = IBMSpace(N)
    s2 = IBMSpace(N - 2)
    sel = np.where(s.Mvals == 0)[0]
    R = lambda X: X.tocsr()[sel][:, sel].toarray()
    if kind == 'stable':
        H = R(build_H(s, s2, h0, h2, C))
    elif kind == 'first':
        H = R(H_first_order(s, s2, np.sqrt(2.0)))
    else:
        raise ValueError(kind)
    C2 = R(s.C2_SU3())
    L2 = R(s.L2())
    nd = R(s.n_d())
    C3full, b = build_C3(s)
    C3 = R(C3full)
    C3 = (C3 + C3.T) / 2.0
    comm = max(abs((C3full @ s.Q_op(q) - s.Q_op(q) @ C3full)).max() for q in range(-2, 3))

    ev, V = np.linalg.eigh(H)
    W = _refine(ev, V, [L2, C2, nd])
    v2 = _var(W, C2)
    v3 = _var(W, C3)
    z2 = v2 < 1e-6
    z3 = v3 < 1e-6
    false_pos = int((z2 & ~z3).sum())

    # C2-degenerate eigenspaces and how C3 splits them
    w, U = np.linalg.eigh(C2)
    vals = np.round(w, 6)
    splits = []
    for val in sorted(set(vals)):
        idx = np.where(vals == val)[0]
        sub = U[:, idx]
        w3 = np.linalg.eigvalsh(sub.T @ C3 @ sub)
        d = sorted(set(np.round(w3, 3)))
        if len(d) > 1:
            splits.append((float(val), d, len(idx)))

    print(f"{kind:6s} N={N:2d}: b={b:.6f}  max|[C3,Q]|={comm:.1e}  "
          f"VarC2=0 -> {int(z2.sum())} states, "
          f"of which VarC3!=0 (conjugate-mix false positives) = {false_pos}")
    for val, d, dim in splits:
        print(f"          C2={val:.0f} (dim {dim}): C3 eigenvalues {d}  <- conjugate pair split")
    return int(z2.sum()), false_pos


if __name__ == "__main__":
    print("Conjugate-irrep injectivity check via the cubic Casimir C3[SU(3)].")
    print("A false positive is a state with VarC2=0 but VarC3!=0 (a conjugate-irrep")
    print("mixture masquerading as exact-label). The manuscript claim is zero of these.\n")
    for kind in ('stable', 'first'):
        for N in (6, 8, 9, 10, 12):
            check(N, kind)
        print()
