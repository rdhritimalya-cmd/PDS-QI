"""
qsim_echo.py  --  The dynamical form of the label-variance diagnostic.

The static diagnostic Var C2[G] (manuscript Sec. III) has an exact dynamical
counterpart: the Loschmidt echo under Casimir evolution,

        E(t) = |<psi| e^{-i C2[G] t} |psi>|^2 .

Because a solvable (exact-label) state is an eigenstate of C2[G], it is RIGID:
E(t) = 1 for all t.  A mixed state is a superposition of C2 blocks and dephases,
E(t) < 1.  The short-time expansion is

        E(t) = 1 - t^2 Var C2[G] + O(t^4),

so the echo is the O(1), bounded repackaging of the label variance: the quantum
evolution performs, through phase interference, the same separation that the
static variance performs by subtracting two O(N^4) moments.  Time evolution is
therefore the diagnostic read dynamically, not an auxiliary benchmark.

This module evaluates E(t) on the exact eigenstates of the N=3 first-order
critical block and reports (i) the solvable/mixed echo separation and (ii) the
numerical agreement with 1 - t^2 Var C2.  No quantum-advantage claim is made:
the model is classically diagonalizable and E(t) is computed here by exact
matrix exponentiation; the point is the structural identity, which is what a
device would measure.

Reproduces manuscript Fig. (echo).  Requires: numpy, scipy, ibm_pds, critical.
"""
import numpy as np, math
from scipy.linalg import expm
from ibm_pds import IBMSpace
from critical import H_first_order


def critical_block(N):
    """Return (H, C2, L2, nd) of the first-order critical Hamiltonian H(beta0=sqrt2),
    restricted to the M=0 block (one member of each L-multiplet)."""
    spN = IBMSpace(N); spN2 = IBMSpace(N - 2)
    m0 = np.where(np.abs(spN.Mvals) < 1e-9)[0]; ix = np.ix_(m0, m0)
    H = H_first_order(spN, spN2, math.sqrt(2.0)).toarray().real[ix]
    C2 = spN.C2_SU3().toarray().real[ix]
    L2 = spN.L2().toarray().real[ix]
    nd = spN.n_d().toarray().real[ix]
    return 0.5 * (H + H.T), 0.5 * (C2 + C2.T), L2, nd


def resolved_eigenstates(H, C2, L2, nd, deg_tol=1e-7):
    """Degeneracy-aware eigenstates: within each degenerate H-level, diagonalize
    L2 then C2 then nd so each eigenvector carries a definite symmetry label."""
    e, V = np.linalg.eigh(H); i = 0
    while i < len(e):
        j = i
        while j + 1 < len(e) and e[j + 1] - e[i] < deg_tol:
            j += 1
        if j > i:
            sub = V[:, i:j + 1]
            for op in (L2, C2, nd):
                _, u = np.linalg.eigh(sub.T @ op @ sub); sub = sub @ u
            V[:, i:j + 1] = sub
        i = j + 1
    return e, V


def label_variance(v, C2):
    return float(v @ C2 @ C2 @ v - (v @ C2 @ v) ** 2)


def echo(v, C2n, t):
    """E(t) = |<v| e^{-i C2n t} |v>|^2 for a real eigenvector v."""
    U = expm(-1j * C2n * t)
    return abs(v.astype(complex) @ (U @ v.astype(complex))) ** 2


def run(N=3, tgrid=None, thr=1e-6, verbose=True):
    H, C2, L2, nd = critical_block(N)
    _, V = resolved_eigenstates(H, C2, L2, nd)
    d = H.shape[0]
    lv = np.array([label_variance(V[:, k], C2) for k in range(d)])
    solv = np.where(lv < thr)[0]; mixed = np.where(lv >= thr)[0]
    span = np.ptp(np.linalg.eigvalsh(C2)); C2n = C2 / span   # normalize evolution time
    if tgrid is None:
        tgrid = np.linspace(0.0, 8.0, 33)
    Es = np.array([[echo(V[:, k], C2n, t) for t in tgrid] for k in range(d)])
    out = {"N": N, "d": d, "solv": solv, "mixed": mixed, "labvar": lv,
           "tgrid": tgrid, "echo": Es, "C2span": span, "V": V, "C2": C2}
    if verbose:
        print(f"N={N}: block dim {d}, {len(solv)} solvable, {len(mixed)} mixed; C2 span {span:.1f}")
        print(f"{'t':>6} {'min echo(solv)':>15} {'max echo(mix)':>14} {'gap':>8}")
        for ti, t in enumerate(tgrid):
            if ti % 4 != 0:
                continue
            es = Es[solv, ti].min(); em = Es[mixed, ti].max()
            print(f"{t:>6.2f} {es:>15.5f} {em:>14.5f} {es - em:>8.4f}")
        # short-time identity check
        tt = 0.1
        print(f"\nShort-time identity  E(t) = 1 - t^2 Var C2  (normalized), t={tt}:")
        for k in list(mixed)[:3]:
            vn = label_variance(V[:, k], C2) / span ** 2
            print(f"  mixed state {k}: echo={echo(V[:,k],C2n,tt):.6f}  "
                  f"1 - t^2 Var={1 - tt**2 * vn:.6f}")
        print(f"\nSolvable states are exact C2-eigenstates -> echo == 1 for all t:")
        for k in solv[:3]:
            print(f"  state {k}: echo(t=0.5)={echo(V[:,k],C2n,0.5):.8f}, "
                  f"echo(t=8)={echo(V[:,k],C2n,8.0):.8f}, Var C2={lv[k]:.1e}")
    return out


if __name__ == "__main__":
    run(N=3)
