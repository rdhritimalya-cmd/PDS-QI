"""
consistency_checks.py -- additional consistency checks:
 (1) gamma->g B(E2) ratios at 168Er (N=16) are independent of theta/alpha,
 (5) small-eps exponent for the eps*C[O(5)] breaking direction (N=10),
 (6) second-order point: tau, n_d and sigma purity (N=10),
 (7) multiplicity of L=3 for N=3,4,5.
Run from the repository root: python consistency_checks.py
"""
import numpy as np
from collections import Counter
from ibm_pds import IBMSpace, build_H, cg
from critical import H_second_order, C2_O5
from kremer import C2_O6
from revision_common import Block, nested_refine, var, H0, H2, CC

print("== (1) B(E2) ratio independence of theta/alpha, 168Er N=16")
s = IBMSpace(16); s2 = IBMSpace(14); sel = np.where(s.Mvals == 0)[0]
R = lambda X: X.tocsr()[sel][:, sel].toarray()
H = R(build_H(s, s2, 0.008, 0.004, 0.013)); L2 = R(s.L2()); ev, V = np.linalg.eigh(H)
l2 = np.einsum('ij,ij->j', V, L2 @ V); Ls = np.round((-1 + np.sqrt(1 + 4 * l2)) / 2).astype(int)
Q0 = R(s.Q_op(0)); sd0 = R(s.hop(s.mode_index(0), 0) + s.hop(0, s.mode_index(0)))
st = lambda L, n: V[:, [j for j in np.argsort(ev) if Ls[j] == L][n]]
g = {L: st(L, 0) for L in (0, 2, 4)}; gam = st(2, 1)
for ta in (1e-4, 1.0, 4.261, 100.0):
    T = Q0 + ta * sd0
    B = {Lf: (2 * Lf + 1) * (g[Lf] @ T @ gam) ** 2 / cg(2, 0, 2, 0, Lf, 0) ** 2 for Lf in (0, 2, 4)}
    print(f"  theta/alpha={ta}: {100 * B[0] / B[2]:.2f} : 100 : {100 * B[4] / B[2]:.2f}"
          f"   <g0|Q|gamma2>={g[0] @ Q0 @ gam:.1e}")

print("== (5) small-eps exponent for eps*C[O5], N=10")
b = Block(10)
Hst = b.R(build_H(b.s, b.s2, H0, H2, CC)); e0, V0 = b.solve(Hst)
vc0, _ = var(b.C2, V0); gb = V0[:, vc0 < 1e-6]
Co5 = b.R(C2_O5(b.s)); rows = []
for eps in (1e-6, 1e-5, 1e-4, 1e-3, 2e-3):
    ee, Ve = np.linalg.eigh(Hst + eps * Co5)
    tr = [Ve[:, int(np.argmax(np.abs(Ve.T @ gb[:, c])))] for c in range(gb.shape[1])]
    rows.append((eps, np.mean([var(b.C2, v[:, None])[0][0] for v in tr])))
    print(f"  eps={eps:.0e} meanVar={rows[-1][1]:.3e}")
a = np.array(rows)
print("  local exponents:", np.round(np.diff(np.log(a[:, 1])) / np.diff(np.log(a[:, 0])), 2))
print("  max |eigenvalue| of C[O5] on the block:", np.abs(np.linalg.eigvalsh(Co5)).max())

print("== (6) second-order point N=10: tau, n_d and sigma purity")
Hs2 = b.R(H_second_order(b.s)[0]); Co6 = b.R(C2_O6(b.s))
e, V = np.linalg.eigh(Hs2); V = nested_refine(e, V, [b.L2, Co5, b.nd])
vn, _ = var(b.nd, V); vs, _ = var(Co6, V); vt, mt = var(Co5, V)
tau = np.round((-3 + np.sqrt(9 + 4 * mt)) / 2).astype(int)
print(f"  tau-pure {np.sum(vt < 1e-6)}/{len(e)}, nd-pure {np.sum(vn < 1e-6)}, sigma-pure {np.sum(vs < 1e-6)}")
print("  tau of nd-pure states:", dict(Counter(tau[vn < 1e-6].tolist())),
      " states with tau >= N-1:", int(np.sum(tau >= 9)))

print("== (7) L=3 multiplicity for N=3,4,5")
for N in (3, 4, 5):
    bb = Block(N); w = np.linalg.eigvalsh(bb.L2)
    L = np.round((-1 + np.sqrt(1 + 4 * w)) / 2).astype(int)
    print(f"  N={N}: number of L=3 states = {np.sum(L == 3)}")
