"""
verify_revision.py -- verification of every exact-diagonalization number in the
revised manuscript (Secs. II-V and Tables I, II, IV). Run from the repository root:

    python verify_revision.py

Sections: (A) conventions, (B) O(5) commutator, (C) stable point Table I for even N
and threshold scan (Table IV), (D) first-order critical point in both degeneracy
readings (Sec. IV B, Fig. 1), (E) second-order point, (F) Table II and the
random consistent-Q sample, (G) consistent-Q line, (H) controlled breaking (Sec. V).
Odd N of Table I: oddN_scan.py. Expected output: logs/log_verify_revision.txt.
"""
import math, numpy as np, scipy.sparse as sp
from ibm_pds import IBMSpace, build_H, f2
from critical import H_first_order, H_second_order, C2_O5
from false_positive import H_cq
from revision_common import nested_refine, var, Block, tower_E, classify, H0, H2, CC


def section(t):
    print("\n" + "=" * 8 + " " + t)

# ---- A. conventions
section("A. arbiter identity and H(sqrt2) relation")
for N in (3, 4, 5, 8, 10):
    s, s2 = IBMSpace(N), IBMSpace(N - 2)
    P = build_H(s, s2, 1.0, 1.0, 0.0)
    Nop = sp.identity(s.dim) * N
    rhs = -s.C2_SU3() + 2 * Nop @ (2 * Nop + 3 * sp.identity(s.dim))
    d = abs((P - rhs).toarray()).max()
    Hc = H_first_order(s, s2, math.sqrt(2), 1.0); Hh2 = build_H(s, s2, 0.0, 1.0, 0.0)
    d2 = abs((Hc - 0.5 * Hh2).toarray()).max()
    print(f"N={N}: max|arbiter| = {d:.1e}   max|H(sqrt2) - h2term/2| = {d2:.1e}")

# ---- B. second-order O(5) commutator
section("B. second-order Hamiltonian commutes with C[O(5)]")
s = IBMSpace(10)
Hso = H_second_order(s)[0]; Co5 = C2_O5(s)
comm = (Hso @ Co5 - Co5 @ Hso).toarray()
print(f"N=10 ||[H, C_O5]||_F = {np.linalg.norm(comm):.1e}")

# ---- C. stable point classes, scaling, thresholds
section("C. stable SU(3)-PDS point (h0/h2=0.35, C/h2=0.0523)")
print(f"{'N':>3} {'dim':>5} {'I':>4} {'II':>3} {'III':>4} {'frac':>5} {'maxpure':>8} {'minmix':>7} {'medmix':>8}")
meds = []; Ns = (4, 6, 8, 10, 12, 14, 16)
for N in Ns:
    b = Block(N)
    H = b.R(build_H(b.s, b.s2, H0, H2, CC))
    e, V = b.solve(H)
    vc, pure, sol, Ls = classify(b, e, V, H2, CC)
    cI = sol.sum(); cII = pure.sum() - cI; cIII = (~pure).sum()
    meds.append(np.median(vc[~pure]))
    print(f"{N:>3} {len(e):>5} {cI:>4} {cII:>3} {cIII:>4} {pure.sum()/len(e):>5.2f} {vc[pure].max():>8.0e} "
          f"{vc[~pure].min():>7.1f} {np.median(vc[~pure]):>8.1f}")
    if N == 10:
        thr = {t: int((vc < t).sum()) for t in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)}
        print("     thresholds N=10:", thr)
        for i in np.where(pure & ~sol)[0]:
            print(f"     Class II: L={Ls[i]} <C2>={V[:,i]@b.C2@V[:,i]:.1f} E={e[i]:.4f}")
p = np.polyfit(np.log(Ns), np.log(meds), 1)[0]
print(f"median mixed variance exponent ~ N^{p:.2f}")

# ---- D. first-order critical point, N=10
section("D. first-order critical H(beta0=sqrt2), N=10 (h2=1 on P2(beta0))")
b = Block(10)
Hc = b.R(H_first_order(b.s, b.s2, math.sqrt(2), 1.0))
for ops, lab in (([b.L2, b.C2, b.nd], "L2>C2>nd"), ([b.L2, b.nd, b.C2], "L2>nd>C2")):
    e, V = b.solve(Hc, ops)
    vc, pure, sol, Ls = classify(b, e, V, 0.5, 0.0)
    vn, mn = var(b.nd, V)
    print(f"order {lab}: SU3-pure {pure.sum()} (tower {sol.sum()}, other {pure.sum()-sol.sum()}), "
          f"nd-pure {np.sum(vn<1e-6)}, nd-pure&not SU3 {np.sum((vn<1e-6)&~pure)}, mixed in both {np.sum((vn>=1e-6)&~pure)}")
    for i in np.where((vn < 1e-6) & ~pure)[0]:
        print(f"    U(5)-pure: E={e[i]:.4f} L={Ls[i]} nd={mn[i]:.2f}")
    for i in np.where(pure & ~sol)[0]:
        print(f"    SU3-pure non-tower: E={e[i]:.4f} L={Ls[i]} <C2>={V[:,i]@b.C2@V[:,i]:.1f}")
    i0 = [i for i in np.argsort(e) if Ls[i] == 0]
    print(f"    lowest L=0 states: E={[round(e[i],6) for i in i0[:3]]}, <nd>/N={[round(mn[i]/10,3) for i in i0[:3]]}")
    print(f"    max pure VarC2 {vc[pure].max():.1e}, min mixed VarC2 {vc[~pure].min():.1f}")
# overlap of deformed L=0 band head with |s^N>
e, V = b.solve(Hc, [b.L2, b.C2, b.nd])
zero_L0 = [i for i in range(len(e)) if abs(e[i]) < 1e-7 and b.Ls(V[:, [i]])[0] == 0]
print(f"    E=0, L=0 degeneracy: {len(zero_L0)}")
sN = np.zeros(len(e)); sN[np.argmin(np.diag(b.nd))] = 1.0
Vz = V[:, zero_L0]; g = Vz[:, np.argmax([var(b.C2, Vz[:, [k]])[0][0] < 1e-6 for k in range(Vz.shape[1])])]
print(f"    |<s^N|g0>| = {abs(sN @ g):.4f}; <nd>/N of g0 = {g@b.nd@g/10:.3f}")

# ---- E. second order
section("E. second-order critical point, N=10")
Hs2 = b.R(H_second_order(b.s)[0]); Co5b = b.R(C2_O5(b.s))
e, V = np.linalg.eigh(Hs2); V = nested_refine(e, V, [b.L2, Co5b, b.nd])
vo, _ = var(Co5b, V); vc, _ = var(b.C2, V)
print(f"max Var C[O5] = {vo.max():.1e}; SU3-pure count = {np.sum(vc<1e-6)}")

# ---- F. specificity table N=10
section("F. SU(3)-exact-label counts, N=10")
cases = {"SU(3) vertex": b.R(H_cq(b.s, 1.0, -math.sqrt(7) / 2)), "U(5) vertex": b.nd.copy(),
         "O(6) vertex": b.R(H_cq(b.s, 1.0, 0.0)), "stable PDS": b.R(build_H(b.s, b.s2, H0, H2, CC)),
         "1st-order crit": Hc, "2nd-order crit": Hs2}
for k, H in cases.items():
    e, V = b.solve(H)
    vc, _ = var(b.C2, V); Ls = b.Ls(V); _, mn = var(b.nd, V)
    pure = vc < 1e-6
    print(f"{k:16s}: {pure.sum():4d}   (L of pure states with nd~0 or N: {[ (int(Ls[i]), round(mn[i],2)) for i in np.where(pure)[0] if mn[i]<0.5 or mn[i]>9.5][:6]})")
rng = np.random.default_rng(3); minnt = []; nontriv = 0
for _ in range(12):
    xi = rng.uniform(0.05, 0.95); chi = rng.uniform(-math.sqrt(7) / 2, 0.0)
    e, V = b.solve(b.R(H_cq(b.s, xi, chi)))
    vc, _ = var(b.C2, V); _, mn = var(b.nd, V); Ls = b.Ls(V)
    triv = (mn < 0.5) | (mn > 9.5)
    nontriv += np.sum((vc < 1e-6) & ~triv)
    minnt.append(vc[~triv].min())
    trivpure = [(int(Ls[i]), round(mn[i], 2)) for i in np.where((vc < 1e-6) & triv)[0]]
print(f"12 random interior points: nontrivial exact-label = {nontriv}; smallest nontrivial Var = {min(minnt):.0f}; "
      f"trivial pure states at last point (L, nd) = {trivpure}")

# ---- G. consistent-Q line
section("G. consistent-Q line H(xi) = (1-xi) nd - xi/(4N) Q.Q, chi=-sqrt7/2, N=10")
for xi in np.round(np.arange(0.1, 0.91, 0.1), 2):
    e, V = b.solve(b.R(H_cq(b.s, xi, -math.sqrt(7) / 2)))
    vc, _ = var(b.C2, V); vn, mn = var(b.nd, V)
    triv = (mn < 0.5) | (mn > 9.5)
    ex = np.sum(((vc < 1e-6) | (vn < 1e-6)) & ~triv)
    g = np.argmin(e)
    print(f"xi={xi:.1f}: nontrivial exact-label (SU3 or U5) = {ex}; ground <nd>/N = {mn[g]/10:.3f}")
for xi in (0.45, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.55):
    e, V = np.linalg.eigh(b.R(H_cq(b.s, xi, -math.sqrt(7) / 2)))
    print(f"   xi={xi:.2f}: ground <nd>/N = {V[:,0]@b.nd@V[:,0]/10:.3f}")

# ---- H. approximate PDS
section("H. approximate PDS, N=10")
Hst = b.R(build_H(b.s, b.s2, H0, H2, CC)); e0, V0 = b.solve(Hst)
vc0, _ = var(b.C2, V0); gb = V0[:, vc0 < 1e-6]
vmixmin = vc0[vc0 >= 1e-6].min()
Co5b = b.R(C2_O5(b.s))
w, U = np.linalg.eigh(b.C2); wr = np.round(w).astype(int)
def metrics(v):
    c = U.T @ v; P = np.array([np.sum(c[wr == val] ** 2) for val in np.unique(wr)]); P = P[P > 1e-16]
    vv = float(v @ b.C2 @ b.C2 @ v - (v @ b.C2 @ v) ** 2)
    return vv, float(-np.sum(P * np.log(P))), float(1 - np.sum(P ** 2))
for name, Vb, grid, fitmax in (("eps*nd", b.nd, [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], 0.05),
                               ("eps*C[O5]", Co5b, [0.002, 0.005, 0.01, 0.02, 0.05], 0.02)):
    rows = []
    for eps in grid:
        ee, Ve = np.linalg.eigh(Hst + eps * Vb)
        tr = [Ve[:, int(np.argmax(np.abs(Ve.T @ gb[:, c])))] for c in range(gb.shape[1])]
        m = [metrics(v) for v in tr]
        rows.append((eps, np.mean([x[0] for x in m]), np.max([x[0] for x in m]),
                     np.mean([x[1] for x in m]), np.mean([x[2] for x in m])))
    arr = np.array(rows); sel = arr[:, 0] <= fitmax
    slope = np.polyfit(np.log(arr[sel, 0]), np.log(arr[sel, 1]), 1)[0]
    print(f"{name}: exponent (eps<={fitmax}) = {slope:.2f}; min mixed at eps=0: {vmixmin:.1f}")
    for r in rows:
        print(f"   eps={r[0]:.3f} meanVar={r[1]:.2e} maxVar={r[2]:.2e} meanS={r[3]:.2e} mean(1-P)={r[4]:.2e}")
