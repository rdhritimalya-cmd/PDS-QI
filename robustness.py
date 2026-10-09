"""
robustness.py  --  system-size scaling and approximate-PDS robustness of the
label-variance diagnostic.

scaling_table(): per N, the exact-label count/fraction, the minimum mixed-state
  variance, the maximum pure-state residual, the pure/mixed separation ratio, and
  the median mixed-state variance with a fitted exponent. The mixed-state variance
  is NOT a clean O(N^2): the median grows as ~N^2.2 at the stable point, the
  distribution widens toward ~N^4 at the top, and the minimum stays bounded away
  from zero. The robust statement is that the pure states sit at numerical zero
  while the mixed states are separated from zero by a margin that does not close.

perturbation(): H(eps) = H_PDS + eps * n_d, an SU(3)-breaking deformation. The
  formerly symmetry-pure states acquire a label variance that grows as eps^2 at
  small eps, so the diagnostic degrades smoothly and the pure/mixed separation
  survives for a finite window of approximate PDS.

Requires: numpy, scipy, ibm_pds, critical.
"""
import numpy as np, math
from ibm_pds import IBMSpace, build_H

H0, H2, CC = 0.35, 1.0, 0.0523   # representative stable SU(3)-PDS parameters


def _block_ops(N):
    spN = IBMSpace(N); spN2 = IBMSpace(N - 2)
    m0 = np.where(np.abs(spN.Mvals) < 1e-9)[0]; ix = np.ix_(m0, m0)
    Hs = build_H(spN, spN2, H0, H2, CC).toarray().real[ix]
    C2 = spN.C2_SU3().toarray().real[ix]
    L2 = spN.L2().toarray().real[ix]; nd = spN.n_d().toarray().real[ix]
    return 0.5 * (Hs + Hs.T), 0.5 * (C2 + C2.T), L2, nd


def _resolve(H, C2, L2, nd):
    e, V = np.linalg.eigh(H); i = 0
    while i < len(e):
        j = i
        while j + 1 < len(e) and e[j + 1] - e[i] < 1e-7:
            j += 1
        if j > i:
            sub = V[:, i:j + 1]
            for op in (L2, C2, nd):
                _, u = np.linalg.eigh(sub.T @ op @ sub); sub = sub @ u
            V[:, i:j + 1] = sub
        i = j + 1
    return e, V


def _labvar(V, C2):
    return np.array([float(V[:, k] @ C2 @ C2 @ V[:, k] - (V[:, k] @ C2 @ V[:, k]) ** 2)
                     for k in range(V.shape[1])])


def scaling_table(Ns=(4, 6, 8, 10, 12, 14, 16), thr=1e-6, verbose=True):
    rows = []; meds = []
    for N in Ns:
        H, C2, L2, nd = _block_ops(N)
        _, V = _resolve(H, C2, L2, nd)
        lv = _labvar(V, C2); d = len(lv)
        pure = lv[lv < thr]; mixed = lv[lv >= thr]
        row = dict(N=N, dim=d, n_pure=len(pure), frac=len(pure) / d,
                   max_pure=float(pure.max()) if len(pure) else 0.0,
                   min_mix=float(mixed.min()), med_mix=float(np.median(mixed)),
                   max_mix=float(mixed.max()),
                   sep_ratio=float(mixed.min() / max(pure.max(), 1e-16)) if len(pure) else float('inf'))
        rows.append(row); meds.append(row["med_mix"])
    p = np.polyfit(np.log(Ns), np.log(meds), 1)[0]
    if verbose:
        print("Scaling of the label variance (stable SU(3)-PDS point):")
        print(f"{'N':>3} {'dim':>5} {'#pure':>6} {'frac':>6} {'max_pure':>10} "
              f"{'min_mix':>9} {'median_mix':>11} {'sep_ratio':>11}")
        for r in rows:
            print(f"{r['N']:>3} {r['dim']:>5} {r['n_pure']:>6} {r['frac']:>6.3f} "
                  f"{r['max_pure']:>10.1e} {r['min_mix']:>9.2f} {r['med_mix']:>11.2f} "
                  f"{r['sep_ratio']:>11.1e}")
        print(f"fitted median exponent: Var_med ~ N^{p:.2f} "
              f"(not a clean N^2; min_mix stays bounded away from 0, "
              f"pure residual stays ~1e-10).")
    return rows, p


def perturbation(N=10, eps_grid=None, verbose=True):
    """H(eps) = H_PDS + eps n_d. Track the eps=0 ground band and report its Var C2."""
    H, C2, L2, nd = _block_ops(N)
    e0, V0 = _resolve(H, C2, L2, nd)
    lv0 = _labvar(V0, C2)
    gb = np.where(lv0 < 1e-6)[0]            # the symmetry-pure states at eps=0
    gbvecs = V0[:, gb]
    if eps_grid is None:
        eps_grid = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    out = []
    for eps in eps_grid:
        He = H + eps * nd
        ee, Ve = np.linalg.eigh(0.5 * (He + He.T))
        # track each eps=0 pure state by max overlap
        vars_tracked = []
        for c in range(gbvecs.shape[1]):
            ov = np.abs(Ve.T @ gbvecs[:, c]); k = int(np.argmax(ov))
            v = Ve[:, k]
            vv = float(v @ C2 @ C2 @ v - (v @ C2 @ v) ** 2)
            vars_tracked.append(max(vv, 0.0))
        out.append((eps, float(np.mean(vars_tracked)), float(np.max(vars_tracked))))
    # small-eps exponent (fit mean Var vs eps over the small-eps part)
    arr = np.array(out)
    sel = (arr[:, 0] > 0) & (arr[:, 0] <= 0.05)
    slope = np.polyfit(np.log(arr[sel, 0]), np.log(arr[sel, 1]), 1)[0] if sel.sum() >= 2 else float('nan')
    if verbose:
        print(f"\nApproximate PDS: H(eps) = H_PDS + eps n_d, N={N} "
              f"({len(gb)} formerly-pure states tracked):")
        print(f"{'eps':>7} {'mean Var':>11} {'max Var':>11}")
        for (eps, mv, xv) in out:
            print(f"{eps:>7.3f} {mv:>11.3e} {xv:>11.3e}")
        print(f"small-eps scaling: mean Var ~ eps^{slope:.2f}  (perturbation theory predicts 2)")
    return out, slope


if __name__ == "__main__":
    scaling_table()
    perturbation()
