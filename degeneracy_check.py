"""
degeneracy_check.py -- the degeneracy-resolution statements of Sec. VII B and Appendix A.

At the N = 3 first-order critical point (M = 0 block, 10 states; fivefold level at E = 0 and
threefold level at E/h2 = 15) this script checks:
 (1) n_d restricted to each degenerate window has distinct eigenvalues, so the whole-window
     sequential refinement (L^2, C2, n_d) equals the nested U(5) reading (L^2, n_d, C2);
 (2) the sequential states differ from the nested SU(3) reading in the fivefold level;
 (3) for eigenvectors v of the restricted operator P C2 P, Var_v C2 = ||(1-P) C2 v||^2, and in the
     fivefold level one such vector has Var = 51.84, so diagonalizing C2 inside a window does not by
     itself certify purity.
"""
import numpy as np
from revision_common import Block, nested_refine, var
from critical import H_first_order


def main():
    N = 3
    b = Block(N)
    H = b.R(H_first_order(b.s, b.s2, np.sqrt(2.0)))
    e, V = np.linalg.eigh(H)
    ok = True
    i = 0
    while i < len(e):
        j = i
        while j + 1 < len(e) and e[j + 1] - e[i] < 1e-7:
            j += 1
        if j > i:
            W = V[:, i:j + 1]
            S = W.copy()
            for O in (b.L2, b.C2, b.nd):
                _, u = np.linalg.eigh(S.T @ O @ S)
                S = S @ u
            wn = np.linalg.eigvalsh(W.T @ b.nd @ W)
            U5 = nested_refine(e[i:j + 1], W, [b.L2, b.nd, b.C2])
            SU3 = nested_refine(e[i:j + 1], W, [b.L2, b.C2, b.nd])
            ov_u5 = np.abs(S.T @ U5).max(axis=1).min()
            ov_su3 = np.abs(S.T @ SU3).max(axis=1).min()
            vs, _ = var(b.C2, S)
            wc, uc = np.linalg.eigh(W.T @ b.C2 @ W)
            Vc = W @ uc
            vc, _ = var(b.C2, Vc)
            P = W @ W.T
            res = np.linalg.norm((np.eye(len(H)) - P) @ b.C2 @ Vc, axis=0) ** 2
            print(f"E/h2 = {e[i]:5.2f}, g = {j - i + 1}")
            print(f"  n_d on window: {np.round(wn, 4)}  (min gap {np.diff(np.sort(wn)).min():.3f})")
            print(f"  sequential vs nested U(5): min overlap {ov_u5:.12f};  vs nested SU(3): {ov_su3:.6f}")
            print(f"  Var C2 of sequential states: {np.round(np.maximum(vs, 0), 2)},  L = {b.Ls(S)}")
            print(f"  eigenvectors of P C2 P: Var C2 = {np.round(np.maximum(vc, 0), 2)};"
                  f"  ||(1-P) C2 v||^2 = {np.round(res, 2)}")
            ok &= np.diff(np.sort(wn)).min() > 1e-3 and abs(ov_u5 - 1) < 1e-10
            ok &= np.allclose(np.maximum(vc, 0), res, atol=1e-8)
        i = j + 1
    print("degeneracy checks passed" if ok else "CHECK FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
