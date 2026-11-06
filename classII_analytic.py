"""
classII_analytic.py -- the SU(3)-pure states outside the closed-form tower (Class II, Sec. IV B).

Checks, for the SU(3)-PDS Hamiltonian H = h0 P0^dag P0 + h2 P2^dag.P2 + C L^2:

 (1) Counting (Elliott's rule). For every N from 5 to 80 there are exactly two angular momenta,
     L = 2N-4 and L = 2N-7, at which the number of states in the symmetric U(6) irrep [N] exceeds
     the number of closed-form tower states by one (at N = 4 only L = 4).
 (2) Construction. With chi the unique state of N-2 bosons with that L (it has n_d = N-2),
     the Class II eigenstate is psi = P0^dag chi / ||P0^dag chi||, and ||P0^dag chi||^2 = 2(2N+5).
 (3) Energies. E = (h0-h2) 2(2N+5) + h2 [2N(2N+3) - f2(lam,mu)] + C L(L+1), with
     (lam,mu) = (2N-4,2) for L = 2N-4 and (2N-8,4) for L = 2N-7, i.e.
        E(2N-4) = 2 h0 (2N+5) + 8 h2 (N-2)     + C L(L+1),
        E(2N-7) = 2 h0 (2N+5) + 2 h2 (10N-23)  + C L(L+1).
     Checked for N = 4..16 at several parameter sets.
 (4) Bands. The Class II states terminate two mixed bands (the K=0_2 band and the lowest odd-L
     non-tower band); Var C2[SU(3)] along each band is printed for N = 10 (stable point) and for
     the 168Er fit (N = 16), and saved to classII_bands.json for fig_classII.png. The decrease
     along each band is checked to be monotonic for N = 5..16 at three parameter sets.
 (5) Completeness. The number of pure states equals (N^2+N+2)/2 (tower, M = 0 block) + 2 for every
     N = 5..16.
 (6) Wider family. psi_II stays an eigenstate of Leviatan's two- and three-body SU(3)-PDS family
        H = h1 Lam^dag Lam + h2 theta2 + C L^2 + h3 P0^dag P0 + h4 P0^dag n_s P0
            + h5 (Lam^dag s P0 + h.c.),   Lam^dag = P0^dag s^dag + P2^dag . d^dag
     (Leviatan, Eur. Phys. J. Spec. Top. 229, 2405 (2020), Eq. (34)), with the energy of (3) and
     h0 -> h2 + h3, for random coefficients at N = 6, 8, 10, 12.
 (7) First-order critical point: at N = 10 the Class II states of H(beta0 = sqrt 2) are the same
     psi_II, with E = half the formula of (3) at h0 = 0, h2 = 1, C = 0.
"""
import math
import scipy.sparse as sp
from critical import H_first_order
import json
import numpy as np
from revision_common import Block, classify, H0, H2, CC
from ibm_pds import build_H, f2


def Lcontent(l, m):
    """Elliott's rule: multiplicity of each L in the SU(3) irrep (l, m)."""
    out = {}
    for K in range(min(l, m), -1, -2):
        Ls = range(max(l, m), -1, -2) if K == 0 else range(K, K + max(l, m) + 1)
        for L in Ls:
            out[L] = out.get(L, 0) + 1
    return out


def irreps(N):
    """SU(3) irreps in the symmetric U(6) irrep [N]: (2N-4k-6m, 2k)."""
    return [(2 * N - 4 * k - 6 * m, 2 * k) for k in range(N + 1) for m in range(N + 1)
            if 2 * N - 4 * k - 6 * m >= 0]


def n_tower(N, L):
    return sum(1 for k in range(N // 2 + 1)
               if (k == 0 and L % 2 == 0 and L <= 2 * N) or (k > 0 and 2 * k <= L <= 2 * N - 2 * k))


def counting(N):
    tot = {}
    for l, m in irreps(N):
        for L, c in Lcontent(l, m).items():
            tot[L] = tot.get(L, 0) + c
    return [L for L in sorted(tot) if tot[L] == n_tower(N, L) + 1]


def E_formula(N, L, h0, h2, C):
    lm = (2 * N - 4, 2) if L == 2 * N - 4 else (2 * N - 8, 4)
    return (h0 - h2) * 2 * (2 * N + 5) + h2 * (2 * N * (2 * N + 3) - f2(*lm)) + C * L * (L + 1)


def _create(spN, spNm1, i):
    rows, cols, vals = [], [], []
    for col, st in enumerate(spNm1.states):
        lst = list(st); amp = math.sqrt(lst[i] + 1); lst[i] += 1
        rows.append(spN.index[tuple(lst)]); cols.append(col); vals.append(amp)
    return sp.csr_matrix((vals, (rows, cols)), shape=(spN.dim, spNm1.dim))


def wider_family():
    from ibm_pds import IBMSpace
    rng = np.random.default_rng(3)
    ok = True
    for N in (6, 8, 10, 12):
        s = {n: IBMSpace(n) for n in (N, N - 2, N - 3)}
        P0d, P2d = s[N].pair_ops(s[N - 2])
        sd = _create(s[N - 2], s[N - 3], 0)
        Ld = P0d @ sd
        for m in range(-2, 3):
            Ld = Ld + ((-1) ** m) * (P2d[m] @ _create(s[N - 2], s[N - 3], s[N].mode_index(-m)))
        P0 = P0d.T
        theta2 = P0d @ P0 + sum(P2d[m] @ P2d[m].T for m in range(-2, 3))
        idchk = abs(theta2 + s[N].C2_SU3() - 2 * N * (2 * N + 3) * sp.identity(s[N].dim)).max()
        h1, h2, C, h3, h4, h5 = rng.uniform(0.2, 1.5, 6)
        LsP0 = Ld @ sd.T @ P0
        Hx = (h1 * (Ld @ Ld.T) + h2 * theta2 + C * s[N].L2() + h3 * (P0d @ P0)
              + h4 * (P0d @ s[N - 2].n_s() @ P0) + h5 * (LsP0 + LsP0.T))
        b = Block(N); Hm = b.R(Hx)
        e, V = b.solve(b.R(build_H(b.s, b.s2, H0, H2, CC)))
        vc, pure, sol, Ls = classify(b, e, V, H2, CC)
        worst_r, worst_e = 0.0, 0.0
        for i in np.where(pure & ~sol)[0]:
            psi = V[:, i]; Hp = Hm @ psi; E = psi @ Hp; L = int(Ls[i])
            worst_r = max(worst_r, np.linalg.norm(Hp - E * psi))
            worst_e = max(worst_e, abs(E - E_formula(N, L, h2 + h3, h2, C)))
        print(f"    N={N}: identity theta2 = 2N(2N+3) - C2 to {idchk:.1e};"
              f" max ||(H - E) psi_II|| = {worst_r:.1e}, max |E - formula| = {worst_e:.1e}")
        ok &= idchk < 1e-9 and worst_r < 1e-9 and worst_e < 1e-9
    return ok


def main():
    ok = True
    print("(1) counting: L values with one state beyond the tower")
    bad = [(N, counting(N)) for N in range(5, 81) if counting(N) != [2 * N - 7, 2 * N - 4]]
    print(f"    N = 4: {counting(4)};  N = 5..80: always [2N-7, 2N-4]; exceptions: {bad}")
    ok &= counting(4) == [4] and not bad

    print("(2,3) construction psi = P0^dag chi and closed-form energies")
    pars = [(H0, H2, CC), (1.1, 0.6, 0.013), (0.05, 1.7, 0.2), (0.008, 0.004, 0.013)]
    worst_ov, worst_E, worst_n = 0.0, 0.0, 0.0
    for N in range(4, 17):
        b, bm = Block(N), Block(N - 2)
        P0d = b.s.pair_ops(b.s2)[0].tocsr()[b.sel][:, bm.sel].toarray()
        wl, ul = np.linalg.eigh(bm.L2)
        Lm = np.round((-1 + np.sqrt(1 + 4 * wl)) / 2).astype(int)
        for h0, h2, C in pars:
            H = b.R(build_H(b.s, b.s2, h0, h2, C))
            e, V = b.solve(H)
            vc, pure, sol, Ls = classify(b, e, V, h2, C)
            for i in np.where(pure & ~sol)[0]:
                L = int(Ls[i])
                idx = np.where(Lm == L)[0]
                assert len(idx) == 1, (N, L)
                chi = ul[:, idx[0]]
                phi = P0d @ chi
                n2 = phi @ phi
                worst_n = max(worst_n, abs(n2 - 2 * (2 * N + 5)))
                worst_ov = max(worst_ov, 1 - abs(phi @ V[:, i]) / np.sqrt(n2))
                worst_E = max(worst_E, abs(e[i] - E_formula(N, L, h0, h2, C)))
        print(f"    N={N:2d}: Class II L = {sorted(set(int(x) for x in Ls[pure & ~sol]))}", flush=True)
    print(f"    max |1 - overlap| = {worst_ov:.1e},  max | ||P0dag chi||^2 - 2(2N+5) | = {worst_n:.1e},"
          f"  max |E - formula| = {worst_E:.1e}")
    ok &= worst_ov < 1e-10 and worst_n < 1e-9 and worst_E < 1e-9

    print("(4) Var C2 along the bands that the Class II states terminate")
    bands = {}
    for tag, N, (h0, h2, C) in (("stable_N10", 10, (H0, H2, CC)), ("er168_N16", 16, (0.008, 0.004, 0.013))):
        b = Block(N)
        H = b.R(build_H(b.s, b.s2, h0, h2, C))
        e, V = b.solve(H)
        vc, pure, sol, Ls = classify(b, e, V, h2, C)
        out = {}
        for name, Lrange in (("K0_2", range(0, 2 * N - 3, 2)), ("odd", range(3, 2 * N - 6, 2))):
            rows = []
            for L in Lrange:
                cand = [i for i in np.where(Ls == L)[0] if not sol[i]]
                i = min(cand, key=lambda j: e[j])          # lowest non-tower state at this L
                rows.append((L, float(e[i]), float(max(vc[i], 0.0))))
            out[name] = rows
            print(f"    {tag} {name}: " + ", ".join(f"L={L}:{v:.1f}" for L, _, v in rows))
            ok &= rows[-1][2] < 1e-8 and all(rows[k][2] > rows[k + 1][2] for k in range(len(rows) - 1))
        bands[tag] = out
    json.dump(bands, open("classII_bands.json", "w"), indent=1)

    print("(4b) monotonic decrease along both bands, N = 5..16")
    mono_ok = True
    for pars in [(H0, H2, CC), (0.008, 0.004, 0.013), (1.1, 0.6, 0.013)]:
        for N in range(5, 17):
            b = Block(N); h0, h2, C = pars
            e, V = b.solve(b.R(build_H(b.s, b.s2, h0, h2, C)))
            vc, pure, sol, Ls = classify(b, e, V, h2, C)
            for Lr in (range(0, 2 * N - 3, 2), range(3, 2 * N - 6, 2)):
                v = []
                for L in Lr:
                    cand = [i for i in np.where(Ls == L)[0] if not sol[i]]
                    v.append(max(vc[min(cand, key=lambda j: e[j])], 0.0))
                mono_ok &= all(v[k] > v[k + 1] for k in range(len(v) - 1)) and v[-1] < 1e-8
    print(f"    monotonic and ending at zero in all cases: {mono_ok}")
    ok &= mono_ok

    print("(5) completeness: pure count = (N^2+N+2)/2 + 2")
    comp_ok = True
    for N in range(5, 17):
        b = Block(N)
        e, V = b.solve(b.R(build_H(b.s, b.s2, H0, H2, CC)))
        vc, pure, sol, Ls = classify(b, e, V, H2, CC)
        comp_ok &= int(sol.sum()) == (N * N + N + 2) // 2 and int(pure.sum()) == (N * N + N + 6) // 2
    print(f"    holds for N = 5..16: {comp_ok}")
    ok &= comp_ok

    print("(6) wider two- and three-body PDS family")
    ok &= wider_family()

    print("(7) first-order critical point, N = 10")
    b, bm = Block(10), Block(8)
    H = b.R(H_first_order(b.s, b.s2, math.sqrt(2.0)))
    e, V = np.linalg.eigh(H)
    P0d = b.s.pair_ops(b.s2)[0].tocsr()[b.sel][:, bm.sel].toarray()
    wl, ul = np.linalg.eigh(bm.L2)
    Lm = np.round((-1 + np.sqrt(1 + 4 * wl)) / 2).astype(int)
    for L in (16, 13):
        phi = P0d @ ul[:, np.where(Lm == L)[0][0]]; phi /= np.linalg.norm(phi)
        ov = np.abs(V.T @ phi); i = int(np.argmax(ov))
        Ep = 0.5 * E_formula(10, L, 0.0, 1.0, 0.0)
        print(f"    L={L}: overlap {ov[i]:.12f}, E = {e[i]:.6f}, predicted {Ep:.6f}")
        ok &= abs(ov[i] - 1) < 1e-10 and abs(e[i] - Ep) < 1e-9
    print("all Class II checks passed" if ok else "CHECK FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
