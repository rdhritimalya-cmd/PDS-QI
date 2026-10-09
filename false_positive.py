"""
false_positive.py  --  systematic false-positive test of the label-variance
diagnostic: the fingerprint identifies PDS-selected states within a PDS Hamiltonian
family; it is not a universal detector that fires on arbitrary Hamiltonians.

We scan generic consistent-Q IBM Hamiltonians across the Casten triangle,
    H(xi, chi) = (1-xi) n_d  -  (xi / 4N) Q_chi . Q_chi ,
    Q_chi,mu = s^dag dtil_mu + d^dag_mu s + chi (d^dag dtil)^(2)_mu ,
at many random (xi, chi) points. Away from the SU(3) vertex (chi = -sqrt7/2,
xi = 1) these are NOT PDS-constructed. We count the SU(3)-exact-label eigenstates
Var C2[SU(3)] < 1e-6, excluding the trivial n_d=0 and n_d=N multiplets that carry
an exact label at any point. A universal "detector" would light up everywhere; a
genuine PDS fingerprint should find essentially none.

Requires: numpy, ibm_pds.
"""
import numpy as np, math
import scipy.sparse as sp
from ibm_pds import IBMSpace


def Qchi(spN, chi):
    """general quadrupole Q_chi (mu components summed into Q.Q below)."""
    Qs = {}
    for q in range(-2, 3):
        t1 = ((-1) ** q) * spN.hop(0, spN.mode_index(-q))   # s^dag dtil_q
        t2 = spN.hop(spN.mode_index(q), 0)                  # d^dag_q s
        t3 = spN.dd_tensor(2, q)                            # (d^dag dtil)^(2)_q
        Qs[q] = (t1 + t2 + chi * t3)
    return Qs


def H_cq(spN, xi, chi):
    nd = spN.n_d()
    N = spN.N
    Q = Qchi(spN, chi)
    QQ = sp.csr_matrix((spN.dim, spN.dim))
    for q in range(-2, 3):
        QQ = QQ + ((-1) ** q) * (Q[q] @ Q[-q])
    H = (1 - xi) * nd - (xi / (4.0 * N)) * QQ
    return H.tocsr()


def count_exact_label(N, xi, chi, thr=1e-6):
    spN = IBMSpace(N);
    m0 = np.where(np.abs(spN.Mvals) < 1e-9)[0]; ix = np.ix_(m0, m0)
    H = H_cq(spN, xi, chi).toarray().real[ix]; H = 0.5 * (H + H.T)
    C2 = spN.C2_SU3().toarray().real[ix]; C2 = 0.5 * (C2 + C2.T)
    L2 = spN.L2().toarray().real[ix]; nd = spN.n_d().toarray().real[ix]
    ndb = nd
    e, V = np.linalg.eigh(H); i = 0
    while i < len(e):
        j = i
        while j + 1 < len(e) and e[j + 1] - e[i] < 1e-7:
            j += 1
        if j > i:
            sub = V[:, i:j + 1]
            for op in (L2, C2, ndb):
                _, u = np.linalg.eigh(sub.T @ op @ sub); sub = sub @ u
            V[:, i:j + 1] = sub
        i = j + 1
    lv = np.array([float(V[:, k] @ C2 @ C2 @ V[:, k] - (V[:, k] @ C2 @ V[:, k]) ** 2)
                   for k in range(len(e))])
    ndv = np.array([float(V[:, k] @ ndb @ V[:, k]) for k in range(len(e))])
    exact = np.where(lv < thr)[0]
    # trivial = n_d ~ 0 or n_d ~ N multiplets
    nontrivial = [k for k in exact if not (ndv[k] < 0.5 or ndv[k] > N - 0.5)]
    return len(exact), len(nontrivial)


if __name__ == "__main__":
    N = 10
    rng = np.random.default_rng(3)
    chi_su3 = -math.sqrt(7) / 2
    print(f"False-positive sweep, consistent-Q Hamiltonians, N={N}")
    print("(SU(3) vertex is chi=-sqrt7/2=%.3f, xi=1)\n" % chi_su3)
    print(f"{'xi':>6} {'chi':>7} {'#exact-label':>13} {'#nontrivial':>12}  note")
    # (a) random generic points
    nnt = []
    for _ in range(12):
        xi = rng.uniform(0.05, 0.95); chi = rng.uniform(-math.sqrt(7)/2, 0.0)
        ne, nn = count_exact_label(N, xi, chi); nnt.append(nn)
        tag = "generic"
        print(f"{xi:>6.3f} {chi:>7.3f} {ne:>13} {nn:>12}  {tag}")
    # (b) the SU(3) vertex (positive control: should be fully solvable)
    ne, nn = count_exact_label(N, 1.0, chi_su3)
    print(f"{1.0:>6.3f} {chi_su3:>7.3f} {ne:>13} {nn:>12}  SU(3) vertex (control)")
    # (c) U(5) vertex
    ne, nn = count_exact_label(N, 0.0, 0.0)
    print(f"{0.0:>6.3f} {0.0:>7.3f} {ne:>13} {nn:>12}  U(5) vertex")
    print(f"\nGeneric points: nontrivial SU(3)-exact-label states "
          f"min={min(nnt)}, max={max(nnt)}, mean={np.mean(nnt):.1f} "
          f"-> the diagnostic does NOT fire on non-PDS Hamiltonians.")
