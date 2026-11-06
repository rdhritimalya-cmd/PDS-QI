"""
measurement_budget.py -- shot-noise budget for reading Var C2[SU(3)] on the 4-qubit N=3 register
(Sec. VII D of the manuscript).

The operators C = C2[SU(3)] and C^2 are decomposed into Pauli strings and grouped into qubit-wise
commuting sets, one measurement setting per set. For a state psi and S shots per setting, an
operator O = sum_g O_g estimated group by group has shot variance
    sigma^2(O) = sum_g Var_psi(O_g) / S,
since all strings in a group are read from the same shots.

Two estimators of Var C are compared, to first order in the shot noise:
 (A) separate: C and C^2 measured with their own groupings, Var = <C^2> - <C>^2,
        sigma_A^2 = sigma^2(C^2) + 4 <C>^2 sigma^2(C);
 (B) centred: (C - c)^2 and C measured, Var = <(C-c)^2> - (<C> - c)^2, with c = <C>,
        sigma_B^2 = sigma^2((C - c)^2)       [the (<C>-c)^2 term has zero first-order noise].
     Measuring C^2 and C on the same shots and keeping their covariance gives the same first-order
     error, so (B) is also the covariance-aware version of (A).
For each estimator the script prints the number of settings, the error at S = 4000 shots per setting,
and the total number of shots needed for an error of 10 (enough to separate a pure state from the
smallest mixed value at this N, 52, by more than five standard deviations).
"""
import math
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator
from ibm_pds import IBMSpace
from critical import H_first_order

N = 3
S_REF = 4000
TARGET = 10.0


def states_and_ops():
    sp, sp2 = IBMSpace(N), IBMSpace(N - 2)
    m0 = np.where(np.abs(sp.Mvals) < 1e-9)[0]
    ix = np.ix_(m0, m0)
    H = H_first_order(sp, sp2, math.sqrt(2.0)).toarray().real[ix]
    C2 = sp.C2_SU3().toarray().real[ix]; L2 = sp.L2().toarray().real[ix]; nd = sp.n_d().toarray().real[ix]
    H = 0.5 * (H + H.T); C2 = 0.5 * (C2 + C2.T)
    e, V = np.linalg.eigh(H)
    i = 0
    while i < len(e):                          # same whole-window refinement as the paper (Sec. VII B)
        j = i
        while j + 1 < len(e) and e[j + 1] - e[i] < 1e-7:
            j += 1
        if j > i:
            sub = V[:, i:j + 1]
            for op in (L2, C2, nd):
                _, u = np.linalg.eigh(sub.T @ op @ sub); sub = sub @ u
            V[:, i:j + 1] = sub
        i = j + 1
    l2 = np.einsum('ij,ij->j', V, L2 @ V)
    Ls = np.round((-1 + np.sqrt(1 + 4 * l2)) / 2).astype(int)
    d = len(e); D = 2 ** int(np.ceil(np.log2(d)))
    C2e = np.zeros((D, D)); C2e[:d, :d] = C2
    psis = []
    for k in range(d):
        w = np.zeros(D); w[:d] = V[:, k]
        psis.append(w)
    return e, Ls, C2e, psis


def group_sigma2(op, psi, groups):
    """sum over groups of Var_psi(O_g)  (per-shot variance summed over settings)."""
    tot = 0.0
    for g in groups:
        M = g.to_matrix()
        Mv = M @ psi
        m = np.real(np.vdot(psi, Mv))
        tot += np.real(np.vdot(Mv, Mv)) - m * m
    return max(tot, 0.0)


def main():
    e, Ls, C2e, psis = states_and_ops()
    opC = SparsePauliOp.from_operator(Operator(C2e)).simplify()
    opC2 = SparsePauliOp.from_operator(Operator(C2e @ C2e)).simplify()
    # drop the identity term from grouping: it carries no shot noise
    def groups_of(op):
        labels = [p.to_label() for p in op.paulis]
        keep = [k for k, l in enumerate(labels) if set(l) != {"I"}]
        sub = SparsePauliOp([op.paulis[k] for k in keep], op.coeffs[keep])
        return sub.group_commuting(qubit_wise=True), len(op.paulis)
    gC, nC = groups_of(opC)
    gC2, nC2 = groups_of(opC2)
    print(f"N={N}: C has {nC} Pauli terms in {len(gC)} qubit-wise commuting groups;"
          f" C^2 has {nC2} terms in {len(gC2)} groups")
    print(f"{'state':>5} {'E/h2':>6} {'Var C':>8} {'<C>':>7} | {'A: sets':>7} {'err@4000':>8} {'shots(err=10)':>13}"
          f" | {'B: sets':>7} {'err@4000':>8} {'shots(err=10)':>13}")
    rows = []
    for k, psi in enumerate(psis):
        mC = float(psi @ C2e @ psi); vC = float(psi @ C2e @ C2e @ psi) - mC ** 2
        # (A) separate
        sA2 = group_sigma2(opC2, psi, gC2) + 4 * mC ** 2 * group_sigma2(opC, psi, gC)
        setsA = len(gC) + len(gC2)
        # (B) centred at c = <C>
        D = C2e.shape[0]
        Oc = (C2e - mC * np.eye(D)) @ (C2e - mC * np.eye(D))
        opOc = SparsePauliOp.from_operator(Operator(Oc)).simplify()
        gOc, _ = groups_of(opOc)
        sB2 = group_sigma2(opOc, psi, gOc)
        setsB = len(gOc) + len(gC)            # C itself is still measured to obtain <C>
        errA = math.sqrt(sA2 / S_REF); errB = math.sqrt(sB2 / S_REF)
        shotsA = setsA * sA2 / TARGET ** 2; shotsB = setsB * sB2 / TARGET ** 2
        rows.append((k, e[k], vC, mC, setsA, errA, shotsA, setsB, errB, shotsB))
        print(f"{k:>5} {e[k]:>6.2f} {vC:>8.2f} {mC:>7.2f} | {setsA:>7} {errA:>8.2f} {shotsA:>13.3g}"
              f" | {setsB:>7} {errB:>8.2f} {shotsB:>13.3g}")
    r = np.array([x[1:] for x in rows])
    pure = r[:, 1] < 1e-6
    print(f"pure states:  median error at 4000 shots/setting  A {np.median(r[pure,4]):.1f}  B {np.median(r[pure,7]):.1f};"
          f"  median total shots for error 10  A {np.median(r[pure,5]):.3g}  B {np.median(r[pure,8]):.3g}")
    print("LaTeX rows: E/h2 & L & Var & <C> & err_A & err_B & shots_A & shots_B")
    for (k, E, v, m, sa, ea, na, sb, eb, nb) in rows:
        print(f"  {E:.0f} & {Ls[k]} & {max(v,0):.0f} & {m:.1f} & {ea:.1f} & {eb:.1f} & {na:.1e} & {nb:.1e} \\\\")
    print(f"mixed states: median error at 4000 shots/setting  A {np.median(r[~pure,4]):.1f}  B {np.median(r[~pure,7]):.1f};"
          f"  median total shots for error 10  A {np.median(r[~pure,5]):.3g}  B {np.median(r[~pure,8]):.3g}")


if __name__ == "__main__":
    main()
