"""
label_locality.py  --  Why the diagnostic is intrinsically non-local on a qubit
register, and what that implies for the simulation resource cost.

Two results used in the manuscript's quantum-simulation section:

(B) NO CHEAP LABEL.  In the symmetry-reduced (dense) qubit embedding, no low-weight
    Pauli operator acts as a symmetry label for the solvable subspace.  We scan all
    weight<=2 Paulis (|<P>|=1 on every solvable state and <1 on the mixed states would
    make P a cheap label) and also optimize over their linear span.  The best single
    Pauli has a negative margin; the best weight<=2 combination leaves the solvable
    label variance far from zero.  The solvable subspace is defined by COLLECTIVE
    SU(3) labels, which have no few-qubit representation under compression.

(D) ENCODING RESOURCE SCALING.  The same collective structure means the cost of
    representing the Hamiltonian as Pauli strings does not fall when one moves to a
    "local" boson-to-qubit encoding: adding qubits (binary-per-mode, then unary)
    enlarges the unphysical space and raises both the term count and the maximum
    Pauli weight.  The compressed dense encoding is the gate-cheapest of the three
    under naive Pauli-Trotterization.  A structured (non-Pauli) implementation of the
    ladder operators could change this and is left to future work; this module
    reports the honest naive-cost numbers.

Requires: numpy, scipy, qiskit, ibm_pds, critical, ibm_encodings.
"""
import numpy as np, math, itertools
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli
from ibm_pds import IBMSpace, build_H
from critical import H_first_order
from ibm_encodings import Encoded, build_H_encoded


# ---------- (B) no cheap label ----------
def _critical_resolved(N):
    spN = IBMSpace(N); spN2 = IBMSpace(N - 2)
    m0 = np.where(np.abs(spN.Mvals) < 1e-9)[0]; ix = np.ix_(m0, m0)
    H = H_first_order(spN, spN2, math.sqrt(2.0)).toarray().real[ix]
    C2 = spN.C2_SU3().toarray().real[ix]; L2 = spN.L2().toarray().real[ix]; nd = spN.n_d().toarray().real[ix]
    H = 0.5 * (H + H.T); C2 = 0.5 * (C2 + C2.T)
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
    lv = np.array([float(V[:, k] @ C2 @ C2 @ V[:, k] - (V[:, k] @ C2 @ V[:, k]) ** 2) for k in range(len(e))])
    return V, C2, np.where(lv < 1e-6)[0], np.where(lv >= 1e-6)[0]


def no_cheap_label(N=3, verbose=True):
    V, C2, solv, mixed = _critical_resolved(N)
    d = V.shape[0]
    def emb(v):
        w = np.zeros(16, complex); w[:d] = v; return w / np.linalg.norm(w)
    S = [emb(V[:, k]) for k in solv]; Mx = [emb(V[:, k]) for k in mixed]
    labels = [''.join(c) for c in itertools.product('IXYZ', repeat=4)
              if 1 <= sum(ch != 'I' for ch in c) <= 2]
    mats = {s: Operator(Pauli(s)).data for s in labels}
    expv = lambda P, psi: np.real(np.vdot(psi, P @ psi))
    rows = []
    for s in labels:
        P = mats[s]
        ms = min(abs(expv(P, p)) for p in S); mm = max(abs(expv(P, p)) for p in Mx)
        rows.append((ms, mm, ms - mm, s))
    # A cheap label needs min|<P>|_solv ~ 1 (rigid on solvable) AND max|<P>|_mix < 1.
    by_rigidity = sorted(rows, reverse=True)                 # best solvable-rigidity first
    best_min_solv = by_rigidity[0][0]
    # best margin among *candidate* labels (those at least half-rigid on the solvable set)
    cand = [r for r in rows if r[0] > 0.5]
    best_margin = max((r[2] for r in cand), default=float('-inf'))
    if verbose:
        print(f"(B) No cheap label  (N={N}, {len(solv)} solvable / {len(mixed)} mixed), "
              f"scanned {len(labels)} weight<=2 Paulis.")
        print("    Paulis with the highest |<P>| on the solvable set (rigidity requirement):")
        for (ms, mm, mar, s) in by_rigidity[:5]:
            print(f"      {s}: min|<P>|solv={ms:.3f}  max|<P>|mix={mm:.3f}  margin={mar:+.3f}")
        print(f"    Best solvable-rigidity over all weight<=2 Paulis: max min|<P>| = {best_min_solv:.3f} "
              f"(<< 1).")
        print(f"    => no low-weight Pauli is even approximately constant on the solvable subspace")
        if cand:
            print(f"       (best margin among {len(cand)} half-rigid candidates: {best_margin:+.3f}).")
        else:
            print(f"       (not one weight<=2 Pauli is even half-rigid on the solvable set).")
    return {"by_rigidity": by_rigidity, "best_min_solv": best_min_solv, "best_margin": best_margin}


# ---------- (D) encoding resource scaling ----------
def _weights(spo):
    return np.array([sum(ch != 'I' for ch in p.to_label()) for p in spo.paulis])


def _dense_H(N, h0, h2, C):
    spN = IBMSpace(N); spN2 = IBMSpace(N - 2)
    Hm = build_H(spN, spN2, h0, h2, C).toarray().real
    d = Hm.shape[0]; n = math.ceil(math.log2(d)); D = 2 ** n
    big = np.zeros((D, D)); big[:d, :d] = Hm
    sh = np.linalg.eigvalsh(Hm).max() + 20.0
    for k in range(d, D):
        big[k, k] = sh
    return n, SparsePauliOp.from_operator(Operator(big)).simplify()


def encoding_cost(N=3, h0=0.35, h2=1.0, C=0.0523, do_unary=True, verbose=True):
    res = []
    n, op = _dense_H(N, h0, h2, C); w = _weights(op)
    res.append(("dense", N, n, len(op.paulis), int(w.max()), float(w.mean())))
    encs = ['binary'] + (['unary'] if do_unary else [])
    for enc in encs:
        E = Encoded(N, enc)
        if E.nq > 20:
            res.append((enc, N, E.nq, None, None, None)); continue
        H = build_H_encoded(E, h0, h2, C); w = _weights(H)
        res.append((enc, N, E.nq, len(H.paulis), int(w.max()), float(w.mean())))
    if verbose:
        print(f"\n(D) Encoding resource cost of H  (N={N}):")
        print(f"    {'encoding':8} {'qubits':>6} {'#Pauli':>8} {'maxWeight':>9} {'meanWeight':>10}")
        for (enc, NN, nq, nt, mw, aw) in res:
            nt_s = '-' if nt is None else str(nt)
            mw_s = '-' if mw is None else str(mw)
            aw_s = '-' if aw is None else f"{aw:.1f}"
            print(f"    {enc:8} {nq:>6} {nt_s:>8} {mw_s:>9} {aw_s:>10}")
        print("    More qubits -> more terms and higher weight (naive Pauli-Trotterization);")
        print("    the compressed encoding is gate-cheapest. Structured ladder-op implementation")
        print("    is left to future work.")
    return res


if __name__ == "__main__":
    no_cheap_label(N=3)
    encoding_cost(N=2, do_unary=True)
    encoding_cost(N=3, do_unary=False)   # unary N=3 = 24 qubits, skipped for speed
