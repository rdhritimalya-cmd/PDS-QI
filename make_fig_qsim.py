"""
Regenerate the quantum-simulation figures (Sec. IX):
  fig_qsim.png    -- (a) VQD-prepared label variance (raw vs label-resolved)
                     (b) Trotter error vs number of steps
  fig_ansatz.png  -- the hardware-efficient real ansatz circuit (n=4, L=2 shown)

Requires: qiskit, qiskit-aer, ibm_pds.py, critical.py, qsim_demo.py, qsim_trotter.py
The VQD run is cached to prepared_N3.pkl on first execution.
"""
import numpy as np, math, os, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qsim_demo as q
import qsim_trotter as qt
from scipy.linalg import expm
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit import QuantumCircuit, ParameterVector


def prepare_N3(reps=6, restarts=4, maxiter=800):
    """Run VQD on the N=3 first-order critical block and cache the states."""
    if os.path.exists("prepared_N3.pkl"):
        return pickle.load(open("prepared_N3.pkl", "rb"))
    N = 3
    Hb, C2b, L2b, ndb = q.block(N)
    d = Hb.shape[0]
    Hbig, n, D, sh = q.embed(Hb)
    found = q.vqd(Hbig, n, d, reps=reps, restarts=restarts, maxiter=maxiter)
    phys = [(E, psi[:d] / math.sqrt(np.vdot(psi[:d], psi[:d]).real)) for E, psi in found]
    data = {"phys": phys, "Hb": Hb, "C2b": C2b, "L2b": L2b, "ndb": ndb, "d": d, "n": n}
    pickle.dump(data, open("prepared_N3.pkl", "wb"))
    return data


def make_qsim_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    # (a) VQD-prepared label variance (raw vs label-resolved), N=3
    D = prepare_N3()
    phys, C2b, L2b, ndb, d = D["phys"], D["C2b"], D["L2b"], D["ndb"], D["d"]
    C2sq = C2b @ C2b
    raw = np.array([max(np.vdot(p, C2sq @ p).real - np.vdot(p, C2b @ p).real ** 2, 1e-16)
                    for _, p in phys])
    res = q.degeneracy_resolve_prepared(phys, d, C2b, L2b, ndb)
    res = np.maximum(res, 1e-16)
    idx = np.arange(d)
    ax1.semilogy(idx, np.sort(raw)[::-1], "s", color="#888", ms=7, label="prepared (raw)")
    ax1.semilogy(idx, np.sort(res)[::-1], "o", color="#c0392b", ms=7,
                 label="prepared + label-resolved")
    ax1.axhline(1e-6, color="k", ls=":", lw=1)
    ax1.text(0.3, 3e-6, "solvable threshold", fontsize=8)
    ax1.set_xlabel("eigenstate index (sorted)")
    ax1.set_ylabel(r"$\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$ on VQD state")
    ax1.set_title(r"(a) VQD-prepared states, $N=3$ first-order critical", fontsize=9.5)
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_ylim(1e-14, 1e3)

    # (b) Trotter error vs steps, N=2 physical initial state
    N = 2
    op, big, n, dd = qt.pauli_H(N)
    terms = [Operator(SparsePauliOp(pl)).data for pl in op.paulis]
    coeffs = np.real(op.coeffs)
    Hb = big[:dd, :dd]
    ev, V = np.linalg.eigh(Hb)
    psi = np.zeros(big.shape[0], complex)
    psi[:dd] = (V[:, 0] + V[:, dd // 2]) / math.sqrt(2)
    psi /= np.linalg.norm(psi)
    t = 0.5
    Uex = expm(-1j * big * t)
    steps = np.array([1, 2, 4, 8, 16, 32, 64])
    f1, f2 = [], []
    for s in steps:
        U1 = qt.trotter1(terms, coeffs, t, s)
        U2 = qt.trotter2(terms, coeffs, t, s)
        f1.append(1 - abs(np.vdot(Uex @ psi, U1 @ psi)) ** 2)
        f2.append(1 - abs(np.vdot(Uex @ psi, U2 @ psi)) ** 2)
    f1, f2 = np.array(f1), np.array(f2)
    ax2.loglog(steps, f1, "o-", color="#2c7fb8", label="first-order Trotter")
    ax2.loglog(steps, f2, "s-", color="#c0392b", label="second-order (Strang)")
    ax2.loglog(steps[3:], f1[3] * (steps[3] / steps[3:]) ** 2, "--",
               color="#2c7fb8", alpha=0.5, lw=1, label=r"$\propto s^{-2}$")
    ax2.loglog(steps[3:], f2[3] * (steps[3] / steps[3:]) ** 4, "--",
               color="#c0392b", alpha=0.5, lw=1, label=r"$\propto s^{-4}$")
    ax2.set_xlabel("number of Trotter steps $s$")
    ax2.set_ylabel(r"state infidelity $1-|\langle\psi_{\rm exact}|\psi_{\rm Trotter}\rangle|^2$")
    ax2.set_title(r"(b) Trotterized $e^{-i\hat Ht}$, $N=2$ (5 qubits)", fontsize=9.5)
    ax2.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig("fig_qsim.png", dpi=150)
    print("saved fig_qsim.png")


def make_ansatz_figure():
    def ansatz(n, reps):
        p = ParameterVector("θ", n * (reps + 1))
        qc = QuantumCircuit(n)
        idx = 0
        for qb in range(n):
            qc.ry(p[idx], qb); idx += 1
        for r in range(reps):
            qc.barrier()
            for qb in range(n - 1):
                qc.cx(qb, qb + 1)
            qc.cx(n - 1, 0)
            qc.barrier()
            for qb in range(n):
                qc.ry(p[idx], qb); idx += 1
        return qc, p

    qc, _ = ansatz(4, 2)   # 4 qubits (N=3 block); L=2 shown, runs use L=6
    fig = qc.draw(output="mpl", fold=-1, initial_state=True, scale=1.0)
    fig.savefig("fig_ansatz.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("saved fig_ansatz.png")


if __name__ == "__main__":
    make_qsim_figure()
    make_ansatz_figure()
