"""
qsim_hardware_noise.py  --  Does the (shallow) label-variance readout survive
realistic device noise?

The static readout is the hardware-affordable observable: a state preparation
(~11-17 two-qubit gates after transpilation) followed by a measurement of
Var C2[SU(3)] = <C2^2> - <C2>^2.  It is noise-FRAGILE because, for a solvable
state, that variance is the near-exact cancellation of two O(N^4) moments; but
the circuit is SHALLOW, so error mitigation can rescue it -- unlike a deep
time-evolution circuit.  The IBM Hamiltonian conserves the total boson number N
and the projection M exactly, so any shot landing outside the physical sector is
a detected error and is discarded (symmetry post-selection), at zero circuit cost.

This module runs a Qiskit Aer density-matrix simulation (noise exact, shot-free)
of the N=3 first-order critical block with a Heron-like depolarizing+readout
model, and reports the solvable-vs-mixed Var contrast with and without symmetry
post-selection, swept over the two-qubit error rate.  State preparation uses
exact (StatePreparation) circuits to isolate the readout's noise behaviour; VQD
prepares the same states at comparable depth (see qsim_demo.py).

Requires: numpy, qiskit, qiskit-aer, ibm_pds, critical.
"""
import numpy as np, math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from ibm_pds import IBMSpace
from critical import H_first_order

BASIS = ["cz", "rz", "sx", "x"]   # Heron-like native set


def _block(N):
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


def _noise_model(p2, p1=3e-4, pro=1e-2):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - pro, pro], [pro, 1 - pro]]))
    return nm


def run(N=3, p2_grid=(0.0, 1e-3, 2.5e-3, 5e-3, 1e-2), verbose=True):
    V, C2, solv, mixed = _block(N)
    d = V.shape[0]; nq = int(np.ceil(np.log2(d))); D = 2 ** nq
    C2e = np.zeros((D, D)); C2e[:d, :d] = C2; C2e2 = C2e @ C2e
    Pphys = np.diag([1.0] * d + [0.0] * (D - d))

    def prep(vec):
        w = np.zeros(D, complex); w[:d] = vec; w /= np.linalg.norm(w)
        qc = QuantumCircuit(nq); qc.append(StatePreparation(w), range(nq))
        return transpile(qc.decompose(reps=6), basis_gates=BASIS, optimization_level=3,
                         seed_transpiler=1)   # full connectivity: no routing permutation

    def rho(qc, p2):
        sim = AerSimulator(method="density_matrix", noise_model=_noise_model(p2))
        c = qc.copy(); c.save_density_matrix()
        return np.asarray(sim.run(c).result().data(0)["density_matrix"])

    def var(r, postselect):
        if postselect:
            r = Pphys @ r @ Pphys; tr = np.trace(r).real
            if tr > 1e-12:
                r = r / tr
        c = np.trace(r @ C2e).real; c2 = np.trace(r @ C2e2).real
        return c2 - c * c

    circs = {k: prep(V[:, k]) for k in range(d)}
    twoq = int(np.median([circs[k].count_ops().get("cz", 0) for k in range(d)]))
    results = []
    if verbose:
        print(f"N={N}: {len(solv)} solvable, {len(mixed)} mixed; prep 2q-gates (median) = {twoq}")
        print(f"{'2q err':>8} {'post-sel':>8} {'maxVar solv':>12} {'minVar mix':>11} {'contrast':>9}")
    for p2 in p2_grid:
        rhos = {k: rho(circs[k], p2) for k in range(d)}
        for ps in (False, True):
            vs = max(var(rhos[k], ps) for k in solv)
            vm = min(var(rhos[k], ps) for k in mixed)
            contrast = vm / max(vs, 1e-9)
            results.append((p2, ps, vs, vm, contrast))
            if verbose:
                print(f"{p2:>8.1e} {str(ps):>8} {vs:>12.2f} {vm:>11.2f} {contrast:>8.1f}x")
    return results


if __name__ == "__main__":
    run(N=3)
