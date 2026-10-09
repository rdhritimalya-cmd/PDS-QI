"""
hardware_run_kingston.py  --  Device run of the label-variance readout on IBM Kingston.

Measures the static diagnostic  Var C2[SU(3)] = <C2^2> - <C2>^2  on variationally/
exactly prepared solvable and mixed eigenstates of the N=3 first-order critical block,
and reports the solvable-vs-mixed contrast.  This is the hardware-affordable observable
(shallow state-preparation circuit, ~17 two-qubit gates after routing); the deep
time-evolution / echo form is NOT run on hardware (see the manuscript's resource
discussion).  No quantum-advantage claim: the model is classically diagonalizable and a
device run is a readout-feasibility demonstration.

Error handling on device: EstimatorV2 with resilience_level=2 (readout twirling/TREX +
zero-noise extrapolation).  The sd-IBM also conserves N and M exactly, so symmetry
post-selection is available as an additional, zero-cost filter via the Sampler path
(documented at the bottom); ZNE is used here as the primary mitigation.

USAGE
  Local prediction on the Kingston noise model (no account needed, runs here):
      python hardware_run_kingston.py
  Real device (needs a saved IBM Quantum account with access to ibm_kingston):
      python hardware_run_kingston.py --device
  First-time account setup (once):
      from qiskit_ibm_runtime import QiskitRuntimeService
      QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<TOKEN>")

Requires: numpy, qiskit, qiskit-ibm-runtime, ibm_pds, critical.
"""
import numpy as np, math, json, sys, argparse
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from ibm_pds import IBMSpace
from critical import H_first_order

N_DEFAULT = 3
N_SOLV_EXEMPLARS = 3
N_MIX_EXEMPLARS = 2
SHOTS = 4000


def block_states(N):
    """Return embedded (4-qubit) C2 operator and chosen solvable/mixed exemplar states."""
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
    d = len(e)
    lv = np.array([float(V[:, k] @ C2 @ C2 @ V[:, k] - (V[:, k] @ C2 @ V[:, k]) ** 2) for k in range(d)])
    solv = [k for k in range(d) if lv[k] < 1e-6][:N_SOLV_EXEMPLARS]
    mixed = [k for k in range(d) if lv[k] >= 1e-6][:N_MIX_EXEMPLARS]
    nq = int(np.ceil(np.log2(d))); D = 2 ** nq
    C2e = np.zeros((D, D)); C2e[:d, :d] = C2
    op_C2 = SparsePauliOp.from_operator(Operator(C2e)).simplify()
    op_C2sq = SparsePauliOp.from_operator(Operator(C2e @ C2e)).simplify()
    states = {}
    for k in solv + mixed:
        w = np.zeros(D, complex); w[:d] = V[:, k]; w /= np.linalg.norm(w)
        states[k] = (w, "solvable" if lv[k] < 1e-6 else "mixed", float(lv[k]))
    return nq, op_C2, op_C2sq, states


def prep_circuit(w, nq):
    qc = QuantumCircuit(nq); qc.append(StatePreparation(w), range(nq))
    return qc.decompose(reps=6)


def get_backend(device):
    if device:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        return service.backend("ibm_kingston"), False
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston(), True


def main(device=False, N=N_DEFAULT):
    from qiskit_ibm_runtime import EstimatorV2, EstimatorOptions
    nq, op_C2, op_C2sq, states = block_states(N)
    backend, is_fake = get_backend(device)
    print(f"backend: {backend.name}  ({'LOCAL noise-model prediction' if is_fake else 'REAL DEVICE'})")
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)

    pubs = []; order = []
    for k, (w, kind, lv) in states.items():
        isa = pm.run(prep_circuit(w, nq))
        twoq = sum(isa.count_ops().get(g, 0) for g in ("cz", "ecr", "cx"))
        print(f"  state {k} ({kind}): transpiled 2q-gates = {twoq}, depth = {isa.depth()}")
        obs = [op_C2.apply_layout(isa.layout), op_C2sq.apply_layout(isa.layout)]
        pubs.append((isa, obs)); order.append((k, kind, lv))

    options = EstimatorOptions(default_shots=SHOTS)
    options.resilience_level = 2            # TREX readout mitigation + ZNE
    est = EstimatorV2(mode=backend, options=options)
    print(f"\nrunning {len(pubs)} states x 2 observables, {SHOTS} shots, resilience_level=2 ...")
    res = est.run(pubs).result()

    out = {"backend": backend.name, "is_fake": is_fake, "N": N, "shots": SHOTS, "states": []}
    print(f"\n{'state':>6} {'kind':>9} {'<C2>':>9} {'<C2^2>':>10} {'Var C2':>9}")
    var_solv, var_mix = [], []
    for (k, kind, lv), pub_res in zip(order, res):
        ev = np.asarray(pub_res.data.evs).ravel()
        c2, c2sq = float(ev[0]), float(ev[1])
        var = c2sq - c2 ** 2
        (var_solv if kind == "solvable" else var_mix).append(var)
        out["states"].append({"k": k, "kind": kind, "exact_labvar": lv,
                              "C2": c2, "C2sq": c2sq, "VarC2": var})
        print(f"{k:>6} {kind:>9} {c2:>9.3f} {c2sq:>10.2f} {var:>9.3f}")
    if var_solv and var_mix:
        contrast = min(var_mix) / max(max(var_solv), 1e-6)
        out["contrast_min_mix_over_max_solv"] = contrast
        print(f"\nmax Var(solvable) = {max(var_solv):.3f}   min Var(mixed) = {min(var_mix):.3f}"
              f"   contrast = {contrast:.1f}x")
        print("(exact: solvable Var = 0, mixed Var = O(N^2); contrast > 1 means the "
              "separation survives)")
    fn = f"kingston_{'device' if device else 'fakenoise'}_N{N}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print("saved", fn)


# ---------------------------------------------------------------------------
# OPTIONAL: symmetry post-selection via the Sampler path.
# The physical sector is the first d computational basis states (N,M=0 conserved).
# With SamplerV2 one measures each qubit-wise-commuting Pauli group, discards shots
# outside the physical sector (detected leakage), and reconstructs <C2>, <C2^2>.
# Post-selection is exact only for the Z-basis group; for rotated-basis groups it is
# applied on the pre-rotation sector membership carried by an ancilla parity check.
# ZNE (used above) is the simpler primary mitigation and needs no ancilla.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true", help="run on real ibm_kingston (needs account)")
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    a = ap.parse_args()
    main(device=a.device, N=a.N)
