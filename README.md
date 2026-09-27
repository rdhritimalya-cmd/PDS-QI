# Quantum-information fingerprints of partial dynamical symmetry in the IBM

Code to reproduce the results of

> D. Roy, *Quantum-information fingerprints of partial dynamical symmetry in the
> interacting boson model* (2026).

The scripts diagonalize the $sd$-interacting boson model in the $m$-scheme boson
Fock basis and compute the symmetry-label variance, block coherence, block
purity, cone rigidity, and the supporting quantum-information measures reported
in the paper. Everything runs on a laptop with NumPy and SciPy; no quantum
hardware or Qiskit is required to reproduce the exact-diagonalization results
(Secs. II–IX). The variational and Trotter benchmarks of Sec. X use Qiskit Aer
and are provided separately.

## Requirements

- Python ≥ 3.9
- NumPy, SciPy
- (optional, for the Sec. X quantum-simulation benchmarks) Qiskit + Qiskit Aer

Install the core dependencies with

```
pip install -r requirements.txt
```

## Quick start

Run the full reproduction suite:

```
python run_all.py
```

This executes every script below and prints a pass/fail summary. Individual
scripts can be run directly, e.g. `python er168_anchor.py`.

## Files

Core library:

- `ibm_pds.py` — $sd$-IBM Fock basis, exact Clebsch–Gordan coefficients, boson
  operators, the SU(3)-PDS Hamiltonian of Leviatan [PRL 77, 818 (1996)], the
  SU(3) quadrupole and quadratic Casimir, and the irrep enumeration.
- `critical.py` — Leviatan's first- and second-order critical Hamiltonians
  [PRL 98, 242502 (2007)] and the O(5) Casimir.
- `run_step0.py` — reduced-density-matrix and entropy helpers (the $s|d$
  bipartition), used by the entanglement calculations.

Reproduction / verification:

- `test_conventions.py` — convention self-tests: the $\theta_2$ normalization
  identity, integer $L$ and $\tau$, and the SU(3) irrep content.
- `casimir3.py` — builds the cubic SU(3) Casimir $C_3$, verifies
  $[C_3, Q]\approx 0$, shows it splits the conjugate-irrep pairs that make $C_2$
  degenerate at $N=6,9,12,15$, and confirms every $\mathrm{Var}\,C_2=0$ state
  also has $\mathrm{Var}\,C_3=0$ (no count changes). Reproduces the check in
  Sec. III.
- `threshold_scan.py` — classification-threshold robustness (Table I).
- `rigidity_cone.py` — cone rigidity at the stable SU(3)-PDS point (Fig. 2).
- `q2_definitive.py` — first-order criticality; confirms the consistent-$Q$ line
  has no interior exact-label states while $H(\beta_0=\sqrt2)$ does (Fig. 3).
- `q2_secondorder.py` — second-order criticality; the O(5) seniority-graded
  entanglement ladder (Fig. 4).
- `kremer.py` — verifies the O(6) $\sigma$-label machinery used in the bridge.
- `kremer_bridge.py` — the purity/coherence bridge (Fig. 5); reproduces
  $\Delta\sigma_{\rm gs}=2.47$ at the U(5) limit ($N=14$) and shows the O(6)
  block purity and $s|d$ entanglement are independent axes.
- `magic.py` — exact stabilizer 2-Rényi entropy; magic is a separate axis from
  the label variance (Sec. VIII).
- `er168_anchor.py` — the $^{168}$Er anchor at Leviatan's fitted parameters:
  exact-label vs closed-form counts, ground- and $\gamma$-band energies, the
  $\beta$-band SU(3) admixtures, and the parameter-free $\gamma\!\to\! g$
  $B(E2)$ ratios (Sec. VII).
- `run_step0_v2.py` — the Step-0 classical baseline, which independently flags
  the two irrep-pure but non-closed-form states per spectrum.

## Conventions

Operator normalizations are fixed by the identity
$P_0^\dagger \tilde P_0 + P_2^\dagger\cdot\tilde P_2 = -C_2[\mathrm{SU(3)}]
+ 2\hat N(2\hat N+3)$ (Leviatan, arXiv:2010.10951), reproduced to $\le
2\times10^{-13}$ by `test_conventions.py`. All spectra are computed in the $M=0$
block, which contains one member of every angular-momentum multiplet.
Degeneracies are resolved by the order-fixed refinement described in the paper
($L^2$, then the relevant Casimir, then $\hat n_d$).

## Section X: quantum-simulation benchmarks (Qiskit)

These reproduce the state-preparation and Trotter benchmarks of Sec. X. They
require `qiskit` and `qiskit-aer` (see `requirements.txt`) and are the only
scripts that do; everything above runs on NumPy/SciPy alone.

- `qsim_demo.py` — encodes the N=3 first-order critical block on 4 qubits,
  verifies the mapped Pauli Hamiltonian reproduces the exact spectrum, prepares
  all ten eigenstates by VQE/VQD (hardware-efficient real ansatz), and prints
  `max|dE|` against exact diagonalization (target < 1e-4; obtained ~1e-13). It
  also provides `degeneracy_resolve_prepared`, the within-window Casimir
  diagonalization that recovers the sharp solvable/mixed split of Fig. 7(a).
- `qsim_trotter.py` — builds the Trotterized propagator and reports first- and
  second-order (Strang) infidelity versus step count, showing the s^-2 and s^-4
  scaling of Fig. 7(b).
- `make_fig_qsim.py` — regenerates `fig_qsim.png` (Fig. 7) and `fig_ansatz.png`
  (Fig. 8) from the two modules above. The VQD run is cached to
  `prepared_N3.pkl` on first execution.

Note: VQD uses a multi-restart optimizer, so the prepared-state energies are
reproducible to the stated tolerance but individual optimizer paths vary run to
run; the physics conclusions (encoding exactness, max|dE|, the degeneracy
resolution) are deterministic.

## License

Released under the MIT License (see `LICENSE`).

## Regenerating the figures

Each manuscript figure has a generator that reads directly from the code above,
so every panel matches its caption and no plot carries an internal working title:

- `mk_fig_gate.py`      -> `fig_gate_N10.png`            (Fig. 1)
- `mk_fig_cone.py`      -> `fig_cone_N10.png`            (Fig. 2)
- `mk_fig_q2.py`        -> `fig_q2_N10.png`              (Fig. 3)
- `mk_fig_q2so.py`      -> `fig_q2_secondorder_N10.png`  (Fig. 4)
- `mk_fig_kremer.py`    -> `fig_q3_kremer_N12.png`       (Fig. 5)
- `mk_fig_er168.py`     -> `fig_er168_anchor.png`        (Fig. 6)
- `make_fig_qsim.py`    -> `fig_qsim.png`, `fig_ansatz.png` (Figs. 7, 8; needs Qiskit)

`figdata.py` holds the shared helpers (degeneracy refinement, label variance,
s|d entropy). Figs. 7 and 8 additionally require `qiskit`, `qiskit-aer` and
`pylatexenc` (for the circuit drawer).
