# Verification report — quantum-section additions

Every new numerical claim, the code that produces it, and its verified value.
All runs: Python 3, numpy 2.5, scipy 1.18, qiskit 2.5, qiskit-aer 0.17,
qiskit-ibm-runtime 0.50. Core physics modules (`ibm_pds.py`, `critical.py`) are
the deposited repository versions, unchanged.

## (A) Echo = dynamical form of the label variance — `qsim_echo.py`

- Solvable states are exact C₂-eigenstates ⇒ **E(t) = 1.00000000** for all t
  (verified at t = 0.5 and t = 8; label variance ≤ 4×10⁻¹²).
- Mixed states dephase: at N=3, min solvable echo − max mixed echo grows from
  0.018 (t=1) to **0.345 (t≈7)**; mixed echo bottoms near 0.64.
- Short-time identity **E(t) = 1 − t²·Var C₂[SU(3)]** confirmed to 6 digits
  (e.g. mixed state: echo 0.999177 vs 1−t²Var 0.999177 at t=0.1, normalized).
- Block classification: 7 solvable / 3 mixed in the N=3 M=0 block (matches the
  manuscript's "seven up to one edge state").

## (B) No cheap label operator — `label_locality.py :: no_cheap_label`

- Scan of all 66 weight-≤2 Pauli operators in the 4-qubit compressed embedding.
- Best solvable-set magnitude attained by any such operator:
  **max min|⟨P⟩|_solv = 0.273** (a label needs ≈ 1).
- Not one weight-≤2 Pauli is even half-rigid (min|⟨P⟩|_solv > 0.5) on the
  solvable set; wherever |⟨P⟩| is large on solvable it is equally large on mixed
  (best margin ≤ 0). ⇒ the diagnostic has no low-weight representation.

## (C) Readout survival under device noise — `qsim_hardware_noise.py`

Aer density-matrix sim, heavy-hex model (2q depol swept, 1q = 3×10⁻⁴,
readout = 1×10⁻²), N=3, exact prep (~11 2q-gates), contrast = Var_mix / Var_solv:

| 2q error | no post-select | with symmetry post-select |
|---|---|---|
| 1×10⁻³ | 2.4× | 4.2× |
| 2.5×10⁻³ | 1.6× | 3.1× |
| 5×10⁻³ | 1.2× | 2.4× |
| 1×10⁻² | 0.9× (lost) | 1.9× |

(The "2q = 0" reference row in the raw output still carries 1q + readout noise;
the true noiseless solvable Var is < 10⁻¹⁰, confirmed in `qsim_echo.py`.)

## (D) Encoding resource scaling — `label_locality.py :: encoding_cost`

Spectrum of every encoding verified against exact Fock diagonalization to
≤ 3×10⁻¹³ (`ibm_encodings.verify_spectrum`). Pauli cost of H:

| encoding | qubits (N=3) | #Pauli terms | max weight | mean weight |
|---|---|---|---|---|
| dense (compressed) | 6 | 1,568 | 6 | 4.5 |
| binary-per-mode | 12 | 15,749 | 8 | 6.8 |
| unary (one-hot, N=2) | 18 | 242,074 | 12 | 9.8 |

⇒ more qubits give more terms and higher weight under naive Pauli-Trotterization;
the compressed encoding is gate-cheapest. Structured ladder-op implementation is
noted as future work.

## Hardware pipeline — `hardware_run_kingston.py`

- Verified to build and run on the **FakeKingston** local noise model
  (transpiled prep = 15 two-qubit gates per state).
- FakeKingston prediction, EstimatorV2 + resilience_level 2 (TREX + ZNE),
  no post-selection: **contrast ≈ 1.2×** (marginal, generic-depolarizing ceiling).
- **REAL DEVICE run (ibm_kingston, job db2u5su8v0ts73c3i75g, 2026-10-07T06:13Z)** —
  `kingston_device_N3.json`, EstimatorV2, 4000 shots, resilience_level 2 (TREX+ZNE),
  NO post-selection. qiskit 2.5.2 / runtime 0.50.0; backend v1.0.0, 156 qubits.
  Physical qubits: solvable {73,79,72,74} depth 61 (15 2q-gates); mixed {148,60,47,21} depth 0.
  **WITH error bars (EstimatorV2 stds, conservatively propagated as independent moments):**
  mixed Var = **236 ± 78** (exact 240; matches to 0.05σ; nonzero at **3.0σ**);
  solvable Var = **36 ± 136** (exact 0; **consistent with zero, 0.27σ**).
  Point estimates match the exact classification. The point-estimate ratio is 6.6, BUT the
  direct mixed−solvable difference is only **1.3σ** at 4000 shots — so "6.6×" is NOT a tight
  measured contrast; the defensible claim is: mixed resolved nonzero, solvable consistent with 0.
  Error bars are conservative (ignore the positive C2–C2² covariance, which the paper's
  δ(Var) formula subtracts and which would shrink them). Large errors come from the O(N⁴)
  moment difference at only 4000 shots — exactly the shot-budget sensitivity the §10.5 text
  predicts. Figure `fig_kingston.png` (manuscript Fig. 11, now with error bars); text §10.5.
  Feasibility/consistency demonstration, not a quantum-advantage claim; a shot-budget +
  covariance study is deferred.

## How to reproduce everything

```
python qsim_echo.py              # (A)
python label_locality.py         # (B) + (D)
python qsim_hardware_noise.py    # (C)
python make_fig_additions.py     # fig_echo.png, fig_noise.png
python hardware_run_kingston.py  # Kingston pipeline (local FakeKingston prediction)
```

## Scaling, robustness, and specificity — `robustness.py`, `false_positive.py`

- **Scaling** (stable SU(3)-PDS, N=4–16): pure residual ≤ 3×10⁻⁹; min mixed variance stays ≥ 22;
  median mixed variance fits **N^2.32** (not clean N²); separation ratio > 10⁹ at every N and does
  not close. The median mixed-state variance fits N^2.32 over the sampled range; the distribution is not described by a single clean N² law.
- **Approximate PDS** (H + ε·n_d, N=10): formerly-pure variance grows as **ε^1.97** (≈ε², perturbation
  theory); pure/mixed separation survives to ~10% breaking. Figure `fig_robustness`.
- **False positive** (Casten-triangle consistent-Q sweep, N=10): **0 nontrivial SU(3)-exact-label
  states** at all 12 generic interior points (xi in [0.05,0.95], chi in [-sqrt7/2,0]; only the n_d=0
  and n_d=N multiplets, which are exact everywhere, remain). Smallest nontrivial label variance over
  all 12 points = **212.6** (~2x10^2, about 8 orders above the 1e-6 threshold) — the minimum observed nontrivial variance is approximately 212.6. The diagnostic does not fire on non-PDS Hamiltonians.
- **Exact-label count table** (`pds_classification_analysis.py`, N=10, M=0 block, dim=203): SU(3) vertex **203/203** (all, as required in the exact-symmetry limit);
  stable SU(3)-PDS point **58**; first-order critical **56**; generic interior **trivial only**;
  U(5), O(6), second-order critical **0** (there the conserved label is O(5) seniority, not SU(3)).
- **Class I/II/III** (`pds_classification_analysis.py`): at each N the exact-label set splits into Class I
  (exact-label AND closed-form solvable) and Class II (exact-label, not solvable — the exhaustion
  states, exactly 2 for every N>=6); the rest are Class III (mixed). N=10: 56 / 2 / 145.
- **Second breaking direction** (`pds_classification_analysis.py`): O(5) Casimir perturbation gives small-eps
  exponent 1.75 (~eps^2), confirming the quadratic law is not special to n_d.
- **Baseline variance/entropy/purity** (`pds_classification_analysis.py`, N=10, eps=0.02): formerly-pure mean
  Var=3.0e-3, S_G=3.7e-6, 1-P_G=4.5e-7 — all three detect the breaking; variance needs only the two
  moments <C2>,<C2^2>, whereas S_G and 1-P_G need the full sector distribution {P_a}.
