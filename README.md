# PDS-QI: code and data

Code, verification scripts, logs and the recorded IBM Quantum output for

> D. Roy, *Symmetry purity from two Casimir moments: partial dynamical symmetry in the interacting
> boson model, from exact spectra to a quantum processor* (2026), arXiv:2608.04486.

Earlier versions of the manuscript (arXiv v1, repository v1.x) carried the title *State-resolved
quantum-information fingerprints of partial dynamical symmetry in the interacting boson model*.
Repository version 2.0.0 accompanies the revised manuscript. Section, table and figure numbers
below refer to the revised version.

The classical calculations (exact diagonalization of the $sd$-IBM in the $M=0$ block) need only
NumPy, SciPy and Matplotlib. The quantum-simulation scripts need Qiskit and Qiskit Aer. Submitting
a new hardware job needs Qiskit IBM Runtime and an IBM Quantum account; `kingston_device_N3.json`
is the recorded output of the run reported in the paper.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py              # all checks (classical + quantum simulations)
python run_all.py --classical  # exact-diagonalization checks only
```

Expected outputs are stored in `logs/`. Variational optimization (`qsim_demo.py`) can follow
different paths between runs; energies agree with exact diagonalization to about 1e-13.

## Where each result comes from

| Manuscript item | Script | Log |
|---|---|---|
| Conventions, Eq. (4), $\hat H(\sqrt2)$ relation, $[\hat H,\hat C_2[O(5)]]$ (Sec. II) | `verify_revision.py` (A, B), `test_conventions.py` | `logs/log_verify_revision.txt` |
| Cubic-Casimir check (Sec. III) | `casimir3.py` | `logs/log_run_all.txt` |
| Table I, even $N$; threshold scan, Table IV | `verify_revision.py` (C) | `logs/log_verify_revision.txt` |
| Table I, odd $N$ | `oddN_scan.py` | `logs/log_oddN.txt` |
| Class II counting, $L$ multiplicities | `consistency_checks.py` (7), `verify_revision.py` (C) | `logs/log_consistency_checks.txt` |
| First-order critical point, both degeneracy readings (Sec. IV B, Fig. 1) | `verify_revision.py` (D), `make_fig_revision.py` | `logs/log_verify_revision.txt` |
| Second-order point (Sec. IV C) | `verify_revision.py` (E), `consistency_checks.py` (6) | both logs |
| Table II, random consistent-$Q$ sample, consistent-$Q$ line (Sec. IV D) | `verify_revision.py` (F, G), `false_positive.py` | `logs/log_verify_revision.txt` |
| Controlled breaking, three measures (Sec. V, Fig. 2b) | `verify_revision.py` (H), `consistency_checks.py` (5), `make_fig_revision.py` | both logs |
| Mixing versus energy at $^{168}$Er (Sec. V) | `energy_mixing.py` | `logs/log_energy_mixing.txt` |
| $^{168}$Er anchor, admixtures, $B(E2)$ (Sec. VI, Fig. 3) | `er168_anchor.py`, `consistency_checks.py` (1), `mk_fig_er168.py` | `logs/log_run_all.txt` |
| Kremer et al. $\Delta\sigma$ reproduction (Sec. VIII) | `kremer.py` | printed output |
| Encodings and spectrum check, Table III (Sec. VII A) | `ibm_encodings.py`, `qsim_trotter.py` | `logs/log_encodings.txt` |
| Encoding cost and leakage check (Sec. VII F) | `check_encodings.py`, `label_locality.py` | `logs/log_encodings.txt`, `logs/log_run_all.txt` |
| VQD preparation and degeneracy resolution (Sec. VII B, Fig. 4a) | `qsim_demo.py`, `make_fig_qsim.py` | `logs/log_run_all.txt` |
| Trotter benchmark (Sec. VII C, Fig. 4b) | `qsim_trotter.py`, `make_fig_qsim.py` | `logs/log_trotter_t0.5.txt` |
| Casimir echo (Sec. VII D, Fig. 6) | `qsim_echo.py`, `make_fig_additions.py` | `logs/log_run_all.txt` |
| Low-weight Pauli scan (Sec. VII D) | `label_locality.py` | `logs/log_run_all.txt` |
| Noise model (Sec. VII E, Fig. 7) | `qsim_hardware_noise.py`, `make_fig_additions.py` | `logs/log_noise.txt` |
| Hardware run (Sec. VII E, Fig. 8) | `hardware_run_kingston.py`, `kingston_device_N3.json`, `make_fig_kingston.py` | JSON file |
| Ansatz drawing (Fig. 5) | `make_fig_qsim.py` | |

## Core modules

- `ibm_pds.py`: $sd$-IBM Fock basis, boson operators, SU(3)-PDS Hamiltonian, quadrupole and Casimir operators.
- `critical.py`: first- and second-order critical Hamiltonians, O(5) Casimir.
- `revision_common.py`: nested degeneracy refinement (Appendix A), $M=0$ block helper, classification into Classes I, II, III.
- `kremer.py`: O(6) Casimir and label machinery.
- `false_positive.py`: consistent-$Q$ Hamiltonians.

## Conventions

Degenerate levels are resolved by the nested refinement of Appendix A ($\hat L^2$, then
$\hat C_2[\mathrm{SU(3)}]$, then $\hat n_d$, unless stated otherwise). At the first-order critical
point the order matters and both readings are reported. The $N=3$ quantum-simulation scripts
diagonalize $\hat L^2$, $\hat C_2$ and $\hat n_d$ in turn over each degenerate window, which gives
the U(5) reading of Sec. IV B. A state is called SU(3)-pure when $\mathrm{Var}\,\hat C_2<10^{-6}$.

## Scripts from the first version

These reproduce material that is not in the revised manuscript (entanglement, cone rigidity, magic,
the Kremer purity/entanglement bridge, the second-order seniority ladder). They are kept for the
record of arXiv v1: `rigidity_cone.py`, `kremer_bridge.py`, `magic.py`, `q2_definitive.py`,
`q2_secondorder.py`, `run_step0.py`, `run_step0_v2.py`, `threshold_scan.py` (an earlier
threshold scan; its tighter tolerance gives 50 instead of 52 states at the $10^{-10}$ floor),
`pds_classification_analysis.py`, `robustness.py`, `mk_fig_gate.py`, `mk_fig_cone.py`,
`mk_fig_kremer.py`, `mk_fig_q2.py`, `mk_fig_q2so.py`, `make_fig_robustness.py`, `figdata.py`.
Note that `pds_classification_analysis.py` and `robustness.py` resolve degeneracies
sequentially rather than in the nested way of Appendix A; the revised numbers come from
`verify_revision.py`.

## Citation

Please cite the manuscript above and the archived software release:

> D. Roy, PDS-QI: code and data, version 2.0.0, Zenodo (2026),
> [doi:10.5281/zenodo.23984936](https://doi.org/10.5281/zenodo.23984936).

Earlier versions of the code have their own records on Zenodo.

## License

MIT License. See `LICENSE`.
