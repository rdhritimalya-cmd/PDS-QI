# State-resolved quantum-information fingerprints of partial dynamical symmetry in the interacting boson model

Code to reproduce and inspect the numerical calculations associated with:

> D. Roy, *State-resolved quantum-information fingerprints of partial dynamical symmetry in the interacting boson model* (2026).

The repository implements calculations for the \(sd\)-interacting boson model, including exact diagonalization, symmetry-label variance, block coherence and purity, cone rigidity, controlled symmetry breaking, critical-point analyses, the \(^{168}\mathrm{Er}\) application, and quantum-simulation benchmarks.

The classical exact-diagonalization calculations use NumPy and SciPy. The quantum-simulation and hardware-oriented scripts have additional Qiskit dependencies. A real-hardware script is included, but access to IBM Quantum hardware and suitable credentials is required to submit a new device run; the included `kingston_device_N3.json` is a recorded result, not a guarantee that the hardware run can be repeated unchanged.

## Requirements

- Python 3.9 or newer
- NumPy, SciPy, Matplotlib
- Optional quantum-simulation dependencies: Qiskit, Qiskit Aer, pylatexenc
- Optional IBM hardware workflow: Qiskit IBM Runtime and an appropriately configured IBM Quantum account

Install the listed dependencies with:

```bash
pip install -r requirements.txt
```

The requirements file includes the optional quantum-simulation packages for convenience. Classical calculations do not require access to quantum hardware.

## Quick start

Run the repository's existing reproduction suite:

```bash
python run_all.py
```

This runs the checks wired into `run_all.py`; it does **not** automatically execute every script in this repository. Run individual analyses as needed, for example:

```bash
python test_conventions.py
python casimir3.py
python threshold_scan.py
python qsim_echo.py
python label_locality.py
python qsim_hardware_noise.py
```

Some calculations may take appreciable time. Qiskit- and IBM Runtime-dependent scripts require the optional dependencies listed above.

## Main modules

### Core IBM and symmetry calculations

- `ibm_pds.py` — \(sd\)-IBM Fock basis, boson operators, SU(3)-PDS Hamiltonian, quadrupole and Casimir operators, and irrep enumeration.
- `critical.py` — first- and second-order critical Hamiltonians and the O(5) Casimir.
- `run_step0.py`, `run_step0_v2.py` — reduced-density-matrix/entropy helpers and classical baseline checks.
- `casimir3.py` — cubic SU(3) Casimir and conjugate-irrep checks.
- `threshold_scan.py` — classification-threshold robustness.
- `rigidity_cone.py` — response along PDS-preserving and symmetry-breaking directions.
- `q2_definitive.py`, `q2_secondorder.py` — first- and second-order critical-point analyses.
- `robustness.py`, `false_positive.py`, `pds_classification_analysis.py` — controlled symmetry breaking, non-PDS controls, and related numerical analyses.
- `label_locality.py` — low-weight Pauli-label and encoding-resource analyses.
- `kremer.py`, `kremer_bridge.py` — O(6) label machinery and the purity/coherence comparison.
- `er168_anchor.py` — the \(^{168}\mathrm{Er}\) application, including SU(3) admixtures and transition-strength ratios.
- `magic.py` — stabilizer 2-Rényi entropy.
- `figdata.py` — shared figure and analysis helpers.

### Quantum-simulation and encoding modules

These scripts support the quantum-simulation discussion in Section 11 of the manuscript:

- `qsim_demo.py` — state preparation and variational benchmarks.
- `qsim_trotter.py` — Trotterized time evolution and step-size scaling.
- `qsim_echo.py` — Casimir Loschmidt-echo diagnostic.
- `qsim_hardware_noise.py` — noise-model simulations.
- `ibm_encodings.py` — alternative encodings and spectrum checks.
- `hardware_run_kingston.py` — IBM Kingston workflow, including local simulation mode and an optional real-device mode.
- `kingston_device_N3.json` — recorded hardware-run results.
- `label_locality.py` — encoding-resource and Pauli-operator analysis.

### Figure-generation scripts

The repository includes figure-generation scripts such as `mk_fig_gate.py`, `mk_fig_cone.py`, `mk_fig_q2.py`, `mk_fig_q2so.py`, `mk_fig_kremer.py`, `mk_fig_er168.py`, `make_fig_robustness.py`, `make_fig_additions.py`, `make_fig_kingston.py`, and `make_fig_qsim.py`. Check each script's source for its exact inputs and output filenames before running it. Quantum-simulation figure generators may require Qiskit dependencies.

## Conventions

Operator normalizations, degeneracy handling, and the \(M=0\) basis conventions are described in the manuscript and exercised by `test_conventions.py`. Numerical zero means a value consistent with the diagonalization or measurement precision; thresholds used for state classification are stated in the manuscript and analysis scripts.

## Reproducibility notes

- Numerical results can depend on the versions of Python and the scientific-computing packages.
- Variational optimization may follow different optimization paths across runs.
- Simulated-noise results are not the same as results from a physical quantum processor.
- `hardware_run_kingston.py` can require IBM account configuration and access to a suitable backend when run in device mode.
- The JSON file records a previous hardware run; it should be treated as recorded output, not as a substitute for independently rerunning the experiment.

## Citation

If you use this software, cite the associated manuscript and the exact archived software version used.

The source code for version 1.1.0 is intended to be archived on Zenodo after its GitHub release. Add the version-specific DOI here after Zenodo has completed archiving.

## License

Released under the MIT License. See `LICENSE`.
