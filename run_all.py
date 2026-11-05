"""
run_all.py -- run the reproduction suite for the manuscript

  D. Roy, "Symmetry purity from two Casimir moments: partial dynamical symmetry in the
  interacting boson model, from exact spectra to a quantum processor" (2026).

Executes the verification and reproduction scripts in sequence and reports pass/fail.
Section, table and figure numbers refer to the revised manuscript. Expected
outputs of the main checks are stored in logs/.

Usage:
    python run_all.py            # classical checks + quantum simulations
    python run_all.py --classical  # exact-diagonalization checks only (no Qiskit needed)
"""
import subprocess
import sys

CLASSICAL = [
    ("test_conventions.py", "Clebsch-Gordan / Casimir convention self-tests (Sec. II)"),
    ("casimir3.py", "Cubic Casimir conjugate-irrep check (Sec. III)"),
    ("verify_revision.py", "All exact-diagonalization numbers of Secs. II-V, Tables I, II, IV"),
    ("oddN_scan.py", "Table I, odd N rows"),
    ("energy_mixing.py", "Mixing versus excitation energy at 168Er (Sec. V)"),
    ("consistency_checks.py", "B(E2) vs theta/alpha, O(5) breaking exponent, second-order labels"),
    ("er168_anchor.py", "168Er anchor: counts, energies, admixtures, B(E2) (Sec. VI)"),
]
QUANTUM = [
    ("qsim_demo.py", "Sec. VII B: VQE/VQD state preparation, max|dE| vs exact"),
    ("qsim_echo.py", "Sec. VII D: Casimir echo and short-time identity"),
    ("label_locality.py", "Sec. VII D/F: low-weight Pauli scan and encoding cost"),
    ("check_encodings.py", "Sec. VII F: encodings reproduce the spectrum, no leakage, Pauli counts"),
    ("qsim_hardware_noise.py", "Sec. VII E: density-matrix noise model"),
]


def main():
    scripts = CLASSICAL if "--classical" in sys.argv else CLASSICAL + QUANTUM
    results = []
    for script, desc in scripts:
        print("=" * 70)
        print(f"RUN  {script}\n     {desc}")
        print("=" * 70, flush=True)
        r = subprocess.run([sys.executable, script])
        results.append((script, r.returncode == 0))
        print()
    print("=" * 70)
    print("SUMMARY")
    for script, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {script}")
    print("=" * 70)
    if all(ok for _, ok in results):
        print("All reproduction scripts completed.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
