"""
run_all.py -- run the full reproduction suite for the manuscript.

Executes every verification and reproduction script in sequence and reports
pass/fail. Intended as a one-command check that the archive reproduces the
numbers quoted in the paper. Small N by default; the heavier N=16 168Er run is
included in er168_anchor.py and may take a minute.

Usage:
    python run_all.py
"""
import subprocess
import sys

SCRIPTS = [
    ("test_conventions.py", "Clebsch-Gordan / Casimir convention self-tests"),
    ("casimir3.py", "Cubic Casimir conjugate-irrep injectivity check (Sec. III)"),
    ("threshold_scan.py", "Classification threshold robustness (Table I)"),
    ("rigidity_cone.py", "Cone rigidity at the stable SU(3)-PDS point (Fig. 2)"),
    ("q2_definitive.py", "First-order criticality: type-I coexistence (Fig. 3)"),
    ("q2_secondorder.py", "Second-order criticality: O(5) seniority ladder (Fig. 4)"),
    ("kremer.py", "Kremer O(6) sigma-label verification"),
    ("kremer_bridge.py", "Purity/coherence bridge, Delta_sigma=2.47 (Fig. 5)"),
    ("magic.py", "Magic (stabilizer 2-Renyi) complementarity (Sec. VIII)"),
    ("er168_anchor.py", "168Er anchor: counts, energies, admixtures, B(E2) (Sec. VII)"),
    ("qsim_demo.py", "Sec. X: qubit encoding + VQE/VQD state prep; max|dE| vs exact"),
    ("qsim_trotter.py", "Sec. X: Trotterized e^{-iHt}, error vs step count"),
]


def main():
    results = []
    for script, desc in SCRIPTS:
        print("=" * 70)
        print(f"RUN  {script}\n     {desc}")
        print("=" * 70)
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
