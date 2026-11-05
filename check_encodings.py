"""
check_encodings.py -- verify the qubit encodings used for the resource count of Sec. VII F.

For the stable SU(3)-PDS Hamiltonian (h0/h2 = 0.35, C/h2 = 0.0523) at N = 2 and 3 this script
builds the qubit operator in the dense, binary-per-mode and structured one-hot (unary) encodings
and checks, for each one,
  * that its restriction to the physical Fock states reproduces the exact spectrum,
  * that it sends no amplitude from a physical state to an unphysical bit string (leakage),
and prints the number of Pauli terms and their maximum and mean weight (the numbers quoted in
Sec. VII F and printed again by label_locality.py).

The structured one-hot mapping writes each ladder operator as weight-2 Pauli strings on the
mode register (see local_ops_unary_structured in ibm_encodings.py). Embedding the one-hot
ladder matrix densely into the 2^(N+1)-dimensional mode register instead (enc='unary_dense')
gives far more terms (242 074 at N = 2) for the same physics.
"""
from collections import defaultdict
import numpy as np
import ibm_encodings as ie
from label_locality import _dense_H

H0, H2, CC = 0.35, 1.0, 0.0523


def weights(op):
    return np.array([np.count_nonzero(p.x | p.z) for p in op.paulis])


def max_leak(E, H):
    sp = ie.IBMSpace(E.N)
    bits = [ie.fock_to_bits(st, E.q, E.enc) for st in sp.states]
    phys = set(bits)
    labels = [p.to_label() for p in H.paulis]
    leak = 0.0
    for b in bits:
        acc = defaultdict(complex)
        for lab, cf in zip(labels, H.coeffs):
            c, nb = ie.pauli_on_basis(lab, b, E.nq)
            if nb not in phys:
                acc[nb] += cf * c
        leak = max(leak, max((abs(v) for v in acc.values()), default=0.0))
    return leak


def main():
    ok = True
    print(f"{'N':>2} {'encoding':8s} {'qubits':>6} {'terms':>7} {'maxW':>4} {'meanW':>5}"
          f" {'spectrum err':>12} {'leak':>8}")
    for N in (2, 3):
        for enc in ("dense", "binary", "unary"):
            if enc == "dense":
                nq, H = _dense_H(N, H0, H2, CC)
                d = len(ie.IBMSpace(N).states)
                M = H.to_matrix()
                Hex = ie.build_H(ie.IBMSpace(N), ie.IBMSpace(N - 2), H0, H2, CC).toarray()
                err = np.abs(np.linalg.eigvalsh(M[:d, :d]) - np.linalg.eigvalsh(Hex)).max()
                leak = np.abs(M[d:, :d]).max()      # coupling from physical to padding states
            else:
                E = ie.Encoded(N, enc)
                nq = E.nq
                H = ie.build_H_encoded(E, H0, H2, CC)
                err = ie.verify_spectrum(E, H, H0, H2, CC)[0]
                leak = max_leak(E, H)
            w = weights(H)
            print(f"{N:>2} {enc:8s} {nq:>6} {len(H.paulis):>7} {w.max():>4} {w.mean():>5.1f}"
                  f" {err:>12.1e} {leak:>8.1e}", flush=True)
            ok &= err < 1e-8 and leak < 1e-10
    print("all encodings reproduce the exact spectrum without leakage" if ok else "CHECK FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
