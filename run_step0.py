"""
run_step0.py -- Step 0 classical-baseline helpers for the PDS + QI project.

Provides the reduced-density-matrix and entropy utilities that the analysis
scripts (q2_definitive.py, q2_secondorder.py, run_step0_v2.py) import:

    m0_block(space)                 -> indices of the M=0 block
    rdm_modes(psi, states, modes)   -> reduced density matrix over `modes`
    vn_entropy(rho)                 -> von Neumann entropy S = -Tr rho ln rho
    purity(rho)                     -> Tr rho^2

Boson-mode bipartition convention
---------------------------------
The sd-IBM Fock basis stores occupations as tuples
    (n_s, n_{d,-2}, n_{d,-1}, n_{d,0}, n_{d,1}, n_{d,2}).
`rdm_modes(psi, states, modes)` traces out all modes NOT listed in `modes`,
returning the reduced density matrix of the subsystem spanned by `modes`.
For the s|d bipartition used throughout the manuscript, call with modes=(0,):
the reduced state of the single s-mode, whose entropy equals that of the
complementary d-block (Schmidt symmetry for a pure global state).

The subsystem Hilbert space is labelled by the occupation vector restricted to
`modes`; because total boson number is fixed, the reduced density matrix is
block-diagonal in that occupation and is built exactly, without truncation.

This module performs no diagonalization of H itself; it only post-processes
eigenvectors supplied by the caller. All routines are exact.
"""
import numpy as np
from math import log


def m0_block(space):
    """Return the integer indices of the M=0 states of an IBMSpace."""
    return np.where(space.Mvals == 0)[0]


def rdm_modes(psi, states, modes):
    """Reduced density matrix of the subsystem spanned by `modes`.

    Parameters
    ----------
    psi : (dim,) complex ndarray
        Amplitudes in the FULL Fock basis `states` (not the M=0 restriction).
        Callers that work in the M=0 block should scatter their block vector
        back into a full-length array first (full[sel] = v).
    states : list of tuples
        The Fock occupation tuples indexing `psi`, in order.
    modes : tuple of int
        Tuple indices (into the occupation tuple) that define subsystem A.
        e.g. (0,) selects the s-mode; (1,2,3,4,5) selects the d-block.

    Returns
    -------
    rho : (dA, dA) complex ndarray
        The reduced density matrix Tr_B |psi><psi|, with rows/cols indexed by
        the distinct subsystem occupation vectors that actually occur.
    """
    modes = tuple(modes)
    comp = tuple(i for i in range(len(states[0])) if i not in modes)

    # Enumerate the subsystem (A) and environment (B) occupation labels present.
    a_labels = {}
    b_labels = {}
    for st in states:
        a = tuple(st[i] for i in modes)
        b = tuple(st[i] for i in comp)
        if a not in a_labels:
            a_labels[a] = len(a_labels)
        if b not in b_labels:
            b_labels[b] = len(b_labels)

    dA = len(a_labels)
    dB = len(b_labels)

    # Coefficient matrix C[a, b] = <a,b|psi>; rho_A = C C^dagger.
    C = np.zeros((dA, dB), dtype=complex)
    for amp, st in zip(psi, states):
        if amp == 0:
            continue
        a = tuple(st[i] for i in modes)
        b = tuple(st[i] for i in comp)
        C[a_labels[a], b_labels[b]] = amp

    rho = C @ C.conj().T
    tr = np.trace(rho).real
    if tr > 0:
        rho = rho / tr
    return rho


def vn_entropy(rho, base=None):
    """von Neumann entropy S = -Tr rho ln rho (natural log by default).

    Eigenvalues below 1e-12 are treated as zero. Pass base=2 for bits.
    """
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    w = w[w > 1e-12]
    S = float(-np.sum(w * np.log(w)))
    if base is not None:
        S /= log(base)
    return S


def purity(rho):
    """Tr rho^2."""
    return float(np.trace(rho @ rho).real)
