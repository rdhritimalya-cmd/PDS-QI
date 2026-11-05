"""
energy_mixing.py -- mixing versus excitation energy at Leviatan's 168Er parameters
(N = 16): median Var C2[SU(3)], median <C2>, median block entropy S and e^S of the
SU(3)-mixed states in 1 MeV bins (Sec. V of the manuscript).
Run from the repository root: python energy_mixing.py
"""
import numpy as np
from ibm_pds import build_H
from revision_common import Block, var

b = Block(16)
e, V = b.solve(b.R(build_H(b.s, b.s2, 0.008, 0.004, 0.013)))
vc, mc = var(b.C2, V)
mix = vc >= 1e-6
w, U = np.linalg.eigh(b.C2)
wr = np.round(w).astype(int)
C = U.T @ V
P = np.array([np.sum(C[wr == v, :] ** 2, axis=0) for v in np.unique(wr)])
S = -np.sum(np.where(P > 1e-15, P * np.log(np.maximum(P, 1e-300)), 0), axis=0)
for lo in range(0, 12):
    s = mix & (e >= lo) & (e < lo + 1)
    if s.sum():
        print(f"[{lo},{lo + 1}) MeV n={s.sum():3d} medVar={np.median(vc[s]):7.0f} "
              f"med<C2>={np.median(mc[s]):7.0f} medS={np.median(S[s]):.2f} "
              f"medEffSectors={np.median(np.exp(S[s])):.1f}")
