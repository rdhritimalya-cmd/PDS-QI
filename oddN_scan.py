"""
oddN_scan.py -- Table I rows for odd boson numbers N = 5, 7, ..., 15 at the stable
SU(3)-PDS point (Class I / II / III, smallest and median mixed variance).
Run from the repository root: python oddN_scan.py
"""
import numpy as np
from ibm_pds import build_H
from revision_common import Block, classify, H0, H2, CC

for N in (5, 7, 9, 11, 13, 15):
    b = Block(N)
    e, V = b.solve(b.R(build_H(b.s, b.s2, H0, H2, CC)))
    vc, pure, sol, Ls = classify(b, e, V, H2, CC)
    print(f"N={N} dim={len(e)} I={sol.sum()} II={pure.sum() - sol.sum()} III={(~pure).sum()} "
          f"maxpure={vc[pure].max():.0e} minmix={vc[~pure].min():.1f} median={np.median(vc[~pure]):.1f}",
          flush=True)
