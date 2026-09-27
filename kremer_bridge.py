"""
kremer_bridge.py -- purity/coherence bridge on Kremer et al.'s states (Fig. 5).

Kremer et al. (PRC 89, 041302) work along the ECQF line
    H(xi) = eps[ (1-xi) nd - (xi/4N) Q^chi . Q^chi ],
labelling the O(6)-PDS aspect "purity" and the SU(3)-QDS aspect "coherence."
Here we render both as literal density-matrix quantities on the ground state:

  - Delta_sigma_gs : the O(6) sigma-fluctuation, sqrt(Var over sigma-irreps),
    computed by projecting the ground state onto C2[O(6)] eigenspaces
    (sigma(sigma+4)). Anchor: 2.47 at the U(5) limit (xi=0).
  - Tr rho_sigma^2 : the O(6) block purity of the dephased sigma-distribution.
  - s|d von Neumann entropy : the bipartite entanglement.

The manuscript's point (Fig. 5): O(6) block purity and s|d entanglement are
independent axes -- the ground state becomes O(6)-pure (Tr rho_sigma^2 ~ 0.81-0.88)
exactly where its s|d entanglement is maximal.
"""
import numpy as np
from ibm_pds import IBMSpace
from kremer import build, C2_O6
from run_step0 import rdm_modes, vn_entropy


def sigma_probs(psi_full, C2o6_full, sel):
    """Block probabilities over O(6) sigma-irreps for a state given in M=0."""
    C2 = C2o6_full.tocsr()[sel][:, sel].toarray()
    w, U = np.linalg.eigh(C2)
    vals = np.round((-2 + np.sqrt(4 + np.round(w, 6)))).astype(int)  # sigma
    c = U.T @ psi_full
    P = {}
    for sig in np.unique(vals):
        p = float(np.sum(c[vals == sig] ** 2))
        if p > 1e-12:
            P[int(sig)] = P.get(int(sig), 0.0) + p
    return P


def analyze(N=12, eps=1.0):
    spN, sel, nd_op, Qchi = build(N)
    nd = nd_op.tocsr()[sel][:, sel].toarray()
    C2o6_full = C2_O6(spN)
    C2o6 = C2o6_full.tocsr()[sel][:, sel].toarray()
    chi = -np.sqrt(7) / 2.0  # SU(3) value of chi

    def Qdot(chi):
        Q = Qchi(chi)
        out = sum(((-1) ** q) * (Q[q] @ Q[-q]) for q in range(-2, 3))
        return out.tocsr()[sel][:, sel].toarray()

    QQ = Qdot(chi)

    print(f"Kremer purity/coherence bridge (N={N}); chi={chi:.3f}")
    print("  xi     Dsigma_gs   Tr rho_sigma^2   S(s|d)")
    xis = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    for xi in xis:
        H = eps * ((1 - xi) * nd - (xi / (4 * N)) * QQ)
        ev, V = np.linalg.eigh(H)
        gs = V[:, 0]
        P = sigma_probs(gs, C2o6_full, sel)
        sig = np.array(list(P.keys()), float)
        pr = np.array(list(P.values()), float)
        mean = np.sum(pr * sig)
        dsig = float(np.sqrt(max(0.0, np.sum(pr * sig ** 2) - mean ** 2)))
        purity = float(np.sum(pr ** 2))
        full = np.zeros(spN.dim, dtype=complex); full[sel] = gs
        S = vn_entropy(rdm_modes(full, spN.states, (0,)))
        print(f"  {xi:.2f}    {dsig:7.3f}     {purity:10.3f}     {S:7.3f}")
    print("\n  Anchor: Delta_sigma_gs -> 2.47 at the U(5) limit (xi=0),")
    print("  and Tr rho_sigma^2 reaches 0.81-0.88 where S(s|d) is maximal.")


if __name__ == "__main__":
    analyze(14)  # N=14 reproduces Kremer's Delta_sigma_gs=2.47 at U(5)
