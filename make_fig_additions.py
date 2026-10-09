"""
Generate the new quantum-section figures:
  fig_echo.png   -- (a) Loschmidt echo under C2[SU(3)]: solvable rigid at E=1,
                        mixed states dephase; (b) short-time E(t)=1-t^2 Var C2.
  fig_noise.png  -- static label-variance readout contrast vs 2q error, with and
                        without symmetry post-selection (Aer density-matrix, N=3).
Style matches the repo's existing mk_fig_*.py (matplotlib, same palette).
"""
import numpy as np, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qsim_echo as qe
import qsim_hardware_noise as qn

BLUE = "#2c7fb8"; RED = "#c0392b"; GREY = "#888888"


def fig_echo(N=3, fname="fig_echo.png"):
    d = qe.run(N=N, tgrid=np.linspace(0, 8, 81), verbose=False)
    t = d["tgrid"]; E = d["echo"]; solv = d["solv"]; mixed = d["mixed"]
    C2 = d["C2"]; V = d["V"]; span = d["C2span"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    # (a) full echo curves
    for k in solv:
        ax1.plot(t, E[k], color=BLUE, lw=1.6, alpha=0.9,
                 label="solvable (exact label)" if k == solv[0] else None)
    for k in mixed:
        ax1.plot(t, E[k], color=RED, lw=1.4, alpha=0.8,
                 label="mixed" if k == mixed[0] else None)
    ax1.set_xlabel(r"evolution time $t$ (units of $1/\mathrm{span}\,\hat C_2$)")
    ax1.set_ylabel(r"Loschmidt echo $E(t)=|\langle\psi|e^{-i\hat C_2 t}|\psi\rangle|^2$")
    ax1.set_title(r"(a) Echo under $\hat C_2[\mathrm{SU(3)}]$, $N=3$", fontsize=10)
    ax1.set_ylim(0.55, 1.02); ax1.legend(fontsize=8, frameon=False, loc="lower left")
    ax1.axhline(1.0, color="k", ls=":", lw=0.8)

    # (b) short-time: E(t) vs 1 - t^2 Var C2 for mixed states
    ts = np.linspace(0, 1.2, 40)
    for j, k in enumerate(mixed):
        vn = qe.label_variance(V[:, k], C2) / span ** 2
        Ek = [qe.echo(V[:, k], C2 / span, tt) for tt in ts]
        ax2.plot(ts, Ek, color=RED, lw=1.4, alpha=0.8,
                 label=r"$E(t)$ (mixed)" if j == 0 else None)
        ax2.plot(ts, 1 - ts ** 2 * vn, color="k", ls="--", lw=1.0,
                 label=r"$1-t^2\,\mathrm{Var}\,\hat C_2$" if j == 0 else None)
    ax2.plot(ts, np.ones_like(ts), color=BLUE, lw=1.6, label="solvable")
    ax2.set_xlabel(r"evolution time $t$")
    ax2.set_ylabel(r"echo $E(t)$")
    ax2.set_title(r"(b) Short-time: echo $=$ label variance", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    plt.tight_layout(); plt.savefig(fname, dpi=160); plt.close()
    print("saved", fname)


def fig_noise(N=3, fname="fig_noise.png"):
    grid = (1e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2)
    res = qn.run(N=N, p2_grid=grid, verbose=False)
    # res rows: (p2, postsel, maxVarSolv, minVarMix, contrast)
    p2 = sorted(set(r[0] for r in res))
    def series(ps): return [next(r[4] for r in res if r[0] == p and r[1] == ps) for p in p2]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(np.array(p2) * 1e3, series(True), "o-", color=BLUE, label="with symmetry post-selection")
    ax.plot(np.array(p2) * 1e3, series(False), "s--", color=RED, label="no post-selection")
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.axhspan(0.5, 1.0, color="0.9", alpha=0.6)
    ax.text(0.15, 0.72, "unresolved", fontsize=8, color="0.4")
    ax.set_xlabel("two-qubit gate error (× $10^{-3}$)")
    ax.set_ylabel(r"contrast  $\mathrm{Var}_{\rm mix}/\mathrm{Var}_{\rm solv}$")
    ax.set_title(r"Static readout survives noise ($N=3$, Aer)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0.5, 7)
    plt.tight_layout(); plt.savefig(fname, dpi=160); plt.close()
    print("saved", fname)


if __name__ == "__main__":
    fig_echo()
    fig_noise()
