"""Regenerates the two figures whose content changed in the revised manuscript:
  fig_q2_N10.png     first-order critical point: SU(3)-pure and U(5)-pure solvable states
  fig_robustness.png scaling of the separation and approximate PDS
Degeneracies are resolved by the nested L^2 -> C2 -> n_d (or L^2 -> n_d -> C2) refinement
described in Appendix A of the manuscript. Run from the repository root:

    python make_fig_revision.py
"""
import sys, math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from ibm_pds import IBMSpace, build_H
from critical import H_first_order

if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Inter"
plt.rcParams.update({"mathtext.fontset": "dejavusans", "axes.linewidth": 0.8})
RED, BLUE, GREY = "#c0392b", "#2c7fb8", "#8a8a8a"
FLOOR = 1e-13
H0, H2, CC = 0.35, 1.0, 0.0523


def nested_refine(e, V, ops, tol=1e-7, optol=1e-6):
    V = V.copy()

    def rec(sub, ops):
        if not ops or sub.shape[1] == 1:
            return sub
        w, u = np.linalg.eigh(sub.T @ ops[0] @ sub); sub = sub @ u
        out, k = [], 0
        while k < len(w):
            l = k
            while l + 1 < len(w) and w[l + 1] - w[k] < optol * max(1.0, abs(w[k])):
                l += 1
            out.append(rec(sub[:, k:l + 1], ops[1:])); k = l + 1
        return np.hstack(out)

    i = 0
    while i < len(e):
        j = i
        while j + 1 < len(e) and e[j + 1] - e[i] < tol:
            j += 1
        if j > i:
            V[:, i:j + 1] = rec(V[:, i:j + 1], ops)
        i = j + 1
    return V


def var(A, V):
    AV = A @ V; m = np.einsum("ij,ij->j", V, AV)
    return np.einsum("ij,ij->j", AV, AV) - m ** 2, m


def block(N):
    s, s2 = IBMSpace(N), IBMSpace(N - 2); sel = np.where(s.Mvals == 0)[0]

    def R(X):
        X = X.tocsr()[sel][:, sel].toarray().real; return 0.5 * (X + X.T)
    return s, s2, R, R(s.C2_SU3()), R(s.L2()), R(s.n_d())


def fig_first_order():
    N = 10
    s, s2, R, C2, L2, nd = block(N)
    H = R(H_first_order(s, s2, math.sqrt(2), 1.0))
    e, V0 = np.linalg.eigh(H)
    Va = nested_refine(e, V0, [L2, C2, nd]); vca, _ = var(C2, Va)
    Vb = nested_refine(e, V0, [L2, nd, C2]); vnb, mnb = var(nd, Vb)
    l2b = np.einsum("ij,ij->j", Vb, L2 @ Vb); Lb = np.round((-1 + np.sqrt(1 + 4 * l2b)) / 2).astype(int)
    pa = vca < 1e-6
    u5 = (vnb < 1e-6) & (mnb < N - 0.5)                    # Leviatan's U(5)-solvable states
    triv = (vnb < 1e-6) & ~u5                                # n_d = N states of unique L
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.1))
    ax[0].scatter(e[~pa], np.maximum(vca[~pa], FLOOR), s=14, c=GREY, label=f"SU(3)-mixed ({(~pa).sum()})")
    ax[0].scatter(e[pa], np.maximum(vca[pa], FLOOR), s=16, c=RED, label=f"SU(3)-pure ({pa.sum()})")
    ax[0].set_yscale("log"); ax[0].set_ylim(3e-14, 1e5)
    ax[0].set_xlabel(r"$E/h_2$"); ax[0].set_ylabel(r"$\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$")
    ax[0].set_title(r"(a) SU(3) reading ($L^2\!\to\!\hat C_2\!\to\!\hat n_d$)", fontsize=9.5)
    ax[0].legend(fontsize=7.5, frameon=False, loc="center right")
    rest = ~(u5 | triv)
    ax[1].scatter(e[rest], np.maximum(vnb[rest], FLOOR), s=14, c=GREY, label="other states")
    ax[1].scatter(e[u5], np.maximum(vnb[u5], FLOOR), s=40, c=BLUE, marker="D", label="U(5)-solvable", zorder=3)
    ax[1].scatter(e[triv], np.maximum(vnb[triv], FLOOR), s=110, facecolors="none", edgecolors="k",
                  linewidths=0.8, label=r"$n_d=N$, unique $L$", zorder=4)
    for i in np.where(u5)[0]:
        ax[1].annotate(rf"$n_d=\tau=L={Lb[i]}$", (e[i], FLOOR), xytext=(e[i] + 8, 1e-10), fontsize=7.5,
                       color="k", arrowprops=dict(arrowstyle="-", lw=0.6, color="k"))
    ax[1].set_yscale("log"); ax[1].set_ylim(3e-14, 1e3)
    ax[1].set_xlabel(r"$E/h_2$"); ax[1].set_ylabel(r"$\mathrm{Var}\,\hat n_d$")
    ax[1].set_title(r"(b) U(5) reading ($L^2\!\to\!\hat n_d\!\to\!\hat C_2$)", fontsize=9.5)
    ax[1].legend(fontsize=7.5, frameon=False, loc="center right")
    plt.tight_layout(); plt.savefig("fig_q2_N10.png", dpi=160); plt.close()
    print(f"fig_q2_N10: SU3-pure {pa.sum()}, mixed {(~pa).sum()}, U5-solvable at E/h2={e[u5].round(3)}, L={Lb[u5]}")


def fig_robustness():
    Ns = list(range(4, 17)); mp, mn, md = [], [], []
    for N in Ns:
        s, s2, R, C2, L2, nd = block(N)
        H = R(build_H(s, s2, H0, H2, CC)); e, V = np.linalg.eigh(H); V = nested_refine(e, V, [L2, C2, nd])
        vc, _ = var(C2, V); p = vc < 1e-6
        mp.append(max(vc[p].max(), 1e-16)); mn.append(vc[~p].min()); md.append(np.median(vc[~p]))
        print(f"N={N}: max pure {vc[p].max():.1e} min mixed {vc[~p].min():.1f} median {np.median(vc[~p]):.1f}")
    s, s2, R, C2, L2, nd = block(10)
    H = R(build_H(s, s2, H0, H2, CC)); e0, V0 = np.linalg.eigh(H); V0 = nested_refine(e0, V0, [L2, C2, nd])
    vc0, _ = var(C2, V0); pure_idx = np.where(vc0 < 1e-6)[0]; gb = V0[:, pure_idx]
    eps = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]); mean_p, max_p, min_m = [], [], []
    for x in eps:
        ee, Ve = np.linalg.eigh(H + x * nd); vce, _ = var(C2, Ve)
        tracked = [int(np.argmax(np.abs(Ve.T @ gb[:, c]))) for c in range(gb.shape[1])]
        others = np.setdiff1d(np.arange(len(ee)), tracked)
        mean_p.append(np.mean(vce[tracked])); max_p.append(np.max(vce[tracked])); min_m.append(np.min(vce[others]))
        print(f"eps={x}: tracked unique {len(set(tracked))}/{len(tracked)} mean {mean_p[-1]:.2e} max {max_p[-1]:.2e} min other {min_m[-1]:.2f}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    a1.semilogy(Ns, mn, "s-", color=RED, label="smallest mixed-state value")
    a1.semilogy(Ns, md, "^--", color=GREY, label="median mixed-state value")
    a1.set_xlabel("boson number $N$"); a1.set_ylabel(r"$\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$")
    a1.set_title("(a) SU(3)-mixed states versus $N$", fontsize=10); a1.set_ylim(5, 3e3)
    a1.legend(fontsize=7.5, frameon=False, loc="upper left")
    a2.loglog(eps, mean_p, "o-", color=RED, label="mean over formerly pure states")
    a2.loglog(eps, max_p, "v:", color=RED, alpha=0.7, label="largest formerly pure value")
    a2.loglog(eps, min_m, "s--", color=GREY, label="smallest value among the other states")
    a2.loglog(eps[:5], mean_p[0] * (eps[:5] / eps[0]) ** 2, "-", color="k", lw=0.8, alpha=0.6, label=r"$\propto\epsilon^2$")
    a2.set_xlabel(r"breaking strength $\epsilon$ in $\hat H_{\rm PDS}+\epsilon\,\hat n_d$")
    a2.set_ylabel(r"$\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$")
    a2.set_title(r"(b) Controlled SU(3) breaking, $N=10$", fontsize=10)
    a2.legend(fontsize=7.2, frameon=False, loc="lower right")
    plt.tight_layout(); plt.savefig("fig_robustness.png", dpi=160); plt.close()


if __name__ == "__main__":
    fig_first_order(); fig_robustness()
