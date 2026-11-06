"""mk_fig_classII.py -- Fig. 1: Var C2[SU(3)] along the two bands terminated by the Class II states.
Reads classII_bands.json written by classII_analytic.py."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Inter"
plt.rcParams.update({"mathtext.fontset": "dejavusans", "axes.linewidth": 0.8})
RED, BLUE = "#c0392b", "#2c7fb8"

d = json.load(open("classII_bands.json"))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
titles = {"stable_N10": r"(a) stable point, $N=10$", "er168_N16": r"(b) $^{168}$Er fit, $N=16$"}
for ax, tag in zip(axs, ("stable_N10", "er168_N16")):
    for name, col, mk, lab in (("K0_2", BLUE, "o", r"$K=0_2$ band (even $L$)"),
                               ("odd", RED, "s", r"lowest odd-$L$ non-tower band")):
        rows = d[tag][name]
        L = [r[0] for r in rows]; v = [r[2] for r in rows]
        ax.plot(L, v, "-", color=col, lw=1.2)
        ax.plot(L[:-1], v[:-1], mk, color=col, ms=4, label=lab)
        ax.plot(L[-1:], v[-1:], mk, mfc="white", mec=col, mew=1.4, ms=6.5)
    N = int(tag.split("N")[-1])
    ax.set_title(titles[tag], fontsize=9.5, loc="left")
    ax.set_xlabel(r"$L$")
    ax.set_xlim(-1, 2 * N - 2)
    top = max(r[2] for r in d[tag]["odd"] + d[tag]["K0_2"]) * 1.08
    ax.set_ylim(-0.04 * top, top)
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.tick_params(labelsize=8.5)
axs[0].set_ylabel(r"Var $\hat C_2[\mathrm{SU(3)}]$")
axs[0].plot([], [], "o", mfc="white", mec="#555555", mew=1.4, ms=6, label="Class II state (Var = 0)")
axs[0].legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("fig_classII.png", dpi=300)
print("wrote fig_classII.png")
