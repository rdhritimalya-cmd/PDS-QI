"""Figure: real ibm_kingston device readout of the label variance (solvable vs mixed),
with conservative error bars. Measured bars with exact-value reference lines. Repo figure style."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE="#2c7fb8"; RED="#c0392b"; GREY="#888888"

def make(src="kingston_device_N3.json", fname="fig_kingston.png"):
    d=json.load(open(src))
    S=[s for s in d["states"] if s["kind"]=="solvable"][0]
    M=[s for s in d["states"] if s["kind"]=="mixed"][0]
    meas=[S["VarC2"], M["VarC2"]]; err=[S["VarC2_err"], M["VarC2_err"]]
    exact=[S["exact_labvar"], M["exact_labvar"]]
    fig,ax=plt.subplots(figsize=(5.0,3.8))
    x=[0,1]
    ax.bar(x, meas, width=0.55, color=[BLUE,RED], zorder=2)
    ax.errorbar(x, meas, yerr=err, fmt="none", ecolor="k", elinewidth=1.3,
                capsize=5, zorder=3)
    # exact reference lines
    ax.hlines(exact[1], x[1]-0.33, x[1]+0.33, color="k", ls="--", lw=1.3, zorder=4)
    ax.text(x[1]-0.37, exact[1]+6, f"exact = {exact[1]:.0f}", ha="left", fontsize=8)
    ax.hlines(0.0, x[0]-0.33, x[0]+0.33, color="k", ls="--", lw=1.3, zorder=4)
    ax.text(x[0]+0.37, 4, "exact = 0", ha="left", fontsize=8)
    # value +/- labels
    ax.text(x[0], meas[0]+err[0]+8, f"{meas[0]:.0f}$\\pm${err[0]:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(x[1], meas[1]+err[1]+8, f"{meas[1]:.0f}$\\pm${err[1]:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["solvable\n(consistent with 0)","mixed\n($\\sim$3$\\sigma$ nonzero)"])
    ax.set_ylabel(r"$\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$  (measured)")
    ax.set_ylim(-120, 345)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("ibm\\_kingston, 4000 shots: point estimates match exact values",
                 fontsize=9.5)
    plt.tight_layout(); plt.savefig(fname, dpi=160); plt.close(); print("saved",fname)

if __name__=="__main__":
    make()
