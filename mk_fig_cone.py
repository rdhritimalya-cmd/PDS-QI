import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rigidity_cone import analyze
N=10
d=analyze(N)
R_h0,R_h4,R_nd,cone,pure,ev=d["R_h0"],d["R_h4"],d["R_nd"],d["cone"],d["pure"],d["ev"]
E=ev
flr=lambda x: np.maximum(x,1e-13)
RED="#c0392b"; ORANGE="#e08214"; GREY="#8a8a8a"
fig,ax=plt.subplots(1,2,figsize=(8.6,3.3))
# A: rigidity vs E; circles=h0, triangles=h4; pure (red/orange) low band, mixed (grey) high band
mix=~pure
ax[0].scatter(E[mix], flr(R_h0[mix]), s=14, c=GREY, marker="o", alpha=.7)
ax[0].scatter(E[mix], flr(R_h4[mix]), s=16, c=GREY, marker="^", alpha=.7, label="mixed")
ax[0].scatter(E[pure], flr(R_h0[pure]), s=20, c=RED, marker="o", label=r"pure, $h_0$")
ax[0].scatter(E[pure], flr(R_h4[pure]), s=24, c=ORANGE, marker="^", label=r"pure, $h_4$")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-13,1e4)
ax[0].axhspan(1e-13,1e-8,color=RED,alpha=0.06)
ax[0].set_xlabel(r"$E/h_2$"); ax[0].set_ylabel(r"off-diagonal coupling $R_D$")
ax[0].set_title(r"(a) rigidity along $h_0$, $h_4$", fontsize=9.5)
ax[0].legend(fontsize=7.5, frameon=False, loc="center right")
# B: direction-selectivity: R along PDS (h0) vs R along breaking (nd)
ax[1].scatter(flr(R_h0[mix]), R_nd[mix], s=16, c=GREY, label="mixed")
ax[1].scatter(flr(R_h0[pure]), R_nd[pure], s=20, c=RED, label="symmetry-pure")
ax[1].set_xscale("log"); ax[1].set_xlim(1e-13,1e3)
ax[1].axvline(1e-8, ls="--", c="k", lw=1, alpha=.6)
ax[1].set_xlabel(r"$R$ along PDS direction $h_0$  (rigid $\Rightarrow$ left)")
ax[1].set_ylabel(r"$R$ along SU(3)-breaking $\hat n_d$")
ax[1].set_title(r"(b) direction selectivity", fontsize=9.5)
ax[1].legend(fontsize=8, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig("fig_cone_N10.png", dpi=150, bbox_inches="tight")
print("saved fig_cone_N10.png")
