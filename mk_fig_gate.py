import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figdata import setup, refine, var, Lvals, sd_entropy_all
N=10
s,s2,sel,H,C2,L2,nd=setup(N,'stable')
ev,V=np.linalg.eigh(H); W=refine(ev,V,[L2,C2,nd])
vc=var(W,C2); pure=vc<1e-6
E=np.einsum('ij,ij->j',W,H@W)
S=sd_entropy_all(s,sel,W); Ls=Lvals(W,L2)
fig,ax=plt.subplots(1,2,figsize=(8.0,3.2))
RED="#c0392b"; GREY="#8a8a8a"
# A: static s|d entropy vs E, no separation
ax[0].scatter(E[~pure], S[~pure], s=16, c=GREY, label="mixed")
ax[0].scatter(E[pure], S[pure], s=18, c=RED, label="symmetry-pure")
ax[0].set_xlabel(r"$E/h_2$"); ax[0].set_ylabel(r"$S_{s|d}$")
ax[0].set_title(r"(a) static entropy: distributions overlap", fontsize=9.5)
ax[0].legend(fontsize=8, frameon=False, loc="lower left")
# B: fixed-L means, pure above mixed
Ls_present=sorted(set(Ls[Ls>=0]))
Lp=[]; mp=[]; mm=[]
for L in Ls_present:
    ap=pure&(Ls==L); am=(~pure)&(Ls==L)
    if ap.sum() and am.sum():
        Lp.append(L); mp.append(S[ap].mean()); mm.append(S[am].mean())
ax[1].plot(Lp, mp, "o-", color=RED, ms=5, lw=1.5, label="symmetry-pure mean")
ax[1].plot(Lp, mm, "s-", color=GREY, ms=5, lw=1.5, label="mixed mean")
ax[1].set_xlabel(r"angular momentum $L$"); ax[1].set_ylabel(r"$\langle S_{s|d}\rangle$ within $L$")
ax[1].set_title(r"(b) fixed $L$: pure states more entangled", fontsize=9.5)
ax[1].legend(fontsize=8, frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig("fig_gate_N10.png", dpi=150, bbox_inches="tight")
print("saved fig_gate_N10.png; L points:",len(Lp))
