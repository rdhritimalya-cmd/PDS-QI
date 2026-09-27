import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figdata import setup, refine, var, Lvals, sd_entropy_all, f2
N=10
s,s2,sel,H,C2,L2,nd=setup(N,'first')
ev,V=np.linalg.eigh(H); W=refine(ev,V,[L2,C2,nd])
vc=var(W,C2); vn=var(W,nd)
E=np.einsum('ij,ij->j',W,H@W)          # energies (H(beta0=sqrt2), h2=1 units => E/h2)
ndm=np.einsum('ij,ij->j',W,nd@W)
pure=vc<1e-6
S=sd_entropy_all(s,sel,W)
npure=int(pure.sum()); nmix=int((~pure).sum())
# band head <nd>/N for the deformed L=0 (lowest E pure state with L=0)
Ls=Lvals(W,L2)
l0=[i for i in range(len(ev)) if Ls[i]==0 and pure[i]]
bh=ndm[min(l0,key=lambda i:E[i])]/N if l0 else np.nan

fig,ax=plt.subplots(1,3,figsize=(9.6,3.1))
RED="#c0392b"; GREY="#8a8a8a"
# A: Var C2[SU(3)]
ax[0].scatter(E[~pure], np.maximum(vc[~pure],1e-13), s=16, c=GREY, label=f"mixed ({nmix})")
ax[0].scatter(E[pure], np.maximum(vc[pure],1e-13), s=18, c=RED, label=f"exact-label ({npure})")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-13,1e4)
ax[0].set_xlabel(r"$E/h_2$"); ax[0].set_ylabel(r"$\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$")
ax[0].set_title(r"(a) $\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$", fontsize=10)
ax[0].legend(fontsize=7.5, frameon=False, loc="center right")
# B: Var n_d
ax[1].scatter(E[~pure], np.maximum(vn[~pure],1e-13), s=16, c=GREY)
ax[1].scatter(E[pure], np.maximum(vn[pure],1e-13), s=18, c=RED)
ax[1].set_yscale("log"); ax[1].set_ylim(1e-13,1e2)
ax[1].set_xlabel(r"$E/h_2$"); ax[1].set_ylabel(r"$\mathrm{Var}\,\hat n_d$")
ax[1].set_title(r"(b) $\mathrm{Var}\,\hat n_d$", fontsize=10)
# C: S_sd vs E, colored by class
ax[2].scatter(E[~pure], S[~pure], s=16, c=GREY, label="mixed")
ax[2].scatter(E[pure], S[pure], s=18, c=RED, label="exact-label")
ax[2].set_xlabel(r"$E/h_2$"); ax[2].set_ylabel(r"$S_{s|d}$")
ax[2].set_title(r"(c) coexistence at all energies", fontsize=10)
ax[2].legend(fontsize=7.5, frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("fig_q2_N10.png", dpi=150, bbox_inches="tight")
print(f"saved fig_q2_N10.png  npure={npure} nmix={nmix} bandhead<nd>/N={bh:.2f}")
