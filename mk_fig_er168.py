import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figdata import setup, refine, var, Lvals
N=12  # caption: shown at N=12 for legibility
s,s2,sel,H,C2,L2,nd=setup(N,'er')
ev,V=np.linalg.eigh(H); W=refine(ev,V,[L2,C2,nd])
vc=np.maximum(var(W,C2),1e-16); E=np.einsum('ij,ij->j',W,H@W)
Ls=Lvals(W,L2); pure=vc<1e-6; lam=0.013
fig,ax=plt.subplots(1,2,figsize=(8.4,3.3))
# (a) spectrum colored by log10 Var, rigid-rotor line
logv=np.log10(vc)
sc=ax[0].scatter(Ls,E,c=logv,cmap="viridis_r",s=26,vmin=-14,vmax=2)
Lline=np.arange(0,2*N+1,2); ax[0].plot(Lline, lam*Lline*(Lline+1), "k--", lw=1, label=r"$\lambda L(L+1)$")
ax[0].set_xlabel(r"angular momentum $L$"); ax[0].set_ylabel(r"$E$ (MeV)")
ax[0].set_title(r"(a) spectrum coloured by $\log_{10}\mathrm{Var}\,\hat C_2$", fontsize=9)
ax[0].set_ylim(-0.15, min(E.max(),6)); ax[0].legend(fontsize=8, frameon=False, loc="upper left")
cb=fig.colorbar(sc,ax=ax[0]); cb.set_label(r"$\log_{10}\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$",fontsize=8)
# (b) bimodal histogram
ax[1].hist(logv, bins=45, color="#3b6ea5", alpha=0.85)
ax[1].axvline(-6, color="k", ls=":", lw=1)
ax[1].text(-5.6, ax[1].get_ylim()[1]*0.9, r"threshold $10^{-6}$", fontsize=8, va="top")
ax[1].set_xlabel(r"$\log_{10}\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$"); ax[1].set_ylabel("eigenstate count")
ax[1].set_title(r"(b) bimodal: %d exact-label vs %d mixed"%(int(pure.sum()),int((~pure).sum())), fontsize=9)
plt.tight_layout(); plt.savefig("fig_er168_anchor.png", dpi=150, bbox_inches="tight")
print(f"saved. pure={int(pure.sum())} mixed={int((~pure).sum())}")
