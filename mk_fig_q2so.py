import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ibm_pds import IBMSpace
from critical import H_second_order, C2_O5
from figdata import sd_entropy_all
N=10
s=IBMSpace(N); sel=np.where(s.Mvals==0)[0]
Rr=lambda X: X.tocsr()[sel][:,sel].toarray()
H,s2=H_second_order(s, A=1.0)
Hb=Rr(H); O5=Rr(C2_O5(s))
ev,V=np.linalg.eigh(Hb)
# Var C2[O(5)] per eigenstate
O5V=O5@V
vO5=np.einsum('ij,ij->j',O5V,O5V)-np.einsum('ij,ij->j',V,O5V)**2
# tau per state
o5m=np.einsum('ij,ij->j',V,O5@V); tau=np.round((-3+np.sqrt(9+4*o5m))/2).astype(int)
# refine within degenerate energy windows so entropy is on eigenstates (order L2? just tau grouping)
S=sd_entropy_all(s,sel,V)
maxv=vO5.max()
fig,ax=plt.subplots(1,2,figsize=(8.0,3.1))
RED="#c0392b"
# A: Var C2[O(5)] vs state index (all near zero)
idx=np.argsort(ev)
ax[0].scatter(np.arange(len(ev)), np.maximum(vO5[idx],1e-13), s=14, c=RED)
ax[0].set_yscale("log"); ax[0].set_ylim(1e-13,1e-8)
ax[0].axhline(maxv, ls=":", c="k", lw=1)
ax[0].text(2, maxv*1.3, r"$\max_\psi=%.1f\times10^{-11}$"%(maxv/1e-11), fontsize=8, va="bottom")
ax[0].set_xlabel("eigenstate index (sorted by $E$)")
ax[0].set_ylabel(r"$\mathrm{Var}\,\hat C_2[\mathrm{O(5)}]$")
ax[0].set_title(r"(a) exact $\tau$ on every state", fontsize=10)
# B: <S> vs tau with band
taus=sorted(set(tau))
means=[S[tau==t].mean() for t in taus]
mins=[S[tau==t].min() for t in taus]; maxs=[S[tau==t].max() for t in taus]
ax[1].fill_between(taus, mins, maxs, color=RED, alpha=0.15)
ax[1].plot(taus, means, "o-", color=RED, ms=5, lw=1.5)
ax[1].set_xlabel(r"O(5) seniority $\tau$"); ax[1].set_ylabel(r"$\langle S_{s|d}\rangle$ within $\tau$")
ax[1].set_title(r"(b) entanglement graded by $\tau$", fontsize=10)
plt.tight_layout(); plt.savefig("fig_q2_secondorder_N10.png", dpi=150, bbox_inches="tight")
print(f"saved. maxVarO5={maxv:.2e}  <S>(tau=0)={S[tau==0].mean():.3f} (tau=6)={S[tau==6].mean():.3f} (tau=N)={S[tau==N].mean():.3f}")
