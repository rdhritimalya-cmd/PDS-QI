import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ibm_pds import IBMSpace
from kremer import build, C2_O6
from run_step0 import rdm_modes, vn_entropy
N=12
spN,sel,nd_op,Qchi=build(N)
nd=nd_op.tocsr()[sel][:,sel].toarray()
C2o6_full=C2_O6(spN); C2o6=C2o6_full.tocsr()[sel][:,sel].toarray()
chi=-np.sqrt(7)/2
def Qdot(chi):
    Q=Qchi(chi); return sum(((-1)**q)*(Q[q]@Q[-q]) for q in range(-2,3)).tocsr()[sel][:,sel].toarray()
QQ=Qdot(chi)
w,U=np.linalg.eigh(C2o6); sig=np.round(-2+np.sqrt(4+np.round(w,6))).astype(int)
def purity_ent(xi):
    H=(1-xi)*nd-(xi/(4*N))*QQ
    ev,V=np.linalg.eigh(H); gs=V[:,0]
    c=U.T@gs; P={}
    for sg in np.unique(sig):
        p=float(np.sum(c[sig==sg]**2))
        if p>1e-12: P[sg]=P.get(sg,0)+p
    pur=sum(p**2 for p in P.values())
    full=np.zeros(spN.dim,dtype=complex); full[sel]=gs
    S=vn_entropy(rdm_modes(full,spN.states,(0,)))
    return pur,S
xis=np.linspace(0,1,41)
pur=[];S=[]
for xi in xis:
    p,s=purity_ent(xi); pur.append(p); S.append(s)
pur=np.array(pur); S=np.array(S); Snorm=S/np.log(2)  # bits-ish; caption uses "maximal"
fig,ax=plt.subplots(1,3,figsize=(9.8,3.1))
NAVY="#2c3e70"; RED="#c0392b"
# left: O(6) block purity vs xi
ax[0].plot(xis,pur,"-",color=NAVY,lw=1.8)
ax[0].set_xlabel(r"$\xi$ (U(5)$\to$SU(3))"); ax[0].set_ylabel(r"$\mathrm{Tr}\,\rho_\sigma^2$")
ax[0].set_title(r"(a) O(6) block purity", fontsize=10)
# centre: s|d entanglement vs xi
ax[1].plot(xis,S,"-",color=RED,lw=1.8)
ax[1].set_xlabel(r"$\xi$ (U(5)$\to$SU(3))"); ax[1].set_ylabel(r"$S_{s|d}$")
ax[1].set_title(r"(b) $s|d$ entanglement", fontsize=10)
# right: independence scatter, colored by xi
sc=ax[2].scatter(pur,S,c=xis,cmap="plasma",s=22)
ax[2].set_xlabel(r"$\mathrm{Tr}\,\rho_\sigma^2$"); ax[2].set_ylabel(r"$S_{s|d}$")
ax[2].set_title(r"(c) independent axes", fontsize=10)
cb=fig.colorbar(sc,ax=ax[2]); cb.set_label(r"$\xi$",fontsize=9)
plt.tight_layout(); plt.savefig("fig_q3_kremer_N12.png", dpi=150, bbox_inches="tight")
# report purity where entanglement maximal
imax=np.argmax(S)
print(f"saved. max S at xi={xis[imax]:.2f}, purity there={pur[imax]:.2f}; purity range near max={pur[S>0.9*S.max()].min():.2f}-{pur[S>0.9*S.max()].max():.2f}")
