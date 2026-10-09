"""Figure: (a) scaling separation (pure residual vs mixed band vs N),
(b) approximate-PDS robustness (Var ~ eps^2). Matches repo figure style."""
import numpy as np, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import robustness as rb

BLUE="#2c7fb8"; RED="#c0392b"; GREY="#888888"

def make(fname="fig_robustness.png"):
    rows,_=rb.scaling_table(verbose=False)
    N=[r["N"] for r in rows]; mp=[r["max_pure"] for r in rows]
    mn=[r["min_mix"] for r in rows]; md=[r["med_mix"] for r in rows]
    pert,_=rb.perturbation(N=10,verbose=False)
    eps=np.array([p[0] for p in pert]); mv=np.array([p[1] for p in pert])

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.4,3.8))
    # (a) scaling
    ax1.semilogy(N,mp,"o-",color=BLUE,label="max pure-state residual")
    ax1.semilogy(N,mn,"s-",color=RED,label="min mixed-state variance")
    ax1.semilogy(N,md,"^--",color=GREY,label="median mixed variance")
    ax1.set_xlabel("boson number $N$")
    ax1.set_ylabel(r"$\mathrm{Var}\,\hat C_2[\mathrm{SU(3)}]$")
    ax1.set_title("(a) Separation does not close",fontsize=10)
    ax1.legend(fontsize=7.5,frameon=False,loc="center left")
    ax1.set_ylim(1e-11,1e4)
    # (b) perturbation
    m=eps>0
    ax2.loglog(eps[m],mv[m],"o-",color=RED,label=r"mean Var of pure states")
    g=eps[m]; ax2.loglog(g, mv[m][0]*(g/g[0])**2,"--",color="k",alpha=0.6,lw=1,label=r"$\propto\epsilon^2$")
    ax2.axhline(rows[3]["min_mix"],color=BLUE,ls=":",lw=1)
    ax2.text(2e-3, rows[3]["min_mix"]*1.3, "mixed band (min)",fontsize=7.5,color=BLUE)
    ax2.set_xlabel(r"SU(3)-breaking strength $\epsilon$  ($H_{\rm PDS}+\epsilon\,\hat n_d$)")
    ax2.set_ylabel(r"$\langle\mathrm{Var}\,\hat C_2\rangle$ (formerly-pure)")
    ax2.set_title("(b) Approximate PDS: smooth, bounded",fontsize=10)
    ax2.legend(fontsize=7.5,frameon=False,loc="lower right")
    plt.tight_layout(); plt.savefig(fname,dpi=160); plt.close(); print("saved",fname)

if __name__=="__main__":
    make()
