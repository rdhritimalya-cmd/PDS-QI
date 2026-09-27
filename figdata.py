import numpy as np, math
from ibm_pds import IBMSpace, build_H, f2, su3_irreps_in_UN
from critical import H_first_order, C2_O5
from run_step0 import rdm_modes, vn_entropy

def R(space, X, sel): return X.tocsr()[sel][:,sel].toarray()

def refine(ev,V,ops,tol=1e-9):
    V=V.copy(); n=len(ev)
    def rec(sub,ops):
        if not ops or sub.shape[1]==1: return sub
        w,u=np.linalg.eigh(sub.T@ops[0]@sub); sub=sub@u
        out=[];k=0
        while k<len(w):
            l=k
            while l+1<len(w) and w[l+1]-w[k]<tol*max(1,abs(w[k])): l+=1
            out.append(rec(sub[:,k:l+1],ops[1:])); k=l+1
        return np.hstack(out)
    i=0
    while i<n:
        j=i
        while j+1<n and ev[j+1]-ev[i]<tol: j+=1
        if j>i: V[:,i:j+1]=rec(V[:,i:j+1],ops)
        i=j+1
    return V

def var(V,A):
    AV=A@V; return np.einsum('ij,ij->j',AV,AV)-np.einsum('ij,ij->j',V,AV)**2

def Lvals(V,L2):
    l2=np.einsum('ij,ij->j',V,L2@V); return np.round((-1+np.sqrt(1+4*l2))/2).astype(int)

def sd_entropy_all(space, sel, W):
    """s|d von Neumann entropy for each column of W (block-restricted vectors)."""
    S=[]
    for c in range(W.shape[1]):
        full=np.zeros(space.dim,dtype=complex); full[sel]=W[:,c]
        S.append(vn_entropy(rdm_modes(full, space.states, (0,))))
    return np.array(S)

def setup(N, kind, h0=0.35, h2=1.0, C=0.0523):
    s=IBMSpace(N); s2=IBMSpace(N-2); sel=np.where(s.Mvals==0)[0]
    RR=lambda X: R(s,X,sel)
    if kind=='stable': H=RR(build_H(s,s2,h0,h2,C))
    elif kind=='first': H=RR(H_first_order(s,s2,math.sqrt(2)))
    elif kind=='er': H=RR(build_H(s,s2,0.008,0.004,0.013))
    C2=RR(s.C2_SU3()); L2=RR(s.L2()); nd=RR(s.n_d())
    return s,s2,sel,H,C2,L2,nd
