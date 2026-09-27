"""
Quantum-simulation half of the PDS+QI project (Sec. X), state-preparation part.

- block(N): the N=... first-order critical M=0 Hamiltonian block plus the
  C2[SU(3)], L^2 and n_d operators, from ibm_pds / critical.
- embed(mat): pad the d-dimensional block into ceil(log2 d) qubits, unphysical
  levels shifted above the spectrum.
- vqd(...): VQE for the ground state + variational quantum deflation (VQD) for
  the excited states, on the Qiskit Aer statevector simulator, with a
  hardware-efficient real (Ry-only) ansatz and an orthogonality penalty.
- degeneracy_resolve_prepared(...): the crux of the section. Within each
  degenerate energy window of the *prepared* states, rediagonalize C2[SU(3)]
  (L^2, then C2, then n_d) so the label variance is read on individual
  eigenstates rather than on an arbitrary VQD superposition; this recovers the
  sharp solvable/mixed split (manuscript Fig. 7a, red).

Running this module prepares all ten eigenstates of the N=3 first-order critical
block and prints max|dE| against exact diagonalization (target < 1e-4). A
statevector simulation of a classically diagonalizable model is not superior to
exact diagonalization: this is a state-preparation and readout demonstration,
with no claim of quantum advantage.
"""
import numpy as np, math
from scipy.optimize import minimize
import ibm_pds as m, critical as cr
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
from qiskit.circuit import QuantumCircuit, ParameterVector
rng=np.random.default_rng(11)

def embed(mat,shift_pad=None):
    d=mat.shape[0]; n=max(1,math.ceil(math.log2(d))); D=2**n
    big=np.zeros((D,D),complex); big[:d,:d]=mat
    sh=shift_pad if shift_pad is not None else (np.linalg.eigvalsh(mat).real.max()+20.0)
    for k in range(d,D): big[k,k]=sh
    return big,n,D,sh

def block(N):
    spN=m.IBMSpace(N); spN2=m.IBMSpace(N-2)
    H=cr.H_first_order(spN,spN2,math.sqrt(2),1.0).toarray().real
    C2=spN.C2_SU3().toarray().real; L2=spN.L2().toarray().real; nd=spN.n_d().toarray().real
    M0=np.where(np.abs(spN.Mvals)<1e-9)[0]
    ix=np.ix_(M0,M0)
    return H[ix],C2[ix],L2[ix],nd[ix]

def refine(Hb,C2b,L2b,ndb):
    e,V=np.linalg.eigh(Hb); i=0
    while i<len(e):
        j=i
        while j+1<len(e) and e[j+1]-e[i]<1e-7: j+=1
        if j>i:
            sub=V[:,i:j+1]
            for op in (L2b,C2b,ndb):
                w,u=np.linalg.eigh(sub.conj().T@op@sub); sub=sub@u
            V[:,i:j+1]=sub
        i=j+1
    return e,V

def ansatz(n,reps):
    p=ParameterVector('t',n*(reps+1)); qc=QuantumCircuit(n); idx=0
    for q in range(n): qc.ry(p[idx],q); idx+=1
    for r in range(reps):
        for q in range(n-1): qc.cx(q,q+1)
        qc.cx(n-1,0)
        for q in range(n): qc.ry(p[idx],q); idx+=1
    return qc,p

def sv(qc,x,p): 
    b=qc.assign_parameters({p[i]:x[i] for i in range(len(p))})
    return Statevector.from_instruction(b).data

def vqd(Hbig,n,kk,reps=5,beta=80.0,restarts=8,maxiter=1500):
    qc,p=ansatz(n,reps); npar=len(p); found=[]
    for k in range(kk):
        best=None
        for _ in range(restarts):
            x0=rng.uniform(-math.pi,math.pi,npar)
            def cost(x):
                psi=sv(qc,x,p); E=np.real(np.vdot(psi,Hbig@psi))
                pen=sum(beta*abs(np.vdot(phi,psi))**2 for _,phi in found)
                return E+pen
            r=minimize(cost,x0,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-13,'gtol':1e-10})
            psi=sv(qc,r.x,p); E=np.real(np.vdot(psi,Hbig@psi))
            pen=sum(beta*abs(np.vdot(phi,psi))**2 for _,phi in found)
            if best is None or E+pen<best[0]: best=(E+pen,psi,E)
        found.append((best[2],best[1]))
    return found

if __name__=="__main__":
    N=3
    Hb,C2b,L2b,ndb=block(N); d=Hb.shape[0]
    Hbig,n,D,sh=embed(Hb)
    e_ex,V_ex=refine(Hb,C2b,L2b,ndb)
    C2sq=C2b@C2b; ndsq=ndb@ndb
    print(f"N={N} first-order critical, M=0 block dim={d}, qubits={n}")
    found=vqd(Hbig,n,d,reps=6)
    found=sorted(found,key=lambda t:t[0])
    dEs=[]; rows=[]
    for i,(E,psi) in enumerate(found):
        phys=psi[:d]; phys=phys/math.sqrt(np.vdot(phys,phys).real)
        vc=np.vdot(phys,C2sq@phys).real-np.vdot(phys,C2b@phys).real**2
        vn=np.vdot(phys,ndsq@phys).real-np.vdot(phys,ndb@phys).real**2
        j=np.argmin(np.abs(e_ex-E)); dE=abs(E-e_ex[j]); dEs.append(dE)
        cls='solv' if (abs(vc)<1e-5 or abs(vn)<1e-5) else 'MIX'
        rows.append((i,E,e_ex[j],dE,vc,vn,cls))
    print(" k |  E_VQD    E_exact   |dE|     VarC2      Varnd    class")
    for (i,E,Ee,dE,vc,vn,cls) in rows:
        print(f" {i:2d}| {E:8.4f} {Ee:8.4f} {dE:.1e} {vc:9.2e} {vn:9.2e}  {cls}")
    print(f"\n max |dE| over all prepared states = {max(dEs):.2e}  (target < 1e-4: {'PASS' if max(dEs)<1e-4 else 'FAIL'})")
    nsolv=sum(1 for r in rows if r[6]=='solv')
    print(f" solvable(prepared)={nsolv}, mixed(prepared)={d-nsolv}  (exact: 8 solvable, 2 mixed)")


def degeneracy_resolve_prepared(found, d, C2b, L2b, ndb):
    """Group VQD states by energy; within each degenerate manifold, rediagonalize
    C2[SU(3)] (as one would to assign a symmetry label). Returns per-state VarC2."""
    import numpy as np, math
    C2sq=C2b@C2b
    states=[]; energies=[]
    for E,psi in found:
        phys=psi[:d]; phys=phys/math.sqrt(np.vdot(phys,phys).real)
        states.append(phys); energies.append(E)
    order=np.argsort(energies); energies=np.array(energies)[order]
    states=[states[i] for i in order]
    varlist=[]; i=0
    while i<len(energies):
        j=i
        while j+1<len(energies) and energies[j+1]-energies[i]<1e-6: j+=1
        M=np.column_stack(states[i:j+1])  # d x g
        # orthonormalize the prepared manifold
        Q,_=np.linalg.qr(M)
        # rediagonalize C2 within it
        sub=Q
        for op in (L2b,C2b,ndb):
            w,u=np.linalg.eigh(sub.conj().T@op@sub); sub=sub@u
        for c in range(sub.shape[1]):
            v=sub[:,c]
            vc=np.vdot(v,C2sq@v).real-np.vdot(v,C2b@v).real**2
            varlist.append(max(vc,0))
        i=j+1
    return np.array(varlist)
