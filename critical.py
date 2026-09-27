"""
Critical-point PDS Hamiltonians  (Leviatan, PRL 98, 242502; nucl-th/0703048).

First-order critical (spherical <-> prolate), Eq. (7):
    H(beta0) = h2 * P2dag(beta0) . P2til(beta0)
    P2mu^dag(beta0) = beta0 s^dag d_mu^dag + sqrt(7/2) (d^dag d^dag)^(2)_mu
  Solvable:
    - deformed ground band |beta0; N, L>, E=0, L=0,2,...,2N          (Eq. 8)
    - spherical U(5) states |nd=tau=L=0>, E=0                         (Eq. 9a)
                            |nd=tau=L=3>, E=3 h2 [beta0^2 (N-3)+5]    (Eq. 9b)
    At beta0=sqrt(2): additional SU(3) solvable gamma^k bands (Eq. 10) ->
      this Hamiltonian is a special case of the PRL-77 SU(3)-PDS. CROSS-CHECK.

Second-order critical (spherical <-> gamma-soft, U(5)<->O(6)), Eq. (6):
    H = eps*nd + A[(d^dag.d^dag - (s^dag)^2) + h.c.],  eps = 4(N-1)A
  Preserves O(5) (good tau) for ALL states -> O(5) PDS type II.

Relation of P2dag(beta0) to the code's P2dag (from 2010.10951 / PRL-77):
    PRL-77 P2mu^dag = 2 s^dag d_mu^dag + sqrt(7)(d^dag d^dag)^(2)_mu
          = sqrt(2) * [ sqrt(2) s^dag d_mu^dag + sqrt(7/2)(d^dag d^dag)^(2)_mu ]
          = sqrt(2) * P2mu^dag(beta0=sqrt(2)).
  So H_PRL77(h2) = h2 P2^dag.P2  (code) = 2 h2 * H(beta0=sqrt2).  [checked below]

We build P2dag(beta0) directly for general beta0 and reuse the s,d Fock machinery.
"""
import numpy as np, scipy.sparse as sp, math
from ibm_pds import IBMSpace, cg

def P2dag_beta(spN, spN2, beta0):
    """(dim_N x dim_{N-2}) matrix for P2mu^dag(beta0) summed as tensor? 
    We need each mu separately for the scalar product. Return dict mu-> matrix."""
    idxN = spN.index
    out = {}
    def add_two(st, i, j):
        lst = list(st); amp = math.sqrt(lst[j] + 1); lst[j] += 1
        amp *= math.sqrt(lst[i] + 1); lst[i] += 1
        return tuple(lst), amp
    for mu in range(-2, 3):
        rows, cols, vals = [], [], []
        for col, st in enumerate(spN2.states):
            # beta0 s^dag d_mu^dag
            ns, amp = add_two(st, 0, spN.mode_index(mu))
            rows.append(idxN[ns]); cols.append(col); vals.append(beta0 * amp)
            # sqrt(7/2) (d^dag d^dag)^(2)_mu
            for m1 in (-2,-1,0,1,2):
                m2 = mu - m1
                if -2 <= m2 <= 2:
                    c = cg(2, m1, 2, m2, 2, mu)
                    if c != 0.0:
                        ns, amp = add_two(st, spN.mode_index(m1), spN.mode_index(m2))
                        rows.append(idxN[ns]); cols.append(col)
                        vals.append(math.sqrt(3.5) * c * amp)
        out[mu] = sp.csr_matrix((vals, (rows, cols)), shape=(spN.dim, spN2.dim))
    return out

def H_first_order(spN, spN2, beta0, h2=1.0):
    P2 = P2dag_beta(spN, spN2, beta0)
    H = sp.csr_matrix((spN.dim, spN.dim))
    for mu in range(-2, 3):
        H = H + h2 * (P2[mu] @ P2[mu].conj().T)
    return H.tocsr()

def H_second_order(spN, A=1.0):
    """H = eps nd + A[(d.d - s^2) + h.c.], eps=4(N-1)A. Uses O(6) pairing.
    (d^dag.d^dag - (s^dag)^2) = P0^dag with the PRL-77 sign? P0^dag = d.d - 2 s^2.
    Here it's d.d - s^2 (O(6) Casimir related). Build directly."""
    N = spN.N
    eps = 4 * (N - 1) * A
    # O(6)-type pair: B^dag = d^dag.d^dag - (s^dag)^2 : maps N-2 -> N
    spN2 = IBMSpace(N - 2)
    idxN = spN.index
    def add_two(st, i, j):
        lst = list(st); amp = math.sqrt(lst[j] + 1); lst[j] += 1
        amp *= math.sqrt(lst[i] + 1); lst[i] += 1
        return tuple(lst), amp
    rows, cols, vals = [], [], []
    for col, st in enumerate(spN2.states):
        for m in (-2,-1,0,1,2):
            ns, amp = add_two(st, spN.mode_index(m), spN.mode_index(-m))
            rows.append(idxN[ns]); cols.append(col); vals.append(((-1)**m) * amp)
        ns, amp = add_two(st, 0, 0)
        rows.append(idxN[ns]); cols.append(col); vals.append(-1.0 * amp)
    Bdag = sp.csr_matrix((vals, (rows, cols)), shape=(spN.dim, spN2.dim))
    Hpair = A * (Bdag @ Bdag.conj().T)          # A * B^dag B  (+ h.c. already Hermitian as B^dag B + B B^dag? )
    # The paper's A[(dd - s^2)+h.c.] means A(B + B^dag) sandwiched... actually the
    # standard reading: A[ (d^dag.d^dag - s^dag^2)(d.d - s^2 as h.c.) ]? 
    # We take the Hermitian operator A * Bdag @ B (annihilation then creation on N).
    H = eps * spN.n_d() + Hpair
    return H.tocsr(), spN2

def C2_O5(spN):
    """Quadratic Casimir of O(5): tau(tau+3). Built from O(5) generators.
    O(5) generators in sd-IBM: L (ang mom, from d) and the (d^dag dtil)^(3) octupole,
    plus (d^dag dtil)^(1). Standard: C2[O(5)] = sum over the 10 generators.
    Simpler robust route: C2[O5] = (1/2) sum_{k=1,3} (d^dag dtil)^(k).(d^dag dtil)^(k) *norm.
    We use: C2[O(5)] = 2[ (d^dag dtil)^(1).(d^dag dtil)^(1) + (d^dag dtil)^(3).(d^dag dtil)^(3) ]."""
    out = sp.csr_matrix((spN.dim, spN.dim))
    for k in (1, 3):
        for q in range(-k, k+1):
            Tkq = spN.dd_tensor(k, q)
            out = out + ((-1)**q) * (Tkq @ spN.dd_tensor(k, -q))
    return (2.0 * out).tocsr()

if __name__ == "__main__":
    # CROSS-CHECK: H_first_order(beta0=sqrt2) == 0.5 * H_PRL77(h2=1)  (i.e. factor 2)
    from ibm_pds import build_H
    N = 6
    spN = IBMSpace(N); spN2 = IBMSpace(N-2)
    Hfo = H_first_order(spN, spN2, math.sqrt(2.0), h2=1.0).toarray()
    HP77 = build_H(spN, spN2, 1.0, 1.0, 0.0).toarray()  # h0=h2=1 -> full P0P0+P2P2 = C2-based... 
    # careful: build_H uses h0 P0P0 + h2 P2P2. PRL-77 H(h0,h2). The SU(3) P2 there is
    # 2 s d + sqrt7(dd). Our beta0 P2 with beta0=sqrt2 is (1/sqrt2)*that. So
    # h2 P2(sqrt2).P2(sqrt2) = (h2/2) * [P2^PRL77.P2^PRL77].
    # build_H with h0=0,h2=1 gives P2^PRL77.P2^PRL77 term only:
    HP77_p2only = build_H(spN, spN2, 0.0, 1.0, 0.0).toarray()
    ratio = HP77_p2only / np.where(np.abs(Hfo) > 1e-9, Hfo, np.nan)
    print("H_first_order(sqrt2) vs PRL77 P2-only term:")
    print("  max|2*Hfo - HP77_p2only| =", np.nanmax(np.abs(2*Hfo - HP77_p2only)))
    # O(5) Casimir integer check
    c2o5 = C2_O5(spN).toarray()
    ev = np.linalg.eigvalsh(c2o5)
    taus = (-3 + np.sqrt(9 + 4*ev))/2
    print("C2[O(5)] -> tau integer? max dev:", np.max(np.abs(taus - np.round(taus))),
          " tau range", taus.min(), taus.max())
