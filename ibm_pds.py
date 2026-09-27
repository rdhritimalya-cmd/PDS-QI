"""
Step 0 -- Classical baseline for the PDS + QI project.

sd-IBM in the m-scheme boson Fock basis at fixed total boson number N.
Constructs the SU(3)-PDS Hamiltonian of Leviatan, PRL 77, 818 (1996):

    H(h0, h2) = h0 P0^dag P0 + h2 P2^dag . P2~  (+ C * L.L)

with
    P0^dag   = d^dag . d^dag - 2 (s^dag)^2
    P2mu^dag = 2 s^dag d^dag_mu + sqrt(7) (d^dag d^dag)^(2)_mu

Solvable subset (verified against nucl-th/9606049 and arXiv:2010.10951):
    ground band  (2N, 0),   K=0,  E = C L(L+1)
    gamma^k bands (2N-4k, 2k), K=2k, E = 6 h2 k (2N-2k+1) + C L(L+1)

Convention arbiter (arXiv:2010.10951 Eq. 27a):
    P0^dag P0 + P2^dag . P2~ = -C2[SU(3)] + 2 N^ (2 N^ + 3)
must hold to machine precision.
"""
import numpy as np
import scipy.sparse as sp
from itertools import product
from functools import lru_cache
from fractions import Fraction
import math

# ----------------------------------------------------------------------
# Clebsch-Gordan <j1 m1 j2 m2 | J M> (exact, Racah formula with Fractions)
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _fact(n):
    return math.factorial(n)

@lru_cache(maxsize=None)
def cg(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan coefficient for integer j's (sufficient here)."""
    if m1 + m2 != M:
        return 0.0
    if not (abs(j1 - j2) <= J <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return 0.0
    # Racah formula
    pref = Fraction(
        _fact(j1 + j2 - J) * _fact(j1 - j2 + J) * _fact(-j1 + j2 + J) * (2 * J + 1),
        _fact(j1 + j2 + J + 1),
    )
    pref *= Fraction(
        _fact(J + M) * _fact(J - M) * _fact(j1 - m1) * _fact(j1 + m1)
        * _fact(j2 - m2) * _fact(j2 + m2), 1
    )
    s = Fraction(0)
    kmin = max(0, j2 - J - m1, j1 + m2 - J)
    kmax = min(j1 + j2 - J, j1 - m1, j2 + m2)
    for k in range(kmin, kmax + 1):
        den = (_fact(k) * _fact(j1 + j2 - J - k) * _fact(j1 - m1 - k)
               * _fact(j2 + m2 - k) * _fact(J - j2 + m1 + k) * _fact(J - j1 - m2 + k))
        s += Fraction((-1) ** k, den)
    val = float(s) * math.sqrt(float(pref))
    return val

# ----------------------------------------------------------------------
# Fock basis: occupations (n_s, n_{d,-2}, n_{d,-1}, n_{d,0}, n_{d,1}, n_{d,2})
# ----------------------------------------------------------------------
D_MS = [-2, -1, 0, 1, 2]   # d-boson magnetic substates, index 1..5 in tuple

def build_basis(N):
    """All Fock states with total boson number N. Returns list of tuples and index dict."""
    states = []
    for ns in range(N, -1, -1):
        nd = N - ns
        # partition nd among 5 d modes
        for c in _compositions(nd, 5):
            states.append((ns,) + c)
    index = {st: i for i, st in enumerate(states)}
    return states, index

def _compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for first in range(n, -1, -1):
        for rest in _compositions(n - first, k - 1):
            yield (first,) + rest

def m_of(state):
    return sum(m * n for m, n in zip(D_MS, state[1:]))

# ----------------------------------------------------------------------
# Sparse one-boson operators  a_i^dag a_j  and pair operators on fixed-N space
# ----------------------------------------------------------------------
class IBMSpace:
    MODE_S = 0  # tuple index of s
    def __init__(self, N):
        self.N = N
        self.states, self.index = build_basis(N)
        self.dim = len(self.states)
        self.Mvals = np.array([m_of(st) for st in self.states])
        self._hop_cache = {}

    def mode_index(self, label):
        """label: 's' or d magnetic quantum number m in -2..2"""
        if label == 's':
            return 0
        return 2 + label + 1  # m=-2 -> 1, ..., m=2 -> 5

    def hop(self, i, j):
        """Matrix of a_i^dag a_j (modes by tuple index), N-conserving."""
        key = (i, j)
        if key in self._hop_cache:
            return self._hop_cache[key]
        rows, cols, vals = [], [], []
        for col, st in enumerate(self.states):
            nj = st[j]
            if nj == 0:
                continue
            if i == j:
                rows.append(col); cols.append(col); vals.append(float(nj))
                continue
            lst = list(st)
            lst[j] -= 1
            lst[i] += 1
            row = self.index[tuple(lst)]
            amp = math.sqrt(nj * (st[i] + 1))
            rows.append(row); cols.append(col); vals.append(amp)
        M = sp.csr_matrix((vals, (rows, cols)), shape=(self.dim, self.dim))
        self._hop_cache[key] = M
        return M

    # ---- pair creation P0^dag, P2mu^dag map N-2 -> N; build as rectangular ops
    def pair_ops(self, space_Nm2):
        """Return P0dag, {mu: P2dag_mu} as (dim_N x dim_{N-2}) sparse matrices."""
        idxN = self.index
        sm2 = space_Nm2
        dimS, dimN = sm2.dim, self.dim

        def add_two(st, i, jj):
            """apply a_i^dag a_jj^dag to occupation tuple st (from N-2 space)."""
            lst = list(st)
            amp = math.sqrt(lst[jj] + 1)
            lst[jj] += 1
            amp *= math.sqrt(lst[i] + 1)
            lst[i] += 1
            return tuple(lst), amp

        # P0^dag = sum_m (-1)^m d_m^dag d_{-m}^dag  - 2 s^dag s^dag
        rows, cols, vals = [], [], []
        for col, st in enumerate(sm2.states):
            for m in D_MS:
                i = self.mode_index(m); j = self.mode_index(-m)
                ns, amp = add_two(st, i, j)
                rows.append(idxN[ns]); cols.append(col); vals.append(((-1) ** m) * amp)
            ns, amp = add_two(st, 0, 0)
            rows.append(idxN[ns]); cols.append(col); vals.append(-2.0 * amp)
        P0d = sp.csr_matrix((vals, (rows, cols)), shape=(dimN, dimS))

        # P2mu^dag = 2 s^dag d_mu^dag + sqrt7 * sum CG(2m1 2m2|2mu) d_m1^dag d_m2^dag
        P2d = {}
        for mu in range(-2, 3):
            rows, cols, vals = [], [], []
            for col, st in enumerate(sm2.states):
                # 2 s^dag d_mu^dag
                ns, amp = add_two(st, 0, self.mode_index(mu))
                rows.append(idxN[ns]); cols.append(col); vals.append(2.0 * amp)
                for m1 in D_MS:
                    m2 = mu - m1
                    if m2 < -2 or m2 > 2:
                        continue
                    c = cg(2, m1, 2, m2, 2, mu)
                    if c == 0.0:
                        continue
                    ns, amp = add_two(st, self.mode_index(m1), self.mode_index(m2))
                    rows.append(idxN[ns]); cols.append(col); vals.append(math.sqrt(7) * c * amp)
            P2d[mu] = sp.csr_matrix((vals, (rows, cols)), shape=(dimN, dimS))
        return P0d, P2d

    # ---- single-boson tensor operators on fixed-N space
    def n_d(self):
        return sum(self.hop(self.mode_index(m), self.mode_index(m)) for m in D_MS)

    def n_s(self):
        return self.hop(0, 0)

    def dd_tensor(self, k, q):
        """(d^dag dtil)^(k)_q with dtil_m = (-1)^m d_{-m}."""
        out = sp.csr_matrix((self.dim, self.dim))
        for m1 in D_MS:
            m2 = q - m1
            if m2 < -2 or m2 > 2:
                continue
            c = cg(2, m1, 2, m2, k, q)
            if c == 0.0:
                continue
            # dtil_{m2} = (-1)^{m2} d_{-m2}
            out = out + c * ((-1) ** m2) * self.hop(self.mode_index(m1), self.mode_index(-m2))
        return out

    def L_op(self, q):
        return math.sqrt(10.0) * self.dd_tensor(1, q)

    def L2(self):
        out = sp.csr_matrix((self.dim, self.dim))
        for q in (-1, 0, 1):
            Lq = self.L_op(q)
            out = out + ((-1) ** q) * (Lq @ self.L_op(-q))
        return out

    def Q_op(self, q):
        """SU(3) quadrupole generator: s^dag dtil_q + d_q^dag s - (sqrt7/2)(d^dag dtil)^(2)_q"""
        # s^dag dtil_q = (-1)^q s^dag d_{-q}
        t1 = ((-1) ** q) * self.hop(0, self.mode_index(-q))
        t2 = self.hop(self.mode_index(q), 0)
        t3 = self.dd_tensor(2, q)
        return t1 + t2 - (math.sqrt(7) / 2.0) * t3

    def C2_SU3(self):
        QQ = sp.csr_matrix((self.dim, self.dim))
        for q in range(-2, 3):
            QQ = QQ + ((-1) ** q) * (self.Q_op(q) @ self.Q_op(-q))
        return 2.0 * QQ + 0.75 * self.L2()


def build_H(space, space_Nm2, h0, h2, C):
    P0d, P2d = space.pair_ops(space_Nm2)
    H = h0 * (P0d @ P0d.conj().T)
    for mu in range(-2, 3):
        H = H + h2 * (P2d[mu] @ P2d[mu].conj().T)
    if C != 0.0:
        H = H + C * space.L2()
    return H.tocsr()


def f2(lam, mu):
    return lam * lam + mu * mu + lam * mu + 3 * (lam + mu)


def su3_irreps_in_UN(N):
    """SU(3) irreps (lambda,mu) contained in the symmetric U(6) irrep [N].
    Standard result: (lam,mu) = (2N-6p-4q, 2q), p,q >= 0 with lam >= 0... 
    Verified numerically against C2 spectrum in tests (do not trust blindly)."""
    out = set()
    for p in range(0, N + 1):
        for q in range(0, N + 1):
            lam = 2 * N - 6 * p - 4 * q
            mu = 2 * q
            if lam >= 0:
                out.add((lam, mu))
    return sorted(out, key=lambda t: -f2(*t))
