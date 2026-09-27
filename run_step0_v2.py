"""Step 0 extended: adds one-body density matrix diagnostics and
identifies SU(3)-pure states outside the predicted solvable family."""
import numpy as np
import json, math, sys
from ibm_pds import IBMSpace, build_H, f2
from run_step0 import m0_block, rdm_modes, vn_entropy, purity

def main(N, h0, h2=1.0, C=0.0523, outfile=None):
    print(f"=== Step 0 v2: N={N}, h0={h0}, h2={h2}, C={C} ===")
    spN = IBMSpace(N); spN2 = IBMSpace(N - 2)
    H = build_H(spN, spN2, h0, h2, C)
    L2 = spN.L2(); C2 = spN.C2_SU3(); nd = spN.n_d()
    sel = m0_block(spN)
    Hb = H[np.ix_(sel, sel)].toarray()
    evals, evecs = np.linalg.eigh(Hb)
    L2b = L2[np.ix_(sel, sel)].toarray()
    C2b = C2[np.ix_(sel, sel)].toarray()
    ndb = nd[np.ix_(sel, sel)].toarray()

    # degeneracy-aware rotation (H -> L2 -> C2), as before
    groups = []
    i = 0
    while i < len(evals):
        j = i
        while j + 1 < len(evals) and evals[j + 1] - evals[i] < 1e-9:
            j += 1
        groups.append((i, j)); i = j + 1
    V = evecs.copy()
    for (i, j) in groups:
        if j > i:
            sub = V[:, i:j + 1]
            w, u = np.linalg.eigh(sub.conj().T @ L2b @ sub)
            sub = sub @ u
            k = 0
            while k <= j - i:
                l = k
                while l + 1 <= j - i and w[l + 1] - w[k] < 1e-9:
                    l += 1
                if l > k:
                    s2 = sub[:, k:l + 1]
                    _, u2 = np.linalg.eigh(s2.conj().T @ C2b @ s2)
                    sub[:, k:l + 1] = s2 @ u2
                k = l + 1
            V[:, i:j + 1] = sub

    # C2 sectors
    c2evals, c2vecs = np.linalg.eigh(C2b)
    c2r = np.round(c2evals, 6)
    sectors = {}
    for idx, v in enumerate(c2r):
        sectors.setdefault(v, []).append(idx)
    sector_keys = sorted(sectors.keys(), reverse=True)
    projs = {k: c2vecs[:, sectors[k]] for k in sector_keys}

    pred = []
    for k in range(0, N // 2 + 1):
        lam, mu = 2 * N - 4 * k, 2 * k
        Ls = range(0, 2 * N + 1, 2) if k == 0 else range(2 * k, 2 * N - 2 * k + 1)
        for L in Ls:
            pred.append(dict(k=k, L=L, lam=lam, mu=mu,
                             E=6 * h2 * k * (2 * N - 2 * k + 1) + C * L * (L + 1),
                             f2=f2(lam, mu)))

    states_list = spN.states
    # one-body density matrix helper: <b_i^dag b_j> over 6 modes
    hop_blocks = {}
    for i6 in range(6):
        for j6 in range(6):
            hop_blocks[(i6, j6)] = spN.hop(i6, j6)[np.ix_(sel, sel)]

    rows = []
    for n in range(len(evals)):
        v = V[:, n]
        E = float(evals[n])
        l2 = float(np.real(v.conj() @ L2b @ v))
        Lval = (-1 + math.sqrt(1 + 4 * max(l2, 0))) / 2
        c2m = float(np.real(v.conj() @ C2b @ v))
        c2var = float(np.real(v.conj() @ (C2b @ (C2b @ v))) - c2m ** 2)
        ndm = float(np.real(v.conj() @ ndb @ v))
        ndvar = float(np.real(v.conj() @ (ndb @ (ndb @ v))) - ndm ** 2)
        solv = None
        for p in pred:
            if (abs(E - p["E"]) < 1e-8 and abs(Lval - p["L"]) < 1e-6
                    and abs(c2m - p["f2"]) < 1e-6 and c2var < 1e-6):
                solv = p; break
        probs = []
        for kk in sector_keys:
            Pv = projs[kk].conj().T @ v
            probs.append(float(np.real(Pv.conj() @ Pv)))
        pv = np.array(probs); pv2 = pv[pv > 1e-14]
        S_irrep = float(-(pv2 * np.log(pv2)).sum())
        n_irreps = int((pv > 1e-10).sum())

        psi_full = np.zeros(spN.dim, dtype=complex); psi_full[sel] = v
        rho_s = rdm_modes(psi_full, states_list, (0,))
        S_sd = vn_entropy(rho_s); pur_s = purity(rho_s)

        # one-body density matrix (6x6), normalized to N
        obdm = np.zeros((6, 6), dtype=complex)
        for i6 in range(6):
            for j6 in range(6):
                obdm[i6, j6] = v.conj() @ (hop_blocks[(j6, i6)] @ v)  # <b_i^dag b_j>: careful
        # <b_i^dag b_j> = v^dag hop(i,j) v ; hop(i,j) = a_i^dag a_j
        obdm = np.zeros((6, 6), dtype=complex)
        for i6 in range(6):
            for j6 in range(6):
                obdm[i6, j6] = v.conj() @ (hop_blocks[(i6, j6)] @ v)
        occ = np.linalg.eigvalsh(obdm).real
        occ = np.clip(occ, 0, None)
        p_occ = occ / occ.sum()
        p_nz = p_occ[p_occ > 1e-14]
        S_ob = float(-(p_nz * np.log(p_nz)).sum())
        cond_frac = float(p_occ.max())

        rows.append(dict(idx=n, E=E, L=round(Lval, 6), c2=c2m, c2var=c2var,
                         nd=ndm, ndvar=ndvar,
                         solvable=(solv is not None),
                         band=(None if solv is None else solv["k"]),
                         S_sd=S_sd, purity_s=pur_s, S_irrep=S_irrep,
                         n_irreps=n_irreps, S_ob=S_ob, cond_frac=cond_frac,
                         occ=[float(x) for x in sorted(occ, reverse=True)]))

    nsolv = sum(r["solvable"] for r in rows)
    anom = [r for r in rows if (not r["solvable"]) and r["c2var"] < 1e-6]
    print(f"solvable: {nsolv}/{len(pred)}; anomalous pure non-solvable: {len(anom)}")
    for r in anom:
        print(f"  anomalous: E={r['E']:.4f} L={r['L']} <C2>={r['c2']:.3f} "
              f"n_irreps={r['n_irreps']} S_sd={r['S_sd']:.4f}")
    if outfile:
        with open(outfile, "w") as f:
            json.dump(dict(N=N, h0=h0, h2=h2, C=C, rows=rows), f)
    return rows

if __name__ == "__main__":
    N = int(sys.argv[1]); h0 = float(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else None
    main(N, h0, outfile=out)
