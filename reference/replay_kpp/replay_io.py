#!/usr/bin/env python3
"""Inputs and outputs of the KPP replay harness (M3 sub-lane KPP; reference/replay_kpp/code/the_main_loop.F, a copy
of the COL harness reference/replay_col with per-case named records).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N] [--cases N]   (run by jobs/replay.sbatch)

The harness is a genmake2 build of vermix/code in which THE_MAIN_LOOP is replaced: THE_MODEL_MAIN runs
INITIALISE_FIXED (grid, INI_EOS, KPP_READPARMS, KPP_INIT_FIXED); the replacement calls KPP_INIT_VARIA, writes the
fixed KPP values (kpp_fixed.bin), then for every case reads the synthetic columns written here (kpp_in.bin), sets the
model state, the grid overrides (maskC/W/S, fCori, nzmax: land and shallow columns that the real one-column vermix grid
does not have) and the KPP.h priors, calls STATEKPP, KPP_FORCING_SURF, KPPMIX, KPP_DOUBLEDIFF on synthetic intermediate
inputs, then KPP_CALC, KPP_DO_EXCH, KPP_CALC_DIFF_T/S, KPP_CALC_VISC, KPP_TRANSPORT_T/S, and writes every output
(kpp_out.bin; all points incl. halos).

Files (big-endian stream of named records): CHARACTER*16 name, int32 rank, int32 n, float64 a(n); the per-case names
are 'cNN:<name>'. A Fortran array A(i,j,k,bi,bj) is the numpy C-order array [bj, bi, k, j, i]; the tile index is
t = (bj-1)*nSx + (bi-1), so 3-D fields are returned as [tile, k, j, i] and 2-D fields as [tile, j, i]; mx_vddiff
(i,j,0:Nr+1,mdiff,bi,bj) is returned as [tile, md, k, j, i] with k = 0..Nr+1.

Regimes (one per column, drawn per case): unstable (surface cooling, strong wind), stable (warming, strong
stratification), neutral (uniform T and S: dbloc = 0 exactly, no buoyancy forcing), shear (strong upper shear: the
zRef < drF(1) branch of KPP_FORCING_SURF), salt fingering and diffusive convection (KPP_DOUBLEDIFF's two branches),
land (maskC = 0, nzmax = 0), shallow (nzmax = 1, 2, 3 or 5), calm (no wind: ustar = epsLoc). fCori has exact zeros and
both signs; the intermediate inputs (dbloc, Ritop, shsq, dVsq, ustar, bo, bosol, TTALPHA, SSBETA, background
diffusivities) come from the same columns through a linear equation of state, masked as KPP_CALC masks them.

Numpy only (no jax): it runs in the batch job before the model.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

SIZE_KEYS = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "Nr")
IN3 = ("theta", "salt", "uVel", "vVel", "IVDConv", "maskC", "maskW", "maskS", "pViscAz", "pDiffKzS", "pDiffKzT",
       "pGhat", "dblocI", "RitopI", "shsqI", "dVsqI", "kapTI", "kapSI", "ghatI", "kapRU", "kapRV", "dfPrior")
IN3P = ("ttalphaI", "ssbetaI")
IN2 = ("sfU", "sfV", "sfT", "sfS", "adjCold", "Qsw", "fCori", "nzmax", "pHbl", "pFrac", "sdensI", "ustarI", "boI",
       "bosolI")
OUT3 = ("sk_dbloc", "sk_dbsfc", "fs_dVsq", "mx_ghat", "dd_kapT", "dd_kapS", "KPPviscAz", "KPPdiffKzS", "KPPdiffKzT",
        "KPPghat", "ex_viscAz", "cd_T0", "cd_S0", "cd_Tk", "cd_Sk", "cv_RU", "cv_RV", "tr_T", "tr_S")
OUT3P = ("sk_alpha", "sk_beta")
OUT2 = ("sk_rho1", "fs_ustar", "fs_bo", "fs_bosol", "mx_hbl", "KPPhbl", "KPPfrac")
OUTV = ("mx_vddiff",)
SCALARS = ("kpp_freq", "kpp_dumpFreq", "minKPPhbl", "epsln", "phepsi", "epsilon", "vonk", "dB_dz", "conc1", "conam",
           "concm", "conc2", "zetam", "conas", "concs", "conc3", "zetas", "Ricr", "cekman", "cmonob", "concv", "Vtc",
           "hbf", "deltaz", "deltau", "zmin", "zmax", "umin", "umax", "Riinfty", "BVSQcon", "difm0", "difs0", "dift0",
           "difmcon", "difscon", "diftcon", "Rrho0", "dsfmax", "cstar", "cg", "KPPwriteState",
           "KPP_ghatUseTotalDiffus", "KPPuseDoubleDiff", "LimitHblStable", "KPPuseSWfrac3D", "num_v_smooth_Ri")
REGIMES = ("unstable", "stable", "neutral", "shear", "finger", "diffusive", "land", "shallow", "calm")
N_CASES = 12
MDIFF = 3                      # KPP_PARAMS.h: parameter( mdiff = 3 )
NNI, NNJ = 890, 480            # KPP_PARAMS.h: parameter (nni = 890, nnj = 480)


def parse_size_h(path):
    """sNx, sNy, OLx, OLy, nSx, nSy, Nr from a SIZE.h PARAMETER statement."""
    text = Path(path).read_text()
    out = {}
    for key in SIZE_KEYS:
        m = re.search(rf"^\s*&?\s*{key}\s*=\s*(\d+)\s*,?", text, re.M)
        if not m:
            raise ValueError(f"{key} not found in {path}")
        out[key] = int(m.group(1))
    return out


def _dims(size):
    T = size["nSx"] * size["nSy"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    return T, Ny, Nx


def make_case(size, rng):
    """One case: ({IN3 name: [tile, k, j, i]}, {IN3P name: [tile, Nr+1, j, i]}, {IN2 name: [tile, j, i]}, regimes)."""
    T, Ny, Nx = _dims(size)
    Nr = size["Nr"]
    g, rho0 = 9.81, 999.8
    z = np.cumsum(np.full(Nr, 10.0)) - 5.0 + 1.5 * np.arange(Nr) ** 1.5      # rough cell-centre depths (m)
    dz = np.diff(np.concatenate([[0.0], z + np.gradient(z) / 2]))
    reg = rng.integers(0, len(REGIMES), (T, Ny, Nx))
    th = np.empty((T, Nr, Ny, Nx))
    sa = np.empty((T, Nr, Ny, Nx))
    u = np.empty((T, Nr, Ny, Nx))
    v = np.empty((T, Nr, Ny, Nx))
    mask = np.ones((T, Nr, Ny, Nx))
    nz = np.full((T, Ny, Nx), float(Nr))
    sfU, sfV, sfT, sfS = (np.empty((T, Ny, Nx)) for _ in range(4))
    for t in range(T):
        for j in range(Ny):
            for i in range(Nx):
                r = REGIMES[reg[t, j, i]]
                hml = rng.uniform(5.0, 120.0)
                ts, ss = rng.uniform(2.0, 28.0), rng.uniform(33.0, 37.0)
                dTdz, dSdz = rng.uniform(0.0, 0.05), rng.uniform(-0.01, 0.01)
                if r == "finger":
                    dTdz, dSdz = rng.uniform(0.02, 0.08), rng.uniform(0.002, 0.012)
                elif r == "diffusive":
                    ts, dTdz, dSdz = rng.uniform(-1.0, 4.0), -rng.uniform(0.005, 0.03), -rng.uniform(0.003, 0.02)
                elif r == "stable":
                    hml, dTdz = rng.uniform(2.0, 15.0), rng.uniform(0.05, 0.2)
                prof = np.maximum(z - hml, 0.0)
                th[t, :, j, i] = np.maximum(ts - dTdz * prof + rng.normal(0.0, 0.002, Nr), -1.9)
                sa[t, :, j, i] = ss - dSdz * prof + rng.normal(0.0, 0.0005, Nr)
                if r == "neutral":
                    th[t, :, j, i], sa[t, :, j, i] = ts, ss
                if r == "unstable":
                    th[t, :3, j, i] -= rng.uniform(0.0, 0.05, 3)
                us = rng.normal(0.0, 0.2)
                vs = rng.normal(0.0, 0.2)
                decay = np.exp(-z / rng.uniform(10.0, 80.0))
                u[t, :, j, i] = us * decay + rng.normal(0.0, 0.01, Nr)
                v[t, :, j, i] = vs * decay + rng.normal(0.0, 0.01, Nr)
                if r == "shear":
                    u[t, 0, j, i] += rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 0.4)
                    th[t, 1:, j, i] -= rng.uniform(0.5, 2.0)
                tau = 10.0 ** rng.uniform(-5.0, -3.3)
                ang = rng.uniform(0.0, 2 * np.pi)
                sfU[t, j, i], sfV[t, j, i] = tau * np.cos(ang), tau * np.sin(ang)
                q = rng.uniform(-300.0, 300.0) / (rho0 * 3994.0)
                if r == "unstable":
                    q = -abs(q)
                elif r == "stable":
                    q = abs(q)
                sfT[t, j, i] = q
                sfS[t, j, i] = rng.normal(0.0, 1.0e-6)
                if r == "neutral":
                    sfT[t, j, i] = sfS[t, j, i] = 0.0
                if r == "calm":
                    sfU[t, j, i] = sfV[t, j, i] = 0.0
                if r == "land":
                    mask[t, :, j, i] = 0.0
                    nz[t, j, i] = 0.0
                if r == "shallow":
                    n = int(rng.choice([1, 2, 3, 5]))
                    mask[t, n:, j, i] = 0.0
                    nz[t, j, i] = float(n)
    s3, s2, s3p = (T, Nr, Ny, Nx), (T, Ny, Nx), (T, Nr + 1, Ny, Nx)
    # linear EOS stand-in for the intermediate inputs (KPPMIX, KPP_FORCING_SURF, KPP_DOUBLEDIFF)
    alpha = 2.0e-4 * (1.0 + 0.3 * rng.uniform(-1.0, 1.0, s3))
    beta = 7.4e-4 * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, s3))
    drho = rho0 * (-alpha * (th - 10.0) + beta * (sa - 35.0))
    mk = mask * np.concatenate([mask[:, 1:], mask[:, -1:]], axis=1)
    dbloc = np.zeros(s3)
    dbloc[:, :-1] = g * (drho[:, 1:] - drho[:, :-1]) / (drho[:, 1:] + rho0)
    dbloc *= mk
    kk = np.arange(1, Nr + 1)[None, :, None, None]
    dbloc[kk == nz[:, None]] = 0.0
    dbsfc = g * (drho - drho[:, :1]) / (drho + rho0) * mask * mask[:, :1]
    dbsfc[kk == nz[:, None]] = 0.0
    ritop = (z - z[0])[None, :, None, None] * dbsfc                        # (zgrid(1)-zgrid(k)) * dbsfc
    um = 0.5 * (u + np.roll(u, -1, axis=3))
    vm = 0.5 * (v + np.roll(v, -1, axis=2))
    shsq = np.zeros(s3)
    shsq[:, :-1] = (um[:, :-1] - um[:, 1:]) ** 2 + (vm[:, :-1] - vm[:, 1:]) ** 2
    dvsq = (um - um[:, :1]) ** 2 + (vm - vm[:, :1]) ** 2 + 1.0e-6 * rng.uniform(0.0, 1.0, s3)
    ustar = np.sqrt(np.sqrt(sfU ** 2 + sfV ** 2))
    ustar[ustar == 0.0] = np.sqrt(0.5 * 1.0e-10 * 10.0)
    ttal = np.empty(s3p)
    ssbe = np.empty(s3p)
    ttal[:, :Nr], ssbe[:, :Nr] = -rho0 * alpha, rho0 * beta
    ttal[:, Nr], ssbe[:, Nr] = ttal[:, Nr - 1], ssbe[:, Nr - 1]
    bo = -g * (ttal[:, 0] * sfT + ssbe[:, 0] * sfS) / (rho0 + drho[:, 0])
    bosol = g * ttal[:, 0] * rng.uniform(-300.0, 0.0, s2) / 3994.0 / rho0 / (rho0 + drho[:, 0])
    kapT = 1.0e-5 + 1.0e-6 * rng.uniform(0.0, 1.0, s3)
    kapS = 1.0e-5 + 1.0e-6 * rng.uniform(0.0, 1.0, s3)
    mW = mask * np.where(rng.uniform(size=s3) < 0.05, 0.0, 1.0)
    mS = mask * np.where(rng.uniform(size=s3) < 0.05, 0.0, 1.0)
    fc = 1.4e-4 * rng.uniform(-1.5, 1.5, s2)
    fc[rng.uniform(size=s2) < 0.15] = 0.0
    adj = np.where(rng.uniform(size=s2) < 0.2, rng.normal(0.0, 1.0e-6, s2), 0.0)
    f3 = {"theta": th, "salt": sa, "uVel": u, "vVel": v,
          "IVDConv": (rng.uniform(size=s3) < 0.2).astype(float), "maskC": mask, "maskW": mW, "maskS": mS,
          "pViscAz": 10.0 ** rng.uniform(-4.0, -2.0, s3), "pDiffKzS": 10.0 ** rng.uniform(-5.0, -3.0, s3),
          "pDiffKzT": 10.0 ** rng.uniform(-5.0, -3.0, s3), "pGhat": rng.uniform(0.0, 50.0, s3),
          "dblocI": dbloc, "RitopI": ritop, "shsqI": shsq, "dVsqI": dvsq, "kapTI": kapT, "kapSI": kapS,
          "ghatI": dbloc.copy(), "kapRU": 10.0 ** rng.uniform(-5.0, -2.0, s3),
          "kapRV": 10.0 ** rng.uniform(-5.0, -2.0, s3), "dfPrior": rng.normal(0.0, 1.0, s3)}
    f3p = {"ttalphaI": ttal, "ssbetaI": ssbe}
    f2 = {"sfU": sfU, "sfV": sfV, "sfT": sfT, "sfS": sfS, "adjCold": adj,
          "Qsw": rng.uniform(-300.0, 50.0, s2), "fCori": fc, "nzmax": nz, "pHbl": rng.uniform(0.0, 200.0, s2),
          "pFrac": rng.uniform(0.0, 1.0, s2), "sdensI": rho0 + drho[:, 0], "ustarI": ustar, "boI": bo,
          "bosolI": bosol}
    assert tuple(f3) == IN3 and tuple(f3p) == IN3P and tuple(f2) == IN2
    return f3, f3p, f2, reg


def make_inputs(size, seed=20261002, n_cases=N_CASES):
    rng = np.random.default_rng(seed)
    return [make_case(size, rng) for _ in range(n_cases)]


def _to_fortran(a, size, rank):
    """[tile, ...] -> [nSy, nSx, ...] (C order = Fortran (..., bi, bj))."""
    if rank <= 1:
        return a
    return a.reshape((size["nSy"], size["nSx"]) + a.shape[1:])


def case_name(c, name):
    """Record name of case c (1-based); case 0: the plain name."""
    return f"c{c:02d}:{name}" if c > 0 else name


def _write_record(fh, name, a):
    a = np.ascontiguousarray(a, dtype=">f8").ravel()
    if len(name) > 16:
        raise ValueError(f"record name {name!r} longer than 16")
    fh.write(name.ljust(16).encode("ascii"))
    fh.write(np.array([a.ndim, a.size], ">i4").tobytes())
    fh.write(a.tobytes())


def write_inputs(rundir, size, seed=20261002, n_cases=N_CASES):
    rundir = Path(rundir)
    cases = make_inputs(size, seed, n_cases)
    p_in = rundir / "kpp_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    with open(p_in, "xb") as fh:
        _write_record(fh, "header", np.array([size["sNx"], size["sNy"], size["Nr"], n_cases], float))
        for c, (f3, f3p, f2, _) in enumerate(cases, start=1):
            for name in IN3:
                _write_record(fh, case_name(c, name), _to_fortran(f3[name], size, 3))
            for name in IN3P:
                _write_record(fh, case_name(c, name), _to_fortran(f3p[name], size, 3))
            for name in IN2:
                _write_record(fh, case_name(c, name), _to_fortran(f2[name], size, 2))
    meta = {"seed": seed, "size": size, "cases": n_cases, "IN3": IN3, "IN3P": IN3P, "IN2": IN2,
            "regimes": {REGIMES[r]: int(sum((case[3] == r).sum() for case in cases)) for r in range(len(REGIMES))},
            "sha256": {p_in.name: hashlib.sha256(p_in.read_bytes()).hexdigest()}}
    (rundir / "kpp_in.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def read_records(path):
    """{name: (rank, float64 array (flat))} of a named-record file, in file order."""
    buf = Path(path).read_bytes()
    pos, out = 0, {}
    while pos < len(buf):
        name = buf[pos:pos + 16].decode("ascii").strip()
        rank, n = np.frombuffer(buf, ">i4", 2, pos + 16)
        pos += 24
        a = np.frombuffer(buf, ">f8", int(n), pos).astype(np.float64)
        pos += 8 * int(n)
        if name in out:
            raise ValueError(f"{path}: record {name} twice")
        out[name] = (int(rank), a)
    return out


def _shape(a, rank, size, name):
    """Flat Fortran-order record -> [tile, k, j, i] (rank 3), [tile, j, i] (rank 2), [tile, md, k, j, i] (rank 4,
    k = 0..Nr+1), 1-D (rank 1)."""
    T, Ny, Nx = _dims(size)
    if rank == 3:
        nk = a.size // (T * Ny * Nx)
        if nk * T * Ny * Nx != a.size:
            raise ValueError(f"{name}: {a.size} values are not [tile, k, {Ny}, {Nx}]")
        return a.reshape(size["nSy"], size["nSx"], nk, Ny, Nx).reshape(T, nk, Ny, Nx)
    if rank == 2:
        return a.reshape(size["nSy"], size["nSx"], Ny, Nx).reshape(T, Ny, Nx)
    if rank == 4:
        return a.reshape(size["nSy"], size["nSx"], MDIFF, size["Nr"] + 2, Ny, Nx).reshape(
            T, MDIFF, size["Nr"] + 2, Ny, Nx)
    return a


def n_cases(rundir):
    return int(read_records(Path(rundir) / "kpp_in.bin")["header"][1][3])


def read_inputs(rundir, size):
    """[case 1..n] of {name: array}."""
    recs = read_records(Path(rundir) / "kpp_in.bin")
    n = int(recs["header"][1][3])
    out = []
    for c in range(1, n + 1):
        d = {}
        for name in IN3 + IN3P:
            d[name] = _shape(recs[case_name(c, name)][1], 3, size, name)
        for name in IN2:
            d[name] = _shape(recs[case_name(c, name)][1], 2, size, name)
        out.append(d)
    return out


def read_outputs(rundir, size):
    """[case 1..n] of {name: array}."""
    recs = read_records(Path(rundir) / "kpp_out.bin")
    out = {}
    for full, (rank, a) in recs.items():
        c, name = int(full[1:3]), full[4:]
        out.setdefault(c, {})[name] = _shape(a, rank, size, name)
    return [out[c] for c in sorted(out)]


def read_fixed(rundir, size):
    recs = read_records(Path(rundir) / "kpp_fixed.bin")
    out = {name: _shape(a, rank, size, name) for name, (rank, a) in recs.items()}
    out["scalars"] = dict(zip(SCALARS, (float(x) for x in out["scalars"])))
    # wmt(0:nni+1,0:nnj+1): Fortran order, i fastest -> [j, i] in C order
    for n in ("wmt", "wst"):
        out[n] = out[n].reshape(NNJ + 2, NNI + 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261002)
    w.add_argument("--cases", type=int, default=N_CASES)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed, a.cases)
    print(f"KPP replay inputs written: {json.dumps(meta['size'])} seed {meta['seed']} cases {meta['cases']} "
          f"regimes {json.dumps(meta['regimes'])} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
