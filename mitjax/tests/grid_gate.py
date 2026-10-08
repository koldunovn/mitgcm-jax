"""Helpers of the grid gates (plan Task 10): build the grid of an M1 variant the way INITIALISE_FIXED does and compare
every GRID.h field the oracle dumps with the oracle, bitwise, on all points (halos and land included).

Oracle: the registered dumps-on run of each variant (reference/reference_runs.py, kind "jdon"; the dumps of the
first iteration, stage G00_geometry, and the hFac arrays of stage S00_begin, group r). Comparison by element
equality (`==`, so +0 and -0 are equal), with the oracle and our field required finite at every point (numpy on the
whole array; Python max() would skip a NaN, review finding on tools/diffdump.py).
"""

import dataclasses
import importlib.util

import numpy as np

from mitjax import paths
from mitjax.config.params import load
from mitjax.io.dump import DumpSet
from mitjax.model.src.ini_parms import Exch2Topology, ini_parms_grid
from mitjax.model.src.ini_cori import ini_cori
from mitjax.model.src.ini_depths import ini_depths
from mitjax.model.src.ini_grid import ini_grid
from mitjax.model.src.ini_masks_etc import ini_masks_etc
from mitjax.model.src.set_grid_factors import set_grid_factors
from mitjax.pkg.rw.read_rec import RW

VARIANTS = [
    ("tutorial_barotropic_gyre", "input"),
    ("tutorial_baroclinic_gyre", "input"),
    ("advect_xy", "input"),
    ("advect_xy", "input.ab3_c4"),
    ("advect_xz", "input"),
    ("advect_xz", "input.nlfs"),
    ("advect_xz", "input.pqm"),
    ("global_ocean.90x40x15", "input"),
    ("tutorial_global_oce_optim", "input_ad"),
]

# Dumped fields that are not GRID.h fields set by the Task 10 routines (named exemptions, L-TOL-5): the V-group
# reference profiles (INI_PARMS: tRef, sRef; SET_REF_STATE: rVel2wUnit, wUnit2rVel, rhoFacC, rhoFacF, dBdrRef,
# phiRef), the r* / linear free-surface fields of group R (INI_LINEAR_PHISURF, INI_NLFS_VARS, CALC_R_STAR /
# UPDATE_R_STAR) and the 3-D mixing parameters of group G.
NOT_TASK10 = {
    "tRef": "PARAMS.h, INI_PARMS", "sRef": "PARAMS.h, INI_PARMS",
    "rVel2wUnit": "SET_REF_STATE", "wUnit2rVel": "SET_REF_STATE", "rhoFacC": "SET_REF_STATE",
    "rhoFacF": "SET_REF_STATE", "dBdrRef": "SET_REF_STATE", "phiRef": "SET_REF_STATE",
    "Bo_surf": "INI_LINEAR_PHISURF", "recip_Bo": "INI_LINEAR_PHISURF",
    "hFac_surfC": "INI_NLFS_VARS / r*", "hFac_surfW": "INI_NLFS_VARS / r*", "hFac_surfS": "INI_NLFS_VARS / r*",
    "etaHnm1": "INI_NLFS_VARS", "pStarFacK": "r*",
    "rStarFacNm1C": "r*", "rStarFacNm1W": "r*", "rStarFacNm1S": "r*", "rStarExpC": "r*", "rStarExpW": "r*",
    "rStarExpS": "r*", "rStarDhCDt": "r*", "rStarDhWDt": "r*", "rStarDhSDt": "r*",
    # 3-D mixing parameters of group G (jaxdump.F JAXDUMP_GEOM, under ALLOW_3D_*): not GRID.h
    "diffKr": "DYNVARS.h, INI_MIXING (ALLOW_3D_DIFFKR)",
    "viscAhDfld": "MOM_VISC.h, MOM_INIT_FIXED (ALLOW_3D_VISCAH)", "viscAhZfld": "MOM_VISC.h, MOM_INIT_FIXED",
    "viscA4Dfld": "MOM_VISC.h, MOM_INIT_FIXED (ALLOW_3D_VISCA4)", "viscA4Zfld": "MOM_VISC.h, MOM_INIT_FIXED",
}
# hFac arrays dumped at S00_begin (group r)
S00_FIELDS = ("hFacC", "hFacW", "hFacS", "recip_hFacC")
# Under r* (nonlinFreeSurf > 0 and select_rStar > 0) INITIALISE_VARIA calls UPDATE_R_STAR(.TRUE.)
# (model/src/initialise_varia.F:297-310), which rescales hFacC/W/S and recompute recip_hFacC/W/S from h0Fac*rStarFac
# before the first dump; the initial geometric factors are h0FacC/W/S (dumped in group R, set by INI_MASKS_ETC).
RSTAR_RESCALED = ("hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS")


def _registry():
    spec = importlib.util.spec_from_file_location("_mjx_reference_runs", paths.REPO / "reference" / "reference_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DS = {}
_EXP = {}


def oracle(exp, inp):
    """(DumpSet of the registered dumps-on run, its first iteration, the run directory)."""
    key = (exp, inp)
    if key not in _DS:
        reg = _registry()
        runs = [r for r in reg.RUNS if (r["exp"], r["input"], r["kind"]) == (exp, inp, "jdon")]
        if len(runs) != 1:
            raise FileNotFoundError(f"{exp}/{inp}: {len(runs)} registered dumps-on runs")
        top = reg.run_top(runs[0])
        ds = DumpSet(top / "dumps")
        _DS[key] = (ds, ds.iterations()[0], top / "rundir")
    return _DS[key]


def experiment(exp, inp):
    if (exp, inp) not in _EXP:
        _EXP[(exp, inp)] = load(exp, inp)
    return _EXP[(exp, inp)]


def exch2_topology(ds):
    """W2 topology of an exch2 build from the oracle's dump headers (jaxdump.F:151-157 writes exch2_tBasex/y of each
    W2 tile; the facet size is the extent of the tiles). Single process: W2_myTileList(bi,bj) = bi + (bj-1)*nSx
    (pkg/exch2/w2_map_procs.F:73-94, default ordering, no blank tiles)."""
    ti = ds.tiles_info
    faces = ds.face_shapes()
    if sorted(faces) != [1]:
        raise NotImplementedError(f"exch2 with facets {sorted(faces)}: only a single facet is ported")
    tiles = sorted(ti)
    return Exch2Topology(exch2_tBasex=tuple(ti[t][1] for t in tiles), exch2_tBasey=tuple(ti[t][2] for t in tiles),
                         exch2_mydNx=(faces[1][1],), exch2_mydNy=(faces[1][0],), W2_myTileList=tuple(tiles),
                         source=f"jaxdump headers of {ds.dir}")


def exchanger(exp):
    from mitjax.eesupp.exch_maps import load_maps
    from mitjax.eesupp.exchange import Exchanger
    return Exchanger(load_maps(exp))


def grid_params(exp, inp):
    e = experiment(exp, inp)
    ds, _, _ = oracle(exp, inp)
    ex2 = exch2_topology(ds) if e.cfg.cpp.ALLOW_EXCH2 else None
    return ini_parms_grid(e, ex2)


def build_grid(exp, inp, params=None, ex=None):
    """The grid of a variant: INI_GRID, SET_GRID_FACTORS, INI_DEPTHS, INI_MASKS_ETC, INI_CORI in the order of
    INITIALISE_FIXED (model/src/initialise_fixed.F:156, 181, 190, 201, 236; LOAD_REF_FILES, INI_EOS, SET_REF_STATE
    and PACKAGES_INIT_FIXED in between write no GRID.h field)."""
    e = experiment(exp, inp)
    cfg = e.cfg
    _, _, rundir = oracle(exp, inp)
    params = grid_params(exp, inp) if params is None else params
    ex = exchanger(exp) if ex is None else ex
    rw = RW(rundir, params.readBinaryPrec, cfg.size)
    grid, delX, delY, latBandClimRelax = ini_grid(cfg=cfg, params=params)
    grid = set_grid_factors(grid, cfg=cfg, params=params)
    grid = ini_depths(grid, cfg=cfg, params=params, ex=ex, rw=rw)
    grid = ini_masks_etc(grid, cfg=cfg, params=params, ex=ex)
    grid = ini_cori(grid, cfg=cfg, params=params)
    return grid


def rstar(exp, inp):
    """r* in effect at the first step: nonlinFreeSurf > 0 and select_rStar > 0 (initialise_varia.F:304-307)."""
    e = experiment(exp, inp)
    from mitjax.params_io import RunParams, fortran_default
    rp = RunParams(e.run)
    nlfs = rp.get("data", "PARM01", "nonlinFreeSurf", default=fortran_default("model/src/set_defaults.F:257",
                                                                              "nonlinFreeSurf", e))
    sel = rp.get("data", "PARM01", "select_rStar", default=fortran_default("model/src/set_defaults.F:260",
                                                                           "select_rStar", e))
    return e.cfg.cpp.NONLIN_FRSURF and nlfs > 0 and sel > 0


def gated_pairs(exp, inp, ds, it):
    """[(our field, (stage, dumped field))] of every comparison, and {dumped field: reason} of the exemptions."""
    rs = rstar(exp, inp)
    pairs, exempt = [], {}
    for (i, stage, name) in ds.keys(it):
        if stage == "G00_geometry":
            if name in NOT_TASK10:
                exempt[name] = NOT_TASK10[name]
            elif rs and name in RSTAR_RESCALED:
                exempt[name] = "rescaled by UPDATE_R_STAR(.TRUE.) in INITIALISE_VARIA (r*)"
            elif name in ("h0FacC", "h0FacW", "h0FacS"):
                pairs.append((name, (stage, name)))
                pairs.append(("hFac" + name[-1], (stage, name)))      # INI_MASKS_ETC's hFac = h0Fac (:488-496)
            else:
                pairs.append((name, (stage, name)))
        elif stage == "S00_begin" and name in S00_FIELDS:
            if rs:
                exempt[f"S00_begin/{name}"] = "rescaled by UPDATE_R_STAR(.TRUE.) in INITIALISE_VARIA (r*)"
            else:
                pairs.append((name, (stage, name)))
    return pairs, exempt


def as_dump_shape(ours, ref):
    """Our storage in the dump's [tile, k, j, i] layout (2-D fields gain k=1; kind-V records are on tile 1 only, one
    constant level per vector element)."""
    a = np.asarray(ours)
    if a.ndim == 1:                                  # vertical vector (Nr) / (Nr+1): dump record on tile 1
        lev = ref[0, :, :, :]
        if not np.all(lev == lev[:, :1, :1]):
            raise ValueError("kind-V record is not constant per level")
        return a, ref[0, :, 0, 0]
    if a.ndim == 3:
        a = a[:, None]
    return a, ref


def compare(grid, exp, inp):
    """{field: (n points compared, n differing, n non-finite ours, n non-finite oracle, n differing bit patterns)} for
    every gated pair, and the exemptions. Every point of every tile, halos included. "Differing" is element
    inequality (`==`: +0 equals -0); the bit-pattern count also sees the sign of a zero (measured equal in all nine
    variants, jobs 27827923 / 27827998, so the gate requires it)."""
    ds, it, _ = oracle(exp, inp)
    pairs, exempt = gated_pairs(exp, inp, ds, it)
    out = {}
    for ours_name, (stage, name) in pairs:
        ref = ds.field(it, stage, name)
        ours, ref = as_dump_shape(getattr(grid, ours_name).data, ref)
        label = ours_name if ours_name == name else f"{ours_name}~{name}"
        if ours.shape != ref.shape:
            out[label] = ("shape", ours.shape, ref.shape)
            continue
        o64, r64 = np.ascontiguousarray(ours, np.float64), np.ascontiguousarray(ref, np.float64)
        out[label] = (ours.size, int(np.count_nonzero(~(ours == ref))), int(np.count_nonzero(~np.isfinite(ours))),
                      int(np.count_nonzero(~np.isfinite(ref))), int(np.count_nonzero(o64.view(np.int64)
                                                                                   != r64.view(np.int64))))
    return out, exempt


def failures(result):
    return {k: v for k, v in result.items() if v[0] == "shape" or any(v[1:])}


# ---------------------------------------------------------------------------------------------------------------
# negative controls

def perturbed_delR(params):
    """delR(Nr) moved by one ulp (the planted error of the vertical-grid control)."""
    d = np.array(params.delR, copy=True)
    d[-1] = np.nextafter(d[-1], np.inf)
    return dataclasses.replace(params, delR=d)


class NoExchange:
    """An exchanger whose every exchange returns its input unchanged (the planted error of the halo control)."""

    def EXCH_XY_RL(self, phi):
        return phi

    def EXCH_UV_XY_RL(self, u, v, withSigns):
        return u, v


# ---------------------------------------------------------------------------------------------------------------
# grid fields dumped from the extended jaxdump group G/V (reference/jaxdump/jaxdump.F at b73c05c; build job 27828735,
# dumps-on runs job 27828737, all nine variants INVISIBLE). Not in reference/reference_runs.py yet (lane A's
# registry): the run is addressed by its job id here.
EXTRA_JOB = 27828737
EXTRA_G = ("kSurfC", "kSurfW", "kSurfS", "kLowC", "topoZ", "cosFacU", "cosFacV", "sqCosFacU", "sqCosFacV")
EXTRA_V = ("deepFacC", "deepFac2C", "deepFacF", "deepFac2F", "recip_deepFacC", "recip_deepFac2C", "recip_deepFacF",
           "recip_deepFac2F", "aHybSigmF", "bHybSigmF", "aHybSigmC", "bHybSigmC", "dAHybSigF", "dBHybSigF",
           "dBHybSigC", "dAHybSigC", "rkSign", "gravitySign")
_DSX = {}


def extra_oracle(exp, inp):
    if (exp, inp) not in _DSX:
        top = paths.REFERENCE_RUNS / exp / inp / f"job{EXTRA_JOB}-jdon"
        ds = DumpSet(top / "dumps")
        _DSX[(exp, inp)] = (ds, ds.iterations()[0])
    return _DSX[(exp, inp)]


def compare_extra(grid, exp, inp):
    """{field: (n, ndiff, nonfinite ours, nonfinite oracle, ndiff bits)} for the EXTRA_G / EXTRA_V fields."""
    ds, it = extra_oracle(exp, inp)
    out = {}
    for name in EXTRA_G + EXTRA_V:
        ref = ds.field(it, "G00_geometry", name)
        val = getattr(grid, name)
        ours = np.asarray(val.data if hasattr(val, "data") else val, np.float64)
        if name in EXTRA_V:
            if ours.ndim == 0:
                ours = ours[None]
            ref = ref[0, :, 0, 0]
            lev = ds.field(it, "G00_geometry", name)[0]
            if not np.all(lev == lev[:, :1, :1]):
                raise ValueError(f"{name}: kind-V record not constant per level")
        elif ours.ndim == 2:                       # cosFacU(j) per tile: the dump repeats it along i
            ours = np.broadcast_to(ours[:, None, :, None], ref.shape)
        else:
            ours = ours[:, None]
        if ours.shape != ref.shape:
            out[name] = ("shape", ours.shape, ref.shape)
            continue
        o64, r64 = np.ascontiguousarray(ours), np.ascontiguousarray(ref, np.float64)
        out[name] = (ours.size, int(np.count_nonzero(~(ours == ref))), int(np.count_nonzero(~np.isfinite(ours))),
                     int(np.count_nonzero(~np.isfinite(ref))),
                     int(np.count_nonzero(o64.view(np.int64) != r64.view(np.int64))))
    return out
