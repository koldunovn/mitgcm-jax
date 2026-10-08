"""Helpers of the cg2d gates (M1 sub-lane CG2D): inputs from the oracle's substep dumps, the ported routines run under
jit at the gate XLA flags with every float a traced argument, and element-wise comparisons on all points.

Oracle: the registered dumps-on run of each variant that calls the solver (reference/reference_runs.py, kind "jdon";
read through mitjax/tests/grid_gate.oracle). Stages: C01_cg2d_inputs (cg2d_b, cg2d_x, the operator aW2d, aS2d, aC2d,
pW, pS, pC right before CALL CG2D), C02_cg2d_solution (cg2d_x right after it, before SOLVE_FOR_PRESSURE's exchange,
and numIters, nIterMin, firstResidual, minResidualSq, lastResidual), S08_update_cg2d (the operator right after
UPDATE_CG2D, global_ocean.90x40x15 only), and the STDOUT of the same run (INI_CG2D's cg2dNorm with 17 significant
digits, which identifies the double; the cg2d lines of every step).

Geometry inputs of INI_CG2D / UPDATE_CG2D are teacher-forced from the oracle (the Task 10 grid is bitwise equal to
them on every dumped field, mitjax/tests/test_grid.py; the grid builder cannot run on this branch until
mitjax/model/grid.py follows the m0-hardening `fortran_default(citation, name, build)` API):
  * dyG, dxG, recip_dxC, recip_dyC, rA, drF: stage G00_geometry;
  * hFacW, hFacS (and hFacC for kSurfC) at INI_CG2D time: h0FacW/S/C of G00 when the build has NONLIN_FRSURF
    (INI_MASKS_ETC sets hFac = h0Fac, ini_masks_etc.F:488-496), else hFacW/S/C of S00_begin (constant without r*);
  * kSurfC: INI_MASKS_ETC's rule from that hFacC (ini_masks_etc.F:173, 185-191), not dumped;
  * deepFac2F = 1 (set_grid_factors.F:56, deepAtmosphere = .FALSE. in every M1 variant; raises otherwise);
  * recip_Bo = 1/gBaro (INI_LINEAR_PHISURF, z coordinates: ini_linear_phisurf.F:78-89; gBaro = gravity unless set,
    ini_parms.F:482; gravity set_defaults.F:101); checked against the dumped recip_Bo where the build dumps it.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.params_io import RunParams, fortran_default
from mitjax.tests import grid_gate

SOLVER_VARIANTS = [
    ("tutorial_barotropic_gyre", "input"),
    ("tutorial_baroclinic_gyre", "input"),
    ("global_ocean.90x40x15", "input"),
    ("tutorial_global_oce_optim", "input_ad"),
]
OPERATOR = ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC")
SCALARS = ("numIters", "nIterMin", "firstResidual", "minResidualSq", "lastResidual")

oracle = grid_gate.oracle
experiment = grid_gate.experiment


_EX = {}


@jax.tree_util.register_pytree_node_class
class NS:
    """A namespace that is a pytree (its attributes are the leaves), so teacher geometry passes jit as arguments."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def tree_flatten(self):
        keys = tuple(sorted(self.__dict__))
        return tuple(self.__dict__[k] for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, leaves):
        return cls(**dict(zip(keys, leaves)))


def exchanger(exp):
    if exp not in _EX:
        _EX[exp] = grid_gate.exchanger(exp)
    return _EX[exp]


def same_bits(a, b):
    """Equal shapes, dtypes and bit patterns (NaN payloads and the sign of zeros included)."""
    a, b = np.atleast_1d(np.asarray(a)), np.atleast_1d(np.asarray(b))
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def interior_mask(sz):
    """[ny, nx] bool of the Fortran interior (cg2d_rule.interior_mask)."""
    from mitjax.ad.cg2d_rule import interior_mask as im
    return im(sz.sNx, sz.sNy, sz.OLx, sz.OLy)


def xy(a, name, sz):
    return FArray(jnp.asarray(a), name, i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))


def xyz(a, name, sz):
    return FArray(jnp.asarray(a), name, i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy),
                  k=(1, sz.Nr))


def dump2d(ds, it, stage, name):
    a = ds.field(it, stage, name)
    assert a.shape[1] == 1, (stage, name, a.shape)
    return a[:, 0]


def ksurfc(hFacC, Nr):
    """INI_MASKS_ETC's kSurfC (ini_masks_etc.F:173, 185-191): Nr+1, then for k = Nr..1 where hFacC(k) /= 0: k."""
    k_surf = np.full(hFacC.shape[:1] + hFacC.shape[2:], Nr + 1, dtype=np.int32)
    for k in range(Nr, 0, -1):
        k_surf = np.where(hFacC[:, k - 1] != 0.0, k, k_surf)
    return k_surf


def gbaro(exp, inp):
    """gBaro as INI_PARMS leaves it: the data value, else gravity (ini_parms.F:482); gravity: data, else
    set_defaults.F:101 (`9.81 _d 0`). buoyancyRelation must be OCEANIC (z coordinates)."""
    e = experiment(exp, inp)
    rp = RunParams(e.run)
    br = rp.get("data", "PARM01", "buoyancyRelation", default=fortran_default("model/src/set_defaults.F:176",
                                                                             "buoyancyRelation", e))
    if br != "OCEANIC":
        raise NotImplementedError(f"recip_Bo teacher: buoyancyRelation {br!r}")
    gravity = rp.get("data", "PARM01", "gravity", default=fortran_default("model/src/set_defaults.F:101", "gravity", e))
    return rp.get("data", "PARM01", "gBaro") if rp.has("data", "PARM01", "gBaro") else gravity


def teacher_geometry(exp, inp, hfac_stage=None, it=None):
    """(grid, surface) namespaces of FArrays for INI_CG2D / UPDATE_CG2D (module docstring). hfac_stage/it: take
    hFacW/S from that stage and iteration (e.g. S07_update_rstar_T for UPDATE_CG2D) instead of INI_CG2D's."""
    e = experiment(exp, inp)
    sz = e.cfg.size
    ds, it0, _ = oracle(exp, inp)
    keys = {k[2] for k in ds.keys(it0) if k[1] == "G00_geometry"}
    g = NS()
    for name in ("dyG", "dxG", "recip_dxC", "recip_dyC", "rA", "maskInC"):
        g.__dict__[name] = xy(dump2d(ds, it0, "G00_geometry", name), name, sz)
    drF = ds.field(it0, "G00_geometry", "drF")[0, :, 0, 0]
    g.drF = FArray(jnp.asarray(drF), "drF", k=(1, sz.Nr), tiled=False)
    if "h0FacW" in keys:
        hW, hS, hC = (ds.field(it0, "G00_geometry", n) for n in ("h0FacW", "h0FacS", "h0FacC"))
    else:
        hW, hS, hC = (ds.field(it0, "S00_begin", n) for n in ("hFacW", "hFacS", "hFacC"))
    if hfac_stage is not None:
        hW, hS = (ds.field(it, hfac_stage, n) for n in ("hFacW", "hFacS"))
    g.hFacW, g.hFacS = xyz(hW, "hFacW", sz), xyz(hS, "hFacS", sz)
    g.kSurfC = xy(ksurfc(np.asarray(hC), sz.Nr), "kSurfC", sz)
    from mitjax.model.src.cg2d_h import ini_parms_cg2d
    if ini_parms_cg2d(e).deepAtmosphere:
        raise NotImplementedError("teacher geometry: deepAtmosphere")
    g.deepFac2F = FArray(jnp.ones(sz.Nr + 1), "deepFac2F", k=(1, sz.Nr + 1), tiled=False)
    recip = 1.0 / gbaro(exp, inp)                                     # ini_linear_phisurf.F:85  1. _d 0 / gBaro
    s = NS(recip_Bo=xy(np.full(hC[:, 0].shape, recip), "recip_Bo", sz))
    if "recip_Bo" in keys:
        d = dump2d(ds, it0, "G00_geometry", "recip_Bo")
        if not np.array_equal(d, np.asarray(s.recip_Bo.data)):
            raise AssertionError(f"{exp}/{inp}: teacher recip_Bo differs from the dumped one")
    return g, s


_STDOUT_NORM = re.compile(r"INI_CG2D: CG2D normalisation factor =\s+(\S+)\s*$")


def stdout_lines(exp, inp):
    _, _, rundir = oracle(exp, inp)
    return (rundir / "output.txt").read_text(errors="replace").splitlines()


def stdout_cg2dnorm(exp, inp):
    """(the printed line without its PRINT_MESSAGE prefix, the value) of INI_CG2D's normalisation factor."""
    hits = []
    for ln in stdout_lines(exp, inp):
        m = _STDOUT_NORM.search(ln)
        if m:
            hits.append((ln[ln.index("INI_CG2D:"):], float(m.group(1).replace("D", "E"))))
    if len(hits) != 1:
        raise ValueError(f"{exp}/{inp}: {len(hits)} INI_CG2D normalisation lines in STDOUT")
    return hits[0]


def stdout_solver_lines(exp, inp):
    """The ' cg2d: Sum(rhs),rhsMax' lines (one per CG2D call) and the SOLVE_FOR_PRESSURE monitor lines
    (cg2d_init_res, cg2d_iters, cg2d_last_res), without the PRINT_MESSAGE prefix, in file order."""
    sums, mon = [], []
    for ln in stdout_lines(exp, inp):
        if ln.startswith(" cg2d: Sum(rhs),rhsMax ="):
            sums.append(ln)
        else:
            m = re.match(r"^\(PID\.TID \d+\.\d+\) (\s*cg2d_(init_res|iters\(min,last\)|min_res|last_res) =.*)$", ln)
            if m:
                mon.append(m.group(1))
    return sums, mon


def run_ini_cg2d(exp, inp):
    """INI_CG2D on the teacher geometry under jit (params traced)."""
    from mitjax.model.src.cg2d_h import ini_parms_cg2d
    from mitjax.model.src.ini_cg2d import ini_cg2d
    e = experiment(exp, inp)
    params = ini_parms_cg2d(e)
    grid, surface = teacher_geometry(exp, inp)
    ex = exchanger(exp)
    f = jax.jit(lambda grid, surface, params, ex: ini_cg2d(cfg=e.cfg, grid=grid, surface=surface, params=params, ex=ex))
    return f(grid, surface, params, ex), params


def run_update_cg2d(exp, inp, cg2dh, myIter, hfac_stage=None, it=None):
    from mitjax.model.src.cg2d_h import ini_parms_cg2d
    from mitjax.model.src.update_cg2d import update_cg2d
    e = experiment(exp, inp)
    params = ini_parms_cg2d(e)
    grid, surface = teacher_geometry(exp, inp, hfac_stage=hfac_stage, it=it)
    ex = exchanger(exp)
    f = jax.jit(lambda c, grid, surface, params, ex: update_cg2d(c, None, myIter, cfg=e.cfg, grid=grid,
                                                                  surface=surface, params=params, ex=ex))
    return f(cg2dh, grid, surface, params, ex)


def cg2dh_from_dump(exp, inp, it, cg2dNorm, stage="C01_cg2d_inputs"):
    """CG2D.h of the oracle at a stage (the six arrays dumped there), with cg2dNorm given and cg2dTolerance_sq =
    cg2dTargetResidual*cg2dTargetResidual (ini_cg2d.F:151, :163)."""
    from mitjax.model.src.cg2d_h import CG2DH, ini_parms_cg2d
    e = experiment(exp, inp)
    sz = e.cfg.size
    ds, _, _ = oracle(exp, inp)
    params = ini_parms_cg2d(e)
    arrs = {n: xy(dump2d(ds, it, stage, n), n, sz) for n in OPERATOR}
    tol = np.float64(params.cg2dTargetResidual)
    return CG2DH(**arrs, cg2dNorm=np.float64(cg2dNorm), cg2dTolerance_sq=np.float64(tol*tol),
                 cg2dNormaliseRHS=True), params


def replay_cg2d(exp, inp, it, solve="literal", cg2d_fn=None):
    """Run cg2d (solve="literal") or cg2d_solve (solve="rule") on the dumped C01 inputs of iteration `it` under jit.
    Returns the outputs of the routine as numpy."""
    from mitjax.model.src import cg2d as cg2d_mod
    e = experiment(exp, inp)
    sz = e.cfg.size
    ds, _, _ = oracle(exp, inp)
    _, norm = stdout_cg2dnorm(exp, inp)
    cg2dh, params = cg2dh_from_dump(exp, inp, it, norm)
    b = xy(dump2d(ds, it, "C01_cg2d_inputs", "cg2d_b"), "cg2d_b", sz)
    x = xy(dump2d(ds, it, "C01_cg2d_inputs", "cg2d_x"), "cg2d_x", sz)
    fn = cg2d_fn or (cg2d_mod.cg2d if solve == "literal" else cg2d_mod.cg2d_solve)
    nIterMin = params.cg2dUseMinResSol - 1                     # solve_for_pressure.F:278
    f = jax.jit(lambda b, x, c, ex: fn(b, x, params.cg2dMaxIters, nIterMin, cfg=e.cfg, cg2dh=c, params=params, ex=ex))
    out = f(b, x, cg2dh, exchanger(exp))
    return jax.tree.map(np.asarray, out)


def compare_solution(exp, inp, it, out):
    """{name: (n compared, n differing, n non-finite ours, n non-finite oracle)} of C02's cg2d_x (all points) and
    the five scalars."""
    ds, _, _ = oracle(exp, inp)
    res = {}
    ref = dump2d(ds, it, "C02_cg2d_solution", "cg2d_x")
    ours = np.asarray(out[1].data if hasattr(out[1], "data") else out[1])
    res["cg2d_x"] = (ref.size, int(np.sum(ours != ref)), int(np.sum(~np.isfinite(ours))),
                     int(np.sum(~np.isfinite(ref))))
    for k, name in enumerate(SCALARS):
        val = np.asarray(out[(5, 6, 2, 3, 4)[k]], np.float64)
        refv = ds.scalar(it, "C02_cg2d_solution", name)
        res[name] = (1, int(val != refv), int(not np.isfinite(val)), int(not np.isfinite(refv)))
    return res


def compare_operator(cg2dh, exp, inp, it, stage):
    """{array: (n compared, n differing, n non-finite ours, n non-finite oracle, n differing bit patterns)}, all
    points (halos included) of the six CG2D.h arrays vs the dump at (it, stage)."""
    ds, _, _ = oracle(exp, inp)
    res = {}
    for n in OPERATOR:
        ref = dump2d(ds, it, stage, n)
        ours = np.asarray(getattr(cg2dh, n).data)
        assert ours.shape == ref.shape, (n, ours.shape, ref.shape)
        res[n] = (ref.size, int(np.sum(ours != ref)), int(np.sum(~np.isfinite(ours))), int(np.sum(~np.isfinite(ref))),
                  int(np.sum(ours.view(np.int64) != ref.view(np.int64))))
    return res
