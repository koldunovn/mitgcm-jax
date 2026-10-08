"""Helpers of the initial-state gates (plan Task 11): the initial State of an M1 variant as INITIALISE_VARIA builds it,
compared with the oracle's state at the start of the first step (stage S00_begin of the registered dumps-on run,
groups d t a r m; and group R of G00_geometry: etaHnm1, hFac_surf*, rStarFacNm1*, rStarExp*, rStarDh*Dt, pStarFacK),
bitwise (equal bit patterns) on every point of every tile, halos included, both sides finite.

Nothing between INITIALISE_VARIA and S00_begin writes these fields: THE_MAIN_LOOP calls MAIN_DO_LOOP -> FORWARD_STEP
after INITIALISE_VARIA, and FORWARD_STEP's statements before the S00 anchor (forward_step.F:400-435) are
ALLOW_AUTODIFF-only counter settings and AUTODIFF_INADMODE_UNSET (reference/jaxdump/SUBSTEPS.md).

Exemptions: the fields written by the INITIALISE_VARIA routines that later tasks port (initialise_varia.PENDING_WRITES),
for the routines the variant executes (`executed_pending`); an exempt field is still compared and reported."""

import numpy as np

from mitjax.eesupp.exch_maps import load_maps  # noqa: F401  (exchanger built by grid_gate)
from mitjax.model.src.ini_parms import ini_parms
from mitjax.model.src.initialise_varia import PENDING_WRITES, executed_pending, initialise_varia
from mitjax.pkg.rw.read_rec import RW
from mitjax.tests import grid_gate as gg

R_GROUP = ("etaHnm1", "hFac_surfC", "hFac_surfW", "hFac_surfS", "rStarFacNm1C", "rStarFacNm1W", "rStarFacNm1S",
           "rStarExpC", "rStarExpW", "rStarExpS", "rStarDhCDt", "rStarDhWDt", "rStarDhSDt", "pStarFacK")

_P = {}


def params(exp, inp):
    if (exp, inp) not in _P:
        e = gg.experiment(exp, inp)
        ds, _, _ = gg.oracle(exp, inp)
        ex2 = gg.exch2_topology(ds) if e.cfg.cpp.ALLOW_EXCH2 else None
        _P[(exp, inp)] = ini_parms(e, ex2)
    return _P[(exp, inp)]


def build_state(exp, inp, grid=None, prm=None, ex=None):
    """(initial State, executed pending routines) of a variant."""
    e = gg.experiment(exp, inp)
    cfg = e.cfg
    prm = params(exp, inp) if prm is None else prm
    ex = gg.exchanger(exp) if ex is None else ex
    grid = gg.build_grid(exp, inp, params=prm.grid, ex=ex) if grid is None else grid
    _, _, rundir = gg.oracle(exp, inp)
    rw = RW(rundir, prm.init.readBinaryPrec, cfg.size)
    pend = executed_pending(cfg, prm)
    cg2dh = None
    if cfg.cpp.NONLIN_FRSURF:     # GO lane: CG2D.h is State under NONLIN_FRSURF (INI_CG2D's, initialise_fixed.F:246)
        from mitjax.model.src.cg2d_h import ini_parms_cg2d
        from mitjax.model.src.ini_cg2d import ini_cg2d
        from mitjax.model.src.ini_linear_phisurf import ini_linear_phisurf
        from mitjax.model.src.ini_parms import ini_parms_dyn
        g2 = ini_linear_phisurf(grid, cfg=cfg, params=ini_parms_dyn(e, prm.grid, prm.time, prm.init))
        # INITIALISE_FIXED's operator with the experiment's own exchanger (a planted INITIALISE_VARIA exchanger,
        # test_init's negative control, does not reach INITIALISE_FIXED)
        cg2dh = ini_cg2d(cfg=cfg, grid=g2, surface=g2, params=ini_parms_cg2d(e), ex=gg.exchanger(exp))
    st = initialise_varia(grid, cfg=cfg, params=prm, ex=ex, rw=rw, pending=pend, cg2dh=cg2dh)
    return st, pend


def exempt_fields(pend):
    out = {}
    for r in pend:
        for f in PENDING_WRITES[r]:
            out.setdefault(f, []).append(r)
    return out


def _leaves(state):
    """(dump name, array) for every State field (AB3 pairs as name_1, name_2)."""
    for n in state.names():
        f = getattr(state, n)
        if isinstance(f, tuple):
            for m, a in enumerate(f, start=1):
                yield f"{n}_{m}", a
        else:
            yield n, f


def compare(state, exp, inp, it=None):
    """{field: (n points, n differing (==), n non-finite ours, n non-finite oracle, n differing bit patterns,
    stage)} for every State field the oracle dumps at S00_begin (or in group R of G00_geometry)."""
    ds, it0, _ = gg.oracle(exp, inp)
    it = it0 if it is None else it
    keys = {(stage, name) for (_, stage, name) in ds.keys(it)}
    out = {}
    for name, a in _leaves(state):
        if ("S00_begin", name) in keys:
            stage = "S00_begin"
        elif name in R_GROUP and ("G00_geometry", name) in keys:
            stage = "G00_geometry"
        else:
            continue
        ref = ds.field(it, stage, name)
        ours = np.asarray(a.data)
        if ours.ndim == 3:
            ours = ours[:, None]
        if ours.shape != ref.shape:
            out[name] = ("shape", ours.shape, ref.shape, 0, 0, stage)
            continue
        o64, r64 = np.ascontiguousarray(ours, np.float64), np.ascontiguousarray(ref, np.float64)
        out[name] = (ours.size, int(np.count_nonzero(~(ours == ref))), int(np.count_nonzero(~np.isfinite(ours))),
                     int(np.count_nonzero(~np.isfinite(ref))),
                     int(np.count_nonzero(o64.view(np.int64) != r64.view(np.int64))), stage)
    return out


def failures(result, exempt=()):
    return {k: v for k, v in result.items() if k not in exempt and (v[0] == "shape" or any(v[1:5]))}
