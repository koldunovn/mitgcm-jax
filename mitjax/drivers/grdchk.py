"""The experiment's own controls and gradient check (lane API, docs plan 20261006 Task 6): which control the run
differentiates, where GRDCHK_MAIN checks it, the adxx files, and the gradient check's lines -- host-side bookkeeping
around the gated drivers (drivers/adjoint_run.py: GenarrAdjoint for xx_genarr2d/3d, Adjoint for xx_gentim2d).

    @63cdc0b pkg/grdchk/grdchk_readparms.F:80-97 (defaults), grdchk_ctrl_fname.F (grdchkvarname -> control),
             pkg/ctrl/ctrl_init_fixed.F:220-350 (the genarr / gentim2d controls CTRL_INIT_CTRLVAR registers),
             pkg/grdchk/grdchk_main.F:202-524, grdchk_print.F (mitjax/pkg/grdchk/grdchk_print.py),
             pkg/autodiff/active_file_control.F:143-171 (the adjoint's adxx record write)

Ported: one process (myXGlobalLo = myYGlobalLo = 1), ncvargrd 'c' (maskC) controls (grdchk_get_mask / grdchk_loc
refuse the others), the adjoint arm (no ALLOW_TANGENTLINEAR_RUN), central differences or not (useCentralDiff).
"""

from dataclasses import dataclass

import numpy as np

# grdchk_readparms.F:81-97: the defaults GRDCHK_READPARMS sets before reading data.grdchk
_GRDCHK_DEFAULTS = (("grdchk_eps", 81), ("nbeg", 82), ("nend", 83), ("nstep", 84), ("useCentralDiff", 85),
                    ("iGloPos", 87), ("jGloPos", 88), ("kGloPos", 89), ("iGloTile", 90), ("jGloTile", 91),
                    ("obcsglo", 94), ("recglo", 95), ("grdchkvarname", 96))


@dataclass(frozen=True)
class Control:
    """One control CTRL_INIT_FIXED registers (ctrl_init_fixed.F:231-261 genarr, :297-345 gentim2d)."""
    name: str               # ncvarfname: xx_genarr<d>d_file / xx_gentim2d_file (blank-stripped)
    kind: str               # "genarr" (drivers/adjoint_run.GenarrAdjoint) or "gentim2d" (Adjoint)
    dim: int                # 2 or 3 (genarr); 2 (gentim2d)
    iarr: int
    ncvargrd: str           # 'c', or 'w' / 's' for xx_fu / xx_fv (ctrl_init_fixed.F:281-284)
    ncvarnrmax: int         # Nr (Arr3D) or 1
    ncvarrecs: int          # 1 (genarr), endrec (gentim2d, :340-345)

    @property
    def key(self):
        return (self.dim, self.iarr)


def grdchk_settings(exp):
    """data.grdchk's GRDCHK_NML with GRDCHK_READPARMS' defaults (grdchk_readparms.F:81-97), and the checks of
    :122-136 (STOPs raised). exp: mitjax.config.params.Experiment."""
    from mitjax.params_io import RunParams, fortran_default
    rp = RunParams(exp.run)
    if not any(f == "data.grdchk" for f, _, _ in exp.run.vars):
        raise FileNotFoundError("OPEN_COPY_DATA_FILE: File data.grdchk does not exist! (GRDCHK_READPARMS, "
                                "pkg/grdchk/grdchk_readparms.F:104-107; eesupp/src/open_copy_data_file.F:72-76)")
    s = {n: rp.get("data.grdchk", "GRDCHK_NML", n,
                   default=fortran_default(f"pkg/grdchk/grdchk_readparms.F:{line}", n, exp))
         for n, line in _GRDCHK_DEFAULTS}
    if rp.has("data.grdchk", "GRDCHK_NML", "grdchkvarindex"):
        raise NotImplementedError("GRDCHK_CTRL_FNAME: data.grdchk sets grdchkvarindex (the ncvarindex numbering of "
                                  "CTRL_INIT_FIXED is not ported); name the control with grdchkvarname")
    sz = exp.cfg.size
    if s["iGloPos"] > sz.sNx or s["jGloPos"] > sz.sNy:                          # :122-126
        raise RuntimeError("ABNORMAL END: S/R GRDCHK_READPARMS: i/j GloPos must be <= sNx/y")
    if s["iGloTile"] > sz.nSx * sz.nPx or s["jGloTile"] > sz.nSy * sz.nPy:      # :127-131
        raise RuntimeError("ABNORMAL END: S/R GRDCHK_READPARMS: i/j GloTile must be <= nSx*nPx/y")
    if sz.nPx * sz.nPy != 1:
        raise NotImplementedError("GRDCHK_READPARMS: more than one process (nPx*nPy > 1) is not ported")
    s["iLocTile"] = s["iGloTile"]                       # :140-141 iLocTile = iGloTile - (myXGlobalLo-1)/sNx, = 1
    s["jLocTile"] = s["jGloTile"]
    return s


def controls(m):
    """The genarr and gentim2d controls of the Model `m` in CTRL_INIT_FIXED's order (ctrl_init_fixed.F:220-350): a
    control is registered when its weight file name is not blank (:231, :251, :297)."""
    from mitjax.pkg.ctrl.ctrl_init_rec import ctrl_init_rec
    from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr, ctrl_readparms_gentim2d, fstr_blank
    cfg, sz = m.cfg, m.cfg.size
    out = []
    for dim, flag in ((2, "ALLOW_GENARR2D_CONTROL"), (3, "ALLOW_GENARR3D_CONTROL")):     # :219, :247
        if not cfg.cpp.flag(flag, "CTRL_OPTIONS.h"):
            continue
        for g in ctrl_readparms_genarr(m.exp.run, dim, cfg):
            if not fstr_blank(g.weight):                                   # :231, :251
                out.append(Control(g.file.strip(), "genarr", dim, g.iarr, "c", sz.Nr if dim == 3 else 1, 1))
    if cfg.cpp.flag("ALLOW_GENTIM2D_CONTROL", "CTRL_OPTIONS.h"):                     # :267
        tp = m.prm.time
        for g in ctrl_readparms_gentim2d(m.exp.run, cfg):
            if fstr_blank(g.xx_gentim2d_weight):                              # :297
                continue
            f = g.xx_gentim2d_file
            grd = "w" if f[:5] == "xx_fu" else "s" if f[:5] == "xx_fv" else "c"            # :281-284
            _, _, startrec, endrec = ctrl_init_rec(f, g.xx_gentim2d_startdate1, g.xx_gentim2d_startdate2,
                                                   g.xx_gentim2d_period, 1, useCAL=cfg.use_flag("useCAL"),
                                                   startTime=tp.startTime, endTime=tp.endTime,
                                                   cal=getattr(m, "cal", None))             # :287-295
            if startrec != 1 or any(p.strip() == "docycle" for p in g.xx_gentim2d_preproc):
                raise NotImplementedError(f"CTRL_INIT_FIXED: gentim2d control {f.strip()} with startrec = "
                                          f"{startrec} or 'docycle' (ctrl_init_fixed.F:309-317) is not covered")
            out.append(Control(f.strip(), "gentim2d", 2, g.iarr, grd, 1, endrec))
    return out


def control_by_name(m, fName):
    """GRDCHK_CTRL_FNAME (grdchk_ctrl_fname.F): the registered control whose ncvarfname is fName."""
    for c in controls(m):
        if c.name == fName.strip():
            return c
    raise ValueError("ABNORMAL END: S/R GRDCHK_CTRL_FNAME: grdchkvarindex is not set and could not be determined "
                     f"from grdchkvarname = {fName.strip()!r} (controls of data.ctrl: "
                     f"{[c.name for c in controls(m)]})")


def grdchk_positions(m, ctl, s):
    """GRDCHK_MAIN's check positions (grdchk_main.F:227-242; with nbeg = 0 GRDCHK_GET_POSITION first): [(icomp,
    LocResult)] (mitjax/pkg/grdchk/grdchk.py), with the control's mask (GRDCHK_GET_MASK, 'c': maskC)."""
    from mitjax.pkg.grdchk import grdchk as gc
    sz = m.cfg.size
    maskC = np.asarray(m.grid.maskC.data)
    nw = gc.ctrl_init_wet_nwetctile(maskC, sz)
    gm = gc.grdchk_get_mask(nw, ncvargrd=ctl.ncvargrd, ncvarnrmax=ctl.ncvarnrmax, ncvarxmax=sz.sNx,
                            ncvarymax=sz.sNy, ncvarrecs=ctl.ncvarrecs)
    kw = dict(gm=gm, maskC=maskC, sz=sz, iLocTile=s["iLocTile"], jLocTile=s["jLocTile"], ncvargrd=ctl.ncvargrd,
              ncvarrecs=ctl.ncvarrecs, ncvarnrmax=ctl.ncvarnrmax, ncvarxmax=sz.sNx, ncvarymax=sz.sNy)
    pos = {n: s[n] for n in ("iGloPos", "jGloPos", "kGloPos", "obcsglo", "recglo")}
    return gc.grdchk_points(nbeg=s["nbeg"], nstep=s["nstep"], nend=s["nend"], position=pos, **kw)


def storage_point(ctl, r, sz):
    """The storage index of a check point in the control record's array: (tile, k, j, i) for an Arr3D control,
    (tile, j, i) otherwise (Fortran i at i-1+OLx; tile (bi, bj) = bi-1 + (bj-1)*nSx)."""
    t = r.itile - 1 + (r.jtile - 1) * sz.nSx
    if ctl.dim == 3:
        return (t, r.layer - 1, r.jtilepos - 1 + sz.OLy, r.itilepos - 1 + sz.OLx)
    return (t, r.jtilepos - 1 + sz.OLy, r.itilepos - 1 + sz.OLx)


def optimcycle(exp):
    """OPTIMCYCLE.h optimcycle: data.optim's OPTIM value, default 0 (pkg/ctrl/optim_readparms.F:79)."""
    from mitjax.params_io import RunParams, fortran_default
    return int(RunParams(exp.run).get("data.optim", "OPTIM", "optimcycle",
                                      default=fortran_default("pkg/ctrl/optim_readparms.F:79", "optimcycle", exp)))


def write_adxx(m, ctl, records, *, mds=None):
    """The adjoint's control-gradient file adxx_<name>.<optimcycle %010d> as the reverse arm of ACTIVE_READ_*_RL
    writes it (pkg/autodiff/active_file_control.F:166-171: MDS_WRITE_FIELD( adfname, prec, w_globFile = .FALSE.,
    useCurrentDir, 'RL', Nr, 1, myNr, ..., jRec, myOptimIter ); the 'ad' prefix from ADACTIVE_READ_*,
    active_file_ad.F:105-106), prec = ctrlprec (ctrl_readparms.F:224-229: 32 with CTRL_SET_PREC_32, else 64).
    records: {irec: numpy [tile, (k,) j, i]} of the gradient (the interior is written). Returns the file stem."""
    import jax.numpy as jnp

    from mitjax.drivers.run import _mds_cal
    from mitjax.farray import FArray
    from mitjax.pkg.mdsio.mdsio_write_field import MdsContext, mds_write_field
    sz = m.cfg.size
    if mds is None:
        mds = MdsContext(m.rundir, sz, exch2=bool(m.cfg.cpp.ALLOW_EXCH2), w2=getattr(m, "w2", None),
                         useSingleCpuIO=m.io.useSingleCpuIO, mdsioLocalDir=m.io.mdsioLocalDir,
                         the_run_name=m.io.the_run_name, useCal=_mds_cal(m))
    prec = 32 if m.cfg.cpp.flag("CTRL_SET_PREC_32", "CTRL_OPTIONS.h") else 64
    it = optimcycle(m.exp)
    stem = f"ad{ctl.name}.{it:010d}"
    b = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
    for irec in sorted(records):
        g = np.asarray(records[irec], dtype=np.float64)
        fld = (FArray(jnp.asarray(g), "adxx", k=(1, sz.Nr), **b) if ctl.dim == 3
               else FArray(jnp.asarray(g), "adxx", **b))
        nz = ctl.ncvarnrmax
        mds_write_field(stem, prec, False, False, "RL", nz, 1, nz, fld, irec, it, mds=mds)
    return stem
