"""CTRL_READPARMS (the gentim2d part) and CTRL_SIZE.h   @63cdc0b pkg/ctrl/ctrl_readparms.F:18-818, CTRL_SIZE.h

Host-side set-up: the time-variable 2-D generic controls (`xx_gentim2d_*`, namelist CTRL_NML_GENARR of data.ctrl)
with the defaults of the Fortran's initialisation loop ctrl_readparms.F:283-308 (per array element; ported as code,
not as `fortran_default`, which takes scalar `name = literal` lines only), then the namelist values of the run
(mitjax/config). Only what the forward of tutorial_global_oce_optim/input_ad reads is ported; a set value of an
unported option raises (KERNEL_GUIDE §3).
"""

from dataclasses import dataclass

from mitjax.io.namelist import NULL

# pkg/ctrl/CTRL_SIZE.h:27-31 (not overridden in tutorial_global_oce_optim/code_ad)
maxCtrlTim2D = 1
maxCtrlProc = 1
# eesupp/inc/EEPARAMS.h:23
MAX_LEN_FNAM = 512


@dataclass(frozen=True)
class Gentim2d:
    """The CTRL_NML_GENARR settings of one gentim2d control iarr (Fortran names)."""
    iarr: int
    xx_gentim2d_file: str
    xx_gentim2d_weight: str
    xx_gentim2d_startdate1: int
    xx_gentim2d_startdate2: int
    xx_gentim2d_period: float
    xx_gentim2d_cumsum: bool
    xx_gentim2d_glosum: bool
    xx_gentim2d_preproc: tuple          # (jarr=1..maxCtrlProc)
    xx_gentim2d_bounds: tuple           # (jarr=1..5)


_SET = {"xx_gentim2d_file", "xx_gentim2d_weight", "xx_gentim2d_startdate1", "xx_gentim2d_startdate2",
        "xx_gentim2d_period", "xx_gentim2d_cumsum", "xx_gentim2d_glosum", "xx_gentim2d_preproc",
        "xx_gentim2d_bounds"}


def _elem(run, key, idx, default):
    v = run.var("data.ctrl", "CTRL_NML_GENARR", key)
    if v is None or v.value is NULL:
        return default
    return dict(v.value).get(idx, default)


def ctrl_readparms_gentim2d(run, cfg=None):
    """[Gentim2d for iarr = 1..maxCtrlTim2D]. Defaults: ctrl_readparms.F:283-308 (file and weight blank :284, :291;
    startdate1/2 = 0 :292-293; period = 0. _d 0 :294; cumsum/glosum = .FALSE. :295-296; preproc = ' ' :298;
    bounds = 0. _d 0 :306); ALLOW_OPENAD (:285-290) is undefined in this build. Other CTRL_NML_GENARR keys
    (preproc_c/_i/_r, gentim2dPrecond, mult_gentim2d) are not read by the ported forward and must not be set."""
    for (f, g, k), v in run.vars.items():
        if f == "data.ctrl" and g == "ctrl_nml_genarr" and k.startswith("xx_gentim2d") and v.value is not NULL:
            if k not in {s.lower() for s in _SET}:
                raise NotImplementedError(f"CTRL_READPARMS: data.ctrl sets {k}, not ported")
    out = []
    ntim, nproc = maxCtrlTim2D, maxCtrlProc
    if cfg is not None:                                         # GO lane: the build's CTRL_SIZE.h
        cs = ctrl_size(cfg)
        ntim, nproc = cs["maxCtrlTim2D"], cs["maxCtrlProc"]
    for iarr in range(1, ntim + 1):
        out.append(Gentim2d(
            iarr=iarr,
            xx_gentim2d_file=_elem(run, "xx_gentim2d_file", (iarr,), " "),
            xx_gentim2d_weight=_elem(run, "xx_gentim2d_weight", (iarr,), " "),
            xx_gentim2d_startdate1=_elem(run, "xx_gentim2d_startdate1", (iarr,), 0),
            xx_gentim2d_startdate2=_elem(run, "xx_gentim2d_startdate2", (iarr,), 0),
            xx_gentim2d_period=float(_elem(run, "xx_gentim2d_period", (iarr,), 0.0)),
            xx_gentim2d_cumsum=bool(_elem(run, "xx_gentim2d_cumsum", (iarr,), False)),
            xx_gentim2d_glosum=bool(_elem(run, "xx_gentim2d_glosum", (iarr,), False)),
            xx_gentim2d_preproc=tuple(_elem(run, "xx_gentim2d_preproc", (j, iarr), " ")
                                      for j in range(1, nproc + 1)),
            xx_gentim2d_bounds=tuple(float(_elem(run, "xx_gentim2d_bounds", (j, iarr), 0.0))
                                     for j in range(1, 6))))
    return out


def fstr_blank(s):
    """Fortran `s .NE. ' '` is False: a CHARACTER value that is all blanks."""
    return s.strip(" ") == ""


def fstr_prefix(s, n, lit):
    """Fortran `s(1:n) .EQ. lit` on a CHARACTER*(MAX_LEN_FNAM) value (blank padded)."""
    return s.ljust(MAX_LEN_FNAM)[:n] == lit.ljust(n)[:n]


# ---------------------------------------------------------------------------------------------------------------
# GOADK lane (M2, global_ocean.90x40x15/code_ad): the generic init. controls xx_genarr2d_* / xx_genarr3d_*
# (CTRL_SIZE.h:21-25 maxCtrlArr2D = maxCtrlArr3D = 1, not overridden in global_ocean.90x40x15/code_ad)
maxCtrlArr2D = 1
maxCtrlArr3D = 1


@dataclass(frozen=True)
class Genarr:
    """The CTRL_NML_GENARR settings of one genarr control iarr (dim 2 or 3; Fortran names without the dim)."""
    dim: int
    iarr: int
    file: str
    weight: str
    bounds: tuple           # (jarr=1..5)
    preproc: tuple          # (jarr=1..maxCtrlProc)
    preproc_c: tuple
    preproc_i: tuple
    preproc_r: tuple


def ctrl_readparms_genarr(run, dim, cfg=None):
    """[Genarr for iarr = 1..maxCtrlArr<dim>D] of xx_genarr<dim>d_*. Defaults: ctrl_readparms.F:242-259 (2-D),
    :263-280 (3-D): file and weight blank, bounds 0. _d 0, preproc / preproc_c ' ', preproc_i 0, preproc_r 0. _d 0
    (ALLOW_OPENAD undefined); then the namelist values of the run. genarr<dim>dPrecond and mult_genarr<dim>d are read
    only by the cost of the controls (ctrl_cost_gen*, not compiled into the forward of this family): setting them
    raises."""
    pre = f"xx_genarr{dim}d_"
    allowed = {pre + s for s in ("file", "weight", "bounds", "preproc", "preproc_c", "preproc_i", "preproc_r")}
    for (f, g, k), v in run.vars.items():
        if f == "data.ctrl" and g == "ctrl_nml_genarr" and v.value is not NULL and \
                (k.startswith(pre) or k.startswith(f"genarr{dim}d") or k.startswith(f"mult_genarr{dim}d")):
            if k not in allowed:
                raise NotImplementedError(f"CTRL_READPARMS: data.ctrl sets {k}, not ported")
    n = maxCtrlArr2D if dim == 2 else maxCtrlArr3D
    nproc = maxCtrlProc
    if cfg is not None:                                         # GO lane: the build's CTRL_SIZE.h
        cs = ctrl_size(cfg)
        n, nproc = cs[f"maxCtrlArr{dim}D"], cs["maxCtrlProc"]
    out = []
    for iarr in range(1, n + 1):
        out.append(Genarr(
            dim=dim, iarr=iarr,
            file=_elem(run, pre + "file", (iarr,), " "),
            weight=_elem(run, pre + "weight", (iarr,), " "),
            bounds=tuple(float(_elem(run, pre + "bounds", (j, iarr), 0.0)) for j in range(1, 6)),
            preproc=tuple(_elem(run, pre + "preproc", (j, iarr), " ") for j in range(1, nproc + 1)),
            preproc_c=tuple(_elem(run, pre + "preproc_c", (j, iarr), " ") for j in range(1, nproc + 1)),
            preproc_i=tuple(int(_elem(run, pre + "preproc_i", (j, iarr), 0)) for j in range(1, nproc + 1)),
            preproc_r=tuple(float(_elem(run, pre + "preproc_r", (j, iarr), 0.0))
                            for j in range(1, nproc + 1))))
    return out


def ctrl_size(cfg):
    """GO lane (global_ocean.cs32x15/code_ad overrides it): the PARAMETERs maxCtrlArr2D, maxCtrlArr3D, maxCtrlTim2D,
    maxCtrlProc of the CTRL_SIZE.h the build compiles (genmake2 takes the experiment's code directory's copy first,
    else pkg/ctrl/CTRL_SIZE.h:21-31)."""
    import re
    from mitjax import paths
    from mitjax.config.params import code_path
    for d in (code_path(cfg), paths.UPSTREAM / "pkg" / "ctrl"):
        f = d / "CTRL_SIZE.h"
        if f.exists():
            txt = f.read_text()
            out = {}
            for n in ("maxCtrlArr2D", "maxCtrlArr3D", "maxCtrlTim2D", "maxCtrlProc"):
                m = re.search(r"^\s+PARAMETER\s*\(\s*" + n + r"\s*=\s*(\d+)\s*\)", txt, re.M | re.I)
                if m is None:
                    raise ValueError(f"{f}: no PARAMETER {n}")
                out[n] = int(m.group(1))
            return out
    raise FileNotFoundError("CTRL_SIZE.h")

