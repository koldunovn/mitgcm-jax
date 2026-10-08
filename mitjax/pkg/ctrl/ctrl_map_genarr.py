"""CTRL_MAP_GENARR2D, CTRL_MAP_GENARR3D: pkg/ctrl/ctrl_map_genarr.F @63cdc0b (GOADK lane, M2:
global_ocean.90x40x15/code_ad: xx_theta, xx_kapgm, xx_kapredi (3-D), xx_bottomdrag (2-D))."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_XY_RL, EXCH_XYZ_RL
from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.pkg.ctrl.ctrl_bound import ctrl_bound_2d, ctrl_bound_3d
from mitjax.pkg.ctrl.ctrl_get_mask import ctrl_get_mask2d, ctrl_get_mask3d
from mitjax.pkg.ctrl.ctrl_readparms import fstr_prefix, maxCtrlProc


def _flags(g, name):
    """The preprocessing switches of :287-310 (3-D) / :88-111 (2-D). WC01 and smooth only set dowc01 / dosmooth /
    numsmo, which nothing reads without ALLOW_SMOOTH (undefined in this build: :335-341 / :136-141 not compiled);
    log10ctrl (EXP(ln10*x), :303-309, :352-358, :369-376) is not ported (raise)."""
    doscaling = True                                                # :282 / :83
    dolog10ctrl = False
    for k2 in range(1, maxCtrlProc + 1):
        if g.preproc[k2 - 1].strip() == "noscaling":                # :300-302 / :101-103
            doscaling = False
        if g.preproc_c[k2 - 1].strip() == "log10ctrl":              # :303-309 / :104-110
            dolog10ctrl = True
    if dolog10ctrl:
        raise NotImplementedError(f"{name}: preproc_c 'log10ctrl' is not ported")
    return doscaling


def ctrl_map_genarr3d(fld, iarr, *, cfg, genarr3d, xx_in, weight_in, maskC, ex):
    """CTRL_MAP_GENARR3D( fld, iarr, myThid )   @63cdc0b pkg/ctrl/ctrl_map_genarr.F:211-417

    C     | Add the generic arrays to the
    C     | corresponding model variables (3-D)

    The control files are arrays (the JAX control vector, as mitjax/pkg/ctrl/ctrl_map_ini_gentim2d.py):
    `xx_in` = the record of `xx_<name>.<optimcycle>` that ACTIVE_READ_XYZ reads (:325-326; forward mode: a plain read
    of the interior), `weight_in` = the record of `xx_genarr3d_weight(iarr)` (READ_REC_3D_RL, :321-322), both FArrays
    (k=1..Nr) whose interiors hold the files. `genarr3d`: the ctrl_readparms_genarr(run, 3) entry of iarr.
    Returns (fld, effective, wgenarr3d): fld after the map and its exchange; `effective` = the record WRITE_REC_3D_RL
    writes to `xx_<name>.effective.<optimcycle>` (:404-405; its interior is the file); wgenarr3d(:,:,:,:,:,iarr) of
    CTRL_GENARR.h (only its interior is set and read).

    Branches (global_ocean.90x40x15/code_ad): ALLOW_OPENAD, ALLOW_TAPENADE, ALLOW_SMOOTH undefined; ALLOW_AUTODIFF
    defined (ACTIVE_READ_XYZ); the file-name and output bookkeeping (:312-319) is not needed with arrays.
    Vectorisation: the scaling and the addition (:342-389) read only their own point: one k, j, i nest. The IF on the
    weight is a `where` with the SQRT guarded (a weight <= 0 lane takes SQRT(1.) and is replaced by 0. _d 0).
    """
    sz = cfg.size
    Nr, OLx, OLy, sNx, sNy = sz.Nr, sz.OLx, sz.OLy, sz.sNx, sz.sNy
    if cfg.cpp.flag("ALLOW_OPENAD", "CTRL_OPTIONS.h") or cfg.cpp.flag("ALLOW_SMOOTH", "CTRL_OPTIONS.h"):
        raise NotImplementedError("CTRL_MAP_GENARR3D: ALLOW_OPENAD / ALLOW_SMOOTH not ported")
    doscaling = _flags(genarr3d, "CTRL_MAP_GENARR3D")
    k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
    zero = jnp.zeros_like(fld.data)
    xx_gen = FArray(zero, "xx_gen", **_decl3(sz))                   # :278  CTRL_ASSIGN(xx_gen, Nr, zeroRL)
    xx_gen = xx_gen.at[i, j, k].set(0.)                             # (ctrl_toolbox.F:39-49: every point)
    wgen = FArray(zero, "wgenarr3d", **_decl3(sz))                  # CTRL_GENARR.h common
    k, j, i = loops_kji((1, Nr), (1, sNy), (1, sNx))
    wgen = wgen.at[i, j, k].set(weight_in[i, j, k])                 # :321-322 READ_REC_3D_RL (interior)
    xx_gen = xx_gen.at[i, j, k].set(xx_in[i, j, k])                 # :325-326 ACTIVE_READ_XYZ (interior)
    mask3D = ctrl_get_mask3d(genarr3d.file, FArray(zero, "mask3D", **_decl3(sz)), cfg=cfg, maskC=maskC)   # :332
    if doscaling:                                                   # :344-366
        pos = wgen[i, j, k] > 0.                                    # :349  .GT.0. (REAL*4 0., exact)
        scaled = xx_gen[i, j, k] / jnp.sqrt(jnp.where(pos, wgen[i, j, k], 1.))   # :350-351
        xx_gen = xx_gen.at[i, j, k].set(jnp.where(pos, scaled, 0.))  # :359-361  0. _d 0
    fld = fld.at[i, j, k].set(fld[i, j, k]                          # :378-385 (dolog10ctrl .FALSE.)
                              + xx_gen[i, j, k]*mask3D[i, j, k])
    fld = ctrl_bound_3d(fld, mask3D, genarr3d.bounds, sz=sz)        # :392
    if not (fstr_prefix(genarr3d.file, 7, "xx_uvel") or fstr_prefix(genarr3d.file, 7, "xx_vvel")):   # :397-399
        fld = EXCH_XYZ_RL(fld, ex=ex)
    return fld, fld, wgen                                           # :404-405 WRITE_REC_3D_RL( fnamegenOut, fld )


def ctrl_map_genarr2d(fld, iarr, *, cfg, genarr2d, xx_in, weight_in, maskC, ex):
    """CTRL_MAP_GENARR2D( fld, iarr, myThid )   @63cdc0b pkg/ctrl/ctrl_map_genarr.F:13-204

    C     | Add the generic arrays to the
    C     | corresponding model variables (2-D)

    As CTRL_MAP_GENARR3D, one level: `xx_in`, `weight_in` 2-D FArrays (ACTIVE_READ_XY :126-127, READ_REC_3D_RL with
    one level :122-123), mask2D from CTRL_GET_MASK2D (:131), CTRL_BOUND_2D (:185), EXCH_XY_RL (:187), the effective
    record WRITE_REC_3D_RL (:192-193). Returns (fld, effective, wgenarr2d)."""
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    if cfg.cpp.flag("ALLOW_OPENAD", "CTRL_OPTIONS.h") or cfg.cpp.flag("ALLOW_SMOOTH", "CTRL_OPTIONS.h"):
        raise NotImplementedError("CTRL_MAP_GENARR2D: ALLOW_OPENAD / ALLOW_SMOOTH not ported")
    doscaling = _flags(genarr2d, "CTRL_MAP_GENARR2D")
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    zero = jnp.zeros_like(fld.data)
    b2 = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    xx_gen = FArray(zero, "xx_gen", **b2).at[i, j].set(0.)          # :78  CTRL_ASSIGN(xx_gen, 1, zeroRL)
    wgen = FArray(zero, "wgenarr2d", **b2)                          # CTRL_GENARR.h common
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    wgen = wgen.at[i, j].set(weight_in[i, j])                       # :122-123 READ_REC_3D_RL (interior)
    xx_gen = xx_gen.at[i, j].set(xx_in[i, j])                       # :126-127 ACTIVE_READ_XY (interior)
    mask2D = ctrl_get_mask2d(genarr2d.file, FArray(zero, "mask2D", **b2), cfg=cfg, maskC=maskC)   # :131
    if doscaling:                                                   # :145-164
        pos = wgen[i, j] > 0.                                       # :149
        scaled = xx_gen[i, j] / jnp.sqrt(jnp.where(pos, wgen[i, j], 1.))       # :150-151
        xx_gen = xx_gen.at[i, j].set(jnp.where(pos, scaled, 0.))    # :159-161  0. _d 0
    fld = fld.at[i, j].set(fld[i, j]                                # :174-179 (dolog10ctrl .FALSE.)
                           + xx_gen[i, j]*mask2D[i, j])
    fld = ctrl_bound_2d(fld, mask2D, genarr2d.bounds, sz=sz)        # :185
    fld = EXCH_XY_RL(fld, ex=ex)                                    # :187
    return fld, fld, wgen                                           # :192-193 WRITE_REC_3D_RL( fnamegenOut, fld )


def _decl3(sz):
    return dict(i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy), k=(1, sz.Nr))
