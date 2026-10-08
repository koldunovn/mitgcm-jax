"""CTRL_MAP_INI_GENTIM2D   @63cdc0b pkg/ctrl/ctrl_map_ini_gentim2d.F:9-506"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.ctrl.ctrl_bound import ctrl_bound_2d
from mitjax.pkg.ctrl.ctrl_get_mask import ctrl_get_mask2d
from mitjax.pkg.ctrl.ctrl_init_rec import ctrl_init_rec
from mitjax.pkg.ctrl.ctrl_readparms import fstr_blank, maxCtrlProc
from mitjax.pkg.ctrl.ctrl_toolbox import xy


def read_interior(target, record, sz):
    """What an MDS read of a tiled (or global) 2-D record does to its target (ACTIVE_READ_XY, READ_REC_XY_RL,
    READ_REC_3D_RL with one level): the interior 1..sNx, 1..sNy gets the record, the halos keep their values (MDS
    files hold no halos). `record` is an FArray whose interior holds the file's values."""
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    return target.at[i, j].set(record[i, j])


def ctrl_map_ini_gentim2d(xx_in, weight_in, *, cfg, gentim2d, maskC, ex, startTime, endTime, useCAL=False, cal=None):
    """CTRL_MAP_INI_GENTIM2D( myThid )

    C     | Dimensionalize and preprocess time variable controls.

    The control files are arrays here (the JAX control vector): `xx_in[iarr]` is the list of the records
    startrec..endrec of `xx_<name>.<optimcycle>` (the first-guess / perturbed control, the file CTRL_INIT_CTRLVAR
    writes as zeros with doInitXX), each an FArray whose interior holds the record; `weight_in[iarr]` the record of
    `xx_gentim2d_weight(iarr)` that READ_REC_3D_RL reads (:440-441). The file `xx_<name>.effective.<optimcycle>`
    that this routine writes (:238, :484) and CTRL_GET_GEN reads is returned as `effective[iarr]`, the list of its
    records (FArrays; only their interiors are meaningful, as in the file). Also returns `wgentim2d[iarr]` (CTRL_GENARR.h
    common; only the interior is read, by CTRL_GET_GEN through `genweight`) and `startdate[iarr]`.

    Branches of this build: ALLOW_OPENAD, ALLOW_ECCO, ALLOW_CTRL_DEBUG, ALLOW_SMOOTH undefined (not compiled);
    ALLOW_AUTODIFF defined: ACTIVE_READ_XY / ACTIVE_WRITE_XY (forward mode: plain record reads and writes of the
    interior). Preprocessing options other than none raise (`docycle`, `WC01`, `smooth`, `noscaling`,
    `variaweight` are not ported).
    Lane M4ADCOL (1D_ocean_ice_column/code_ad): ALLOW_ECCO defined: xx_gen_tmp = 0 (:103, a local not read
    otherwise here) and the 'rmcycle' block (:248-402): without an 'rmcycle' preproc replicated_ntimes = 0 and the
    IF (replicated_ntimes.GT.0) block (:272-402) is skipped ('rmcycle' raises with the other preprocs); useCAL with
    `cal` (cal.h) in CTRL_INIT_REC."""
    sz = cfg.size
    if cfg.cpp.ALLOW_OPENAD or cfg.cpp.ALLOW_SMOOTH:
        raise NotImplementedError("CTRL_MAP_INI_GENTIM2D: ALLOW_OPENAD/ALLOW_SMOOTH not ported")
    # the tile count of the arrays this program holds (a device's block under shard_map; P=1: nSx*nSy)
    nT = next(iter(xx_in.values()))[0].data.shape[0] if xx_in else ex.layout.nTiles
    zero = jnp.zeros((nT, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx), jnp.float64)
    xx_gen = xy(zero, "xx_gen", sz)                                     # :97-108 xx_gen = 0. _d 0 (all points)
    mask2D = xy(zero, "mask2D", sz)
    effective, wgentim2d, startdate = {}, {}, {}
    for g in gentim2d:                                                  # :111 DO iarr = 1, maxCtrlTim2D
        iarr = g.iarr
        if fstr_blank(g.xx_gentim2d_weight):                            # :118
            continue
        startdate[iarr], diffrec, startrec, endrec = ctrl_init_rec(     # :124-131
            g.xx_gentim2d_file, g.xx_gentim2d_startdate1, g.xx_gentim2d_startdate2, g.xx_gentim2d_period, 1,
            useCAL=useCAL, startTime=startTime, endTime=endTime, cal=cal)
        doscaling = True                                                # :144-146
        for k2 in range(1, maxCtrlProc + 1):                            # :149-164
            p = g.xx_gentim2d_preproc[k2 - 1].strip()
            if p in ("WC01", "smooth", "noscaling", "docycle", "variaweight", "rmcycle"):
                raise NotImplementedError(f"CTRL_MAP_INI_GENTIM2D: preproc {p!r} not ported")
            if p:
                raise NotImplementedError(f"CTRL_MAP_INI_GENTIM2D: unknown preproc {p!r}")
        replicated_nrec = diffrec                                       # :179
        replicated_ntimes = 0                                           # :180
        if len(xx_in[iarr]) != endrec - startrec + 1:
            raise ValueError(f"xx_in[{iarr}]: {len(xx_in[iarr])} records, the Fortran reads {startrec}..{endrec}")
        eff = [None] * diffrec
        for jrec in range(1, replicated_ntimes + 2):                    # :201
            for iRec in range(1, replicated_nrec + 1):                  # :202
                kRec = replicated_nrec * (jrec - 1) + iRec              # :206
                lRec = startrec + iRec - 1                              # :207
                if kRec <= endrec:                                      # :208
                    xx_gen = read_interior(xx_gen, xx_in[iarr][lRec - startrec], sz)     # :223-225
                    eff[kRec - 1] = xx_gen                              # :238-239 ACTIVE_WRITE_XY (interior)
        if cfg.cpp.ALLOW_ECCO:                                          # :248-402 (lane M4ADCOL)
            replicated_nrec = diffrec                                   # :250
            replicated_ntimes = 0                                       # :251 ('rmcycle' raised above, :252-264)
            if replicated_ntimes > 0:                                   # :272
                raise NotImplementedError("CTRL_MAP_INI_GENTIM2D: rmcycle (:272-402) not ported")
        w = xy(zero, "wgentim2d", sz)                                   # CTRL_GENARR.h common (zero storage)
        for iRec in range(1, diffrec + 1):                              # :410
            xx_gen = read_interior(xx_gen, eff[iRec - 1], sz)           # :423-425 ACTIVE_READ_XY
            w = read_interior(w, weight_in[iarr], sz)                   # :432-441 jrec=1 (no 'variaweight')
            mask2D = ctrl_get_mask2d(g.xx_gentim2d_file, mask2D, cfg=cfg, maskC=maskC)   # :444
            j = loop_j(1, sz.sNy)                                       # :453-469
            i = loop_i(1, sz.sNx)
            keep = (mask2D[i, j] != 0.0) & (w[i, j] > 0.0)              # :457-458 (REAL*4 0. is exact)
            safe_w = jnp.where(keep, w[i, j], 1.0)                      # guard: sqrt/divide of land lanes stay finite
            scaled = xx_gen[i, j] / jnp.sqrt(safe_w) if doscaling else xx_gen[i, j]       # :459-462
            xx_gen = xx_gen.at[i, j].set(jnp.where(keep, scaled, 0.0))  # :464 else 0. _d 0
            xx_gen = ctrl_bound_2d(xx_gen, mask2D, g.xx_gentim2d_bounds, sz=sz)          # :472-473
            xx_gen = xy(ex.EXCH_XY_RL(xx_gen.data), "xx_gen", sz)       # :475 EXCH_XY_RL
            eff[iRec - 1] = xx_gen                                      # :484-485 ACTIVE_WRITE_XY (interior)
        effective[iarr] = eff
        wgentim2d[iarr] = w
    return effective, wgentim2d, startdate
