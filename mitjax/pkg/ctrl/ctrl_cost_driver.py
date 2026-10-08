"""CTRL_COST_DRIVER   @63cdc0b pkg/ctrl/ctrl_cost_driver.F:7-165 (lane M4ADCOL)"""

from mitjax.farray import FArray
from mitjax.pkg.ctrl.ctrl_cost_gen import ctrl_cost_gen2d, ctrl_cost_gen3d
from mitjax.pkg.ctrl.ctrl_get_mask import ctrl_get_mask2d, ctrl_get_mask3d
from mitjax.pkg.ctrl.ctrl_readparms import fstr_blank


def _dodimensionalcost(preproc):
    return any(p.strip() == "noscaling" for p in preproc)              # :71-76, :137-142


def ctrl_cost_driver(*, cfg, useCtrlCostContribution, gentim2d, genarr3d, xx_tim2d, recs_tim2d, xx_arr3d,
                     wgentim2d, wgenarr3d, maskC, genarr2d=(), xx_arr2d=None, wgenarr2d=None):
    """CTRL_COST_DRIVER( myTime, myIter, myThid )

    The generic-control cost terms (Tikhonov regularisation) of the gentim2d and genarr3d controls of this build
    (ALLOW_GENTIM2D_CONTROL, ALLOW_GENARR3D_CONTROL; ALLOW_GENARR2D_CONTROL undefined in 1D_ocean_ice_column/code_ad,
    :105-131 raise when compiled). The loop over ivar = 1..maxcvars (:66) visits each control once by its type; the
    terms are independent arrays, so the visit order does not change a value. `xx_tim2d[iarr]`: {irec: FArray} the
    control records (xx_<name>.<optimcycle>), `recs_tim2d[iarr]` = (ncvarrecstart, ncvarrecsend) (:79-80; the
    'replicate' preproc, :81-87, raises); `xx_arr3d[iarr]`: the record of xx_<name>.<optimcycle>; `wgentim2d`,
    `wgenarr3d` (CTRL_GENARR.h). Returns ({iarr: (num, objf)} gentim2d, {iarr: (num, objf)} genarr3d), [tile]
    each (objf_gentim2d / num_gentim2d, objf_genarr3d / num_genarr3d of CTRL_GENARR.h; their other slots keep
    CTRL_INIT_VARIABLES' 0. _d 0). With useCtrlCostContribution .FALSE. (:65) nothing is computed.
    Lane M4ADLAB (lab_sea/code_ad, ALLOW_GENARR2D_CONTROL): the 'Arr2D' arm (:106-131) for `genarr2d`
    (ctrl_readparms_genarr(run, 2)) with `xx_arr2d[iarr]` the record of xx_<name>.<optimcycle> and `wgenarr2d[iarr]`:
    CTRL_COST_GEN2D( 1, 1, ..., zeroRL, wgenarr2d, ... ) -> returned as a third dict (only when the option is
    defined: the two-dict return of the other builds is kept)."""
    arr2d = cfg.cpp.flag("ALLOW_GENARR2D_CONTROL", "CTRL_OPTIONS.h")
    sz = cfg.size
    t2, a3, a2 = {}, {}, {}
    if not useCtrlCostContribution:                                    # :65
        return (t2, a2, a3) if arr2d else (t2, a3)
    for g in gentim2d:                                                 # :68-103 'Tim2D'
        if fstr_blank(g.xx_gentim2d_weight):                           # :78
            continue
        if any(p.strip() == "replicate" for p in g.xx_gentim2d_preproc):
            raise NotImplementedError("CTRL_COST_DRIVER: the 'replicate' preproc (:81-87) is not ported")
        startrec, endrec = recs_tim2d[g.iarr]                          # :79-80
        w = wgentim2d[g.iarr]
        mask2D = ctrl_get_mask2d(g.xx_gentim2d_file, FArray(w.data * 0.0, "mask2D", tiled=w.tiled, _dims=w.dims),
                                 cfg=cfg, maskC=maskC)                 # :90
        t2[g.iarr] = ctrl_cost_gen2d(startrec, endrec, xx_tim2d[g.iarr], w,                     # :92-100
                                     _dodimensionalcost(g.xx_gentim2d_preproc), mask2D, sz=sz)
    for g in (genarr2d if arr2d else ()):                              # :106-131 'Arr2D'
        if fstr_blank(g.weight):                                       # :120
            continue
        w = wgenarr2d[g.iarr]
        mask2D = ctrl_get_mask2d(g.file, FArray(w.data * 0.0, "mask2D", tiled=w.tiled, _dims=w.dims),
                                 cfg=cfg, maskC=maskC)                 # :118
        a2[g.iarr] = ctrl_cost_gen2d(1, 1, {1: xx_arr2d[g.iarr]}, w, _dodimensionalcost(g.preproc), mask2D,
                                     sz=sz)                            # :121-127
    for g in genarr3d:                                                 # :134-156 'Arr3D'
        if fstr_blank(g.weight):                                       # :147
            continue
        w = wgenarr3d[g.iarr]
        mask3D = ctrl_get_mask3d(g.file, FArray(w.data * 0.0, "mask3D", tiled=w.tiled, _dims=w.dims),
                                 cfg=cfg, maskC=maskC)                 # :145
        a3[g.iarr] = ctrl_cost_gen3d(xx_arr3d[g.iarr], w, _dodimensionalcost(g.preproc), mask3D, sz=sz)  # :148-153
    return (t2, a2, a3) if arr2d else (t2, a3)
