"""CTRL_GET_GEN   @63cdc0b pkg/ctrl/ctrl_get_gen.F:3-230"""

import jax.numpy as jnp
import numpy as np

from mitjax.config.fortran import real4
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.pkg.ctrl.ctrl_get_gen_rec import ctrl_get_gen_rec
from mitjax.pkg.ctrl.ctrl_map_ini_gentim2d import read_interior
from mitjax.pkg.ctrl.ctrl_readparms import fstr_prefix
from mitjax.pkg.ctrl.ctrl_swapffields import ctrl_swapffields


def ctrl_get_gen(xx_gen_file, xx_genstartdate, xx_genperiod, genmask, genfld, xx_gen0, xx_gen1,
                 xx_gen_remo_intercept, xx_gen_remo_slope, genweight, myTime, myIter, *, cfg, effective, clock):
    """ctrl_get_gen( xx_gen_file, xx_genstartdate, xx_genperiod, genmask, genfld, xx_gen0, xx_gen1,
                     xx_gen_dummy, xx_gen_remo_intercept, xx_gen_remo_slope, genweight, myTime, myIter, myThid )

    c     o new generic routine for reading time dependent control variables

    `effective`: the records of `xx_<name>.effective.<optimcycle>` (CTRL_MAP_INI_GENTIM2D's output) that
    ACTIVE_READ_XY reads (:116-118, :150-152; interior only, halos of the target unchanged). `clock`: host floats
    and switches of the model clock (useCAL, startTime, deltaTClock, externForcingCycle). Returns
    (genfld, xx_gen0, xx_gen1): genfld is the output, xx_gen0/xx_gen1 the CTRL_GENARR.h state the Fortran updates by
    reference. ALLOW_SMOOTH (:129-137, :158-166) and CTRL_SKIP_FIRST_TWO_ATM_REC_ALL (:184-190) are undefined;
    ALLOW_AUTODIFF is defined (ACTIVE_READ_XY, not READ_REC_XY_RL). genweight and xx_gen_remo_* are arguments the
    Fortran passes; genweight is not read here (as in the Fortran)."""
    sz = cfg.size
    if cfg.cpp.ALLOW_SMOOTH or cfg.cpp.CTRL_SKIP_FIRST_TWO_ATM_REC_ALL:
        raise NotImplementedError("CTRL_GET_GEN: ALLOW_SMOOTH / CTRL_SKIP_FIRST_TWO_ATM_REC_ALL not ported")
    # :92-100 file name (the record list `effective` stands for ctrlDir//xx_gen_file//'.effective.'//optimcycle)
    if clock.get("useCAL") and not isinstance(myIter, (int, np.integer)):
        # lane M4ADCOL: the useCAL arm of CTRL_GET_GEN_REC is host calendar arithmetic; a traced clock (a scan step)
        # reads its per-step values from the table the set-up computed (ctrl_get_gen_rec.gen_rec_table), indexed by
        # iloop - 1 = myIter - nIter0 (forward_step.F:429)
        tab = clock["table"][(tuple(xx_genstartdate), float(xx_genperiod))]
        it = myIter - clock["nIter0"]
        genfac, genfirst, genchanged, gencount0, gencount1 = (jnp.asarray(c)[it] for c in tab)   # :103-107
    else:
        genfac, genfirst, genchanged, gencount0, gencount1 = ctrl_get_gen_rec(   # :103-107
            xx_genstartdate, xx_genperiod, myTime, myIter, **clock)
    if isinstance(genfirst, (bool, np.bool_)):                         # host clock: the Fortran's IFs
        if genfirst:                                                   # :109
            xx_gen1 = read_interior(xx_gen1, effective[gencount0 - 1], sz)  # :116-118
            # :119-123 `if (.false.)`: never executed
        if genfirst or genchanged:                                     # :141
            xx_gen0, xx_gen1 = ctrl_swapffields(xx_gen0, xx_gen1, sz=sz)   # :142
            xx_gen1 = read_interior(xx_gen1, effective[gencount1 - 1], sz)  # :150-152
    else:
        # traced clock (a scan step): `first` is a run-time test of the model clock (KERNEL_GUIDE §4), so both IF
        # blocks are computed and selected with where; every operand is finite (copies and file records)
        cand = read_interior(xx_gen1, _record(effective, gencount0), sz)    # :109, :116-118
        xx_gen1 = _where(genfirst, cand, xx_gen1)
        sw0, sw1 = ctrl_swapffields(xx_gen0, xx_gen1, sz=sz)            # :141-142
        sw1 = read_interior(sw1, _record(effective, gencount1), sz)     # :150-152
        doit = genfirst | genchanged
        xx_gen0 = _where(doit, sw0, xx_gen0)
        xx_gen1 = _where(doit, sw1, xx_gen1)
    tauu = fstr_prefix(xx_gen_file, 7, "xx_tauu")                      # :175, :191
    tauv = fstr_prefix(xx_gen_file, 7, "xx_tauv")                      # :176, :192
    if (tauu or tauv) and not isinstance(gencount0, (int, np.integer)):
        raise NotImplementedError("CTRL_GET_GEN: xx_tauu/xx_tauv with a traced record counter (:183-193)")
    if (tauu or tauv) and xx_genperiod != 0.0 and gencount0 <= 2:      # :183-193 (host operands first)
        doCtrlUpdate = False                                           # :194
    else:
        doCtrlUpdate = True                                            # :196
    if tauu or tauv:                                                   # :198-199
        gensign = real4("-1.")                                         # :200 REAL*4 -1. (exact)
    else:
        gensign = real4("1.")                                          # :202 REAL*4 1. (exact)
    if doCtrlUpdate:                                                   # :207
        j = loop_j(1, sz.sNy)                                          # :212
        i = loop_i(1, sz.sNx)                                          # :213
        genfld = genfld.at[i, j].set(genfld[i, j]                      # :214-216
                                     + gensign*genfac*xx_gen0[i, j]
                                     + gensign*(1.0 - genfac)*xx_gen1[i, j])
        genfld = genfld.at[i, j].set(                                  # :217-220
            genmask[i, j]*(genfld[i, j]
                           - (xx_gen_remo_intercept + xx_gen_remo_slope*(myTime - clock["startTime"]))))
    return genfld, xx_gen0, xx_gen1


def _record(effective, count):
    """Record `count` (1-based) of the effective file: a host int indexes the list; a traced counter (useCAL table)
    selects from the stacked records (every record finite)."""
    if isinstance(count, (int, np.integer)):
        return effective[count - 1]
    r0 = effective[0]
    data = jnp.stack([r.data for r in effective])[count - 1]
    return FArray(data, r0.name, tiled=r0.tiled, _dims=r0.dims)


def _where(c, a, b):
    """FArray select (same declaration as `b`)."""
    return FArray(jnp.where(c, a.data, b.data), b.name, tiled=b.tiled, _dims=b.dims)
