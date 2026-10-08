"""PTRACERS_INIT_VARIA: pkg/ptracers/ptracers_init_varia.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_XYZ_RL
from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.pkg.generic_advdiff.gad_h import nSOM
from mitjax.pkg.ptracers.ptracers_fields_h import PtracersFields
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL

_OPT = "PTRACERS_OPTIONS.h"


def ptracers_init_varia(nIter0, pickupSuff, *, cfg, grid, ptr, ex, rw, like3, like2):
    """PTRACERS_INIT_VARIA( myThid )   @63cdc0b pkg/ptracers/ptracers_init_varia.F:9-134

    C     Initialize PTRACERS data structures

    `nIter0`, `pickupSuff`: PARAMS.h (static); `like3`, `like2`: a 3-D and a 2-D FArray of the build (declarations of
    pTracer and surfaceForcingPTr). Returns (ptr, ptf): `ptr` with PTRACERS_START.h set (PTRACERS_StepFwd = .TRUE.,
    PTRACERS_startAB = nIter0 - PTRACERS_Iter0 for every tracer, :43-47) and PTRACERS_FIELDS.h `ptf`: pTracer =
    PTRACERS_ref(k), gpTrNm1 = 0, surfaceForcingPTr = 0 on every point of every tracer (:50-91), the SOM moments 0
    (:73-86, PTRACERS_ALLOW_DYN_STATE with SOM advection; nSOM moments as GAD.h), then, if nIter0 = PTRACERS_Iter0,
    each tracer's PTRACERS_initialFile read (READ_FLD_XYZ_RL) and exchanged (_EXCH_XYZ_RL, :96-105), then pTracer =
    0 where maskC = 0 (:108-121). Raise: PTRACERS_READ_PICKUP (:124-129: nIter0 > PTRACERS_Iter0, or a pickupSuff)."""
    sz = cfg.size
    num, nUse = ptr.PTRACERS_num, ptr.PTRACERS_numInUse
    ptr = ptr.replace(static=dict(PTRACERS_StepFwd=(True,) * num,                       # :45
                                  PTRACERS_startAB=(nIter0 - ptr.PTRACERS_Iter0,) * num))   # :46
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :59-61
    j2 = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                        # :67
    i2 = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                        # :68
    pT, gN, sF, som = [], [], [], []
    for n in range(num):                                                        # :50  DO iTracer = 1, PTRACERS_num
        ref = ptr.PTRACERS_ref[n]
        pT.append(like3.local("pTracer").at[i, j, k].set(ref[k]))               # :62
        gN.append(like3.local("gpTrNm1").at[i, j, k].set(0.))                   # :63
        sF.append(like2.local("surfaceForcingPTr").at[i2, j2].set(0.))          # :69
        if cfg.cpp.flag("PTRACERS_ALLOW_DYN_STATE", _OPT) and ptr.PTRACERS_SOM_Advection[n]:   # :73-86
            som.append(tuple(like3.local("som").at[i, j, k].set(0.) for _ in range(nSOM)))
        else:
            som.append(None)
    if nIter0 == ptr.PTRACERS_Iter0:                                            # :96
        for n in range(nUse):                                                   # :97
            f = ptr.PTRACERS_initialFile[n]
            if f.strip() != "":                                                 # :99
                pT[n] = READ_FLD_XYZ_RL(f, " ", pT[n], 0, rw=rw)                # :100-101
                pT[n] = EXCH_XYZ_RL(pT[n], ex=ex)                               # :102
    for n in range(nUse):                                                       # :108-121
        pT[n] = pT[n].at[i, j, k].set(jnp.where(grid.maskC[i, j, k] == 0., 0., pT[n][i, j, k]))   # :114-115
    if nIter0 > ptr.PTRACERS_Iter0 or (nIter0 == ptr.PTRACERS_Iter0 and str(pickupSuff).strip() != ""):   # :124-126
        raise NotImplementedError("PTRACERS_INIT_VARIA: PTRACERS_READ_PICKUP is not ported")
    return ptr, PtracersFields(pTracer=tuple(pT), gpTrNm1=tuple(gN), surfaceForcingPTr=tuple(sF), som=tuple(som))
