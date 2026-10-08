"""UPDATE_ETAH: model/src/update_etah.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_XY_RL
from mitjax.farray import loop_i, loop_j


def update_etah(myTime, myIter, *, cfg, params, state, ex):
    """UPDATE_ETAH( myTime, myIter, myThid )   @63cdc0b model/src/update_etah.F:7-89

    C     | SUBROUTINE UPDATE_ETAH
    C     | o Update etaH (free-surface height at the time of the mass/tracer step) and keep etaHnm1.

    Returns the State (etaHnm1, etaH). The REAL test `implicDiv2Dflow.EQ.1. _d 0` (:54) selects the update range, so
    it is decided on the host (params.implicDiv2DFlow_eq_1, the value of Cg2dParams). ALLOW_OBCS raises."""
    if cfg.cpp.ALLOW_OBCS:
        raise NotImplementedError("UPDATE_ETAH: ALLOW_OBCS is not ported")
    sz = cfg.size
    jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                        # :45
    iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                        # :46
    etaHnm1 = state.etaHnm1.at[iA, jA].set(state.etaH[iA, jA])                  # :47
    if params.implicDiv2DFlow_eq_1:                                             # :54-60
        etaH = state.etaH.at[iA, jA].set(state.etaN[iA, jA])                    # :57
    else:                                                                       # :61-69
        j = loop_j(1, sz.sNy)
        i = loop_i(1, sz.sNx)
        etaH = state.etaH.at[i, j].set(state.etaN[i, j]                         # :64-66
                                       + (1. - params.implicDiv2DFlow)*state.dEtaHdt[i, j]
                                       * params.deltaTFreeSurf)
    if not params.implicDiv2DFlow_eq_1 or (params.useOBCS and params.nonlinFreeSurf > 0):   # :84-86
        etaH = EXCH_XY_RL(etaH, ex=ex)
    return state.replace(etaHnm1=etaHnm1, etaH=etaH)
