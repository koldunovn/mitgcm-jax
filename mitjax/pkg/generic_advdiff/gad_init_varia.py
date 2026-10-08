"""GAD_INIT_VARIA: pkg/generic_advdiff/gad_init_varia.F @63cdc0b (the SOM moments at the start of a run)."""

from mitjax.farray import loops_kji

_OPT = "GAD_OPTIONS.h"      # gad_init_varia.F:1  #include "GAD_OPTIONS.h"


def gad_init_varia(state, *, cfg, params, startTime, baseTime, nIter0, pickupSuff):
    """GAD_INIT_VARIA( myThid )   @63cdc0b pkg/generic_advdiff/gad_init_varia.F:6-75

    C     | SUBROUTINE GAD_INIT_VARIA
    C     | o Initialise to zero all GAD_SOM fields

    Returns the State with som_T, som_S (GAD_SOM_VARS.h) set. Without GAD_ALLOW_TS_SOM_ADV the routine is empty
    (:33) and returns its input. :41-56 every moment 0. _d 0 on every point; :58-71 with tempSOM_Advection or
    saltSOM_Advection a cold start (startTime.EQ.baseTime .AND. nIter0.EQ.0 .AND. pickupSuff.EQ.' ') does nothing
    more; a restart reads the moments (GAD_READ_PICKUP, then GAD_SOM_EXCHANGES): not ported, raises. The host
    values startTime, baseTime, nIter0, pickupSuff are the run's (PARAMS.h); the k/j/i nest is pointwise."""
    if not (cfg.cpp.ALLOW_GENERIC_ADVDIFF and cfg.cpp.flag("GAD_ALLOW_TS_SOM_ADV", _OPT)):
        return state
    sz = cfg.size
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    som_T = tuple(a.at[i, j, k].set(0.) for a in state.som_T)                  # :41-56  0. _d 0
    som_S = tuple(a.at[i, j, k].set(0.) for a in state.som_S)
    if params.tempSOM_Advection or params.saltSOM_Advection:                    # :58
        if not (startTime == baseTime and nIter0 == 0 and str(pickupSuff).strip() == ""):   # :59-60
            raise NotImplementedError("GAD_INIT_VARIA: GAD_READ_PICKUP of the SOM moments (restart, :68-69) is "
                                      "not ported")
    return state.replace(som_T=som_T, som_S=som_S)
