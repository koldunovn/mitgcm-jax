"""SEAICE_COST_SENSI   @63cdc0b pkg/seaice/seaice_cost_sensi.F:3-33 (lane M4ADCOL)"""

from mitjax.pkg.seaice.seaice_cost_test import seaice_cost_test


def seaice_cost_sensi(cost, myTime, myIter, *, cfg, sp, AREA, HEFF, rA, endTime, startTime, lastinterval,
                      deltaTClock):
    """SEAICE_COST_SENSI( myTime, myIter, myThid )

    C     | o driver for seaice sensitivity cost functions

    ALLOW_COST (:22): SEAICE_COST_TEST (:25), then SEAICE_COST_ACCUMULATE_MEAN (:27-28), whose body
    (seaice_cost_accumulate_mean.F:36-66) is compiled only under ALLOW_SEAICE_COST_EXPORT (undefined in
    1D_ocean_ice_column/code_ad: the routine does nothing; the flag raises)."""
    cost = seaice_cost_test(cost, myTime, myIter, cfg=cfg, sp=sp, AREA=AREA, HEFF=HEFF, rA=rA,     # :25
                            endTime=endTime, startTime=startTime, lastinterval=lastinterval,
                            deltaTClock=deltaTClock)
    if cfg.cpp.flag("ALLOW_SEAICE_COST_EXPORT", "SEAICE_OPTIONS.h"):                              # :27-28
        raise NotImplementedError("SEAICE_COST_ACCUMULATE_MEAN: ALLOW_SEAICE_COST_EXPORT is not ported")
    return cost
