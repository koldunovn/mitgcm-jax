"""SEAICE_BUDGET_OCEAN: pkg/seaice/seaice_budget_ocean.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def seaice_budget_ocean(UG, TSURF, myTime, myIter, *, cfg, ff):
    """SEAICE_BUDGET_OCEAN( UG, TSURF, netHeatFlux, SWHeatFlux, bi, bj, myTime, myIter, myThid )
    @63cdc0b pkg/seaice/seaice_budget_ocean.F:7-155

    C     | o Calculate surface heat fluxes over open ocean
    C     |   see Hibler, MWR, 108, 1943-1973, 1980
    C     |   If SEAICE_EXTERNAL_FLUXES is defined this routine simply
    C     |   simply copies the global fields to the seaice-local fields.

    Returns (netHeatFlux, SWHeatFlux), (1:sNx,1:sNy) as interior arrays [tile, j, i]: Qnet and Qsw of FFIELDS.h
    (:105-109, SEAICE_EXTERNAL_FLUXES). The build without SEAICE_EXTERNAL_FLUXES (:110-150) raises."""
    if not cfg.cpp.flag("SEAICE_EXTERNAL_FLUXES", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_BUDGET_OCEAN: the build without SEAICE_EXTERNAL_FLUXES is not ported")
    del UG, TSURF                                                              # (read only without EXTERNAL_FLUXES)
    sz = cfg.size
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    netHeatFlux = ff.Qnet[i, j]                                                # :108
    SWHeatFlux = ff.Qsw[i, j]                                                  # :109
    return netHeatFlux, SWHeatFlux
