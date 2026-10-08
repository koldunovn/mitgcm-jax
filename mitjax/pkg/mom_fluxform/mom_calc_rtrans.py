"""pkg/mom_fluxform/mom_calc_rtrans.F: vertical transports above U and V points (MOM_CALC_RTRANS)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_FLUXFORM_OPTIONS.h"     # mom_calc_rtrans.F:1


def mom_calc_rtrans(k, rTransU, rTransV, myTime, myIter, *, cfg, grid, params, state):
    """MOM_CALC_RTRANS(k, bi, bj, rTransU, rTransV, myTime, myIter, myThid)
    @63cdc0b pkg/mom_fluxform/mom_calc_rtrans.F:3-169

    C     | o Calculate vertical transports at interface k
    C     |   above U & V points (West & South face)
    C     | r coordinate (z or p):
    C     |  is simply half of the 2 vert. Transp at Center location
    C     | r* coordinate: less simple since
    C     |  d.eta/dt / H has to be evaluated locally at U & V points
    C     k       :: vertical level
    C     rTransU :: vertical transport (above U point)
    C     rTransV :: vertical transport (above V point)
    C     myTime  :: current time
    C     myIter  :: current iteration number

    Returns (rTransU, rTransV), and under NONLIN_FRSURF with select_rStar /= 0 (GO lane, :108-166)
    (rTransU, rTransV, state): the State carries MOM_FLUXFORM.h's COMMON /LOCAL_MOM_CALC_RTRANS/ dWtransC, dWtransU,
    dWtransV (written at k = 1, updated from interface k-1 for 2 <= k <= Nr; the caller passes the State on from
    level to level). `k` (1..Nr+1) and the PARAMS.h selectors are static; myTime, myIter are not read.
    Reads wVel (DYNVARS.h) from `state`. Ported: the k > Nr arm and the interior arm (:82-106, both builds). Not
    ported (raise): under #ifdef NONLIN_FRSURF, the k = Nr+1 real-fresh-water arm in p-coordinates (:68-81) and the
    r* modification (:111-164, select_rStar .NE. 0: the dWtrans state of the r* code) -- no M1 variant runs them
    (global_ocean.90x40x15 compiles NONLIN_FRSURF with select_rStar = 0). `0.5 _d 0` and `0.` are exact. The
    point loops run on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    rA, deepFac2F = grid.rA, grid.deepFac2F
    rhoFacF = params.rhoFacF
    wVel = state.wVel
    nonlin = cfg.cpp.flag("NONLIN_FRSURF", _OPT)

    j = loop_j(1-OLy+1, sNy+OLy)
    i = loop_i(1-OLx+1, sNx+OLx)
    if nonlin and (k == Nr+1 and
                   params.useRealFreshWaterFlux and params.usingPCoords):   # :68-81
        raise NotImplementedError("MOM_CALC_RTRANS: k = Nr+1 with useRealFreshWaterFlux in p-coordinates "
                                  "is not ported")
    elif k > Nr:                                                    # :82-91 (NONLIN_FRSURF), :84-91
        rTransU = rTransU.at[i, j].set(0.)
        rTransV = rTransV.at[i, j].set(0.)
    else:                                                           # :92-106
#-    Calculate vertical transports above U & V points (West & South face):
        rTransU = rTransU.at[i, j].set(
            0.5*(wVel[i-1, j, k]*rA[i-1, j]
                 + wVel[i, j, k]*rA[i, j]
                 )*deepFac2F[k]*rhoFacF[k])
        rTransV = rTransV.at[i, j].set(
            0.5*(wVel[i, j-1, k]*rA[i, j-1]
                 + wVel[i, j, k]*rA[i, j]
                 )*deepFac2F[k]*rhoFacF[k])

    if nonlin and params.select_rStar != 0:                         # :108-166 (GO lane)
        if cfg.cpp.DISABLE_RSTAR_CODE:
            return rTransU, rTransV, state
        g = grid
        dWtransC, dWtransU, dWtransV = state.dWtransC, state.dWtransU, state.dWtransV
        if k == 1:                                                  # :114-129  Initialise dWtrans
            jA = loop_j(1-OLy, sNy+OLy)
            iA = loop_i(1-OLx, sNx+OLx)
            dWtransC = dWtransC.at[iA, jA].set(state.rStarDhCDt[iA, jA]                   # :118-120
                                               * (g.Ro_surf[iA, jA]-g.R_low[iA, jA])
                                               * rA[iA, jA])
            dWtransU = dWtransU.at[i, j].set(0.5*(dWtransC[i-1, j]+dWtransC[i, j]))       # :125-126
            dWtransV = dWtransV.at[i, j].set(0.5*(dWtransC[i, j-1]+dWtransC[i, j]))       # :127-128
        elif k <= Nr:                                               # :131-160
            jA = loop_j(1-OLy, sNy+OLy)
            iA = loop_i(1-OLx, sNx+OLx)
            dWtransC = dWtransC.at[iA, jA].set(dWtransC[iA, jA]                            # :135-137
                                               - state.rStarDhCDt[iA, jA]*g.drF[k-1]*g.h0FacC[iA, jA, k-1]
                                               * rA[iA, jA])
            dWtransU = dWtransU.at[i, j].set(dWtransU[i, j]                                # :142-144
                                             - state.rStarDhWDt[i, j]*g.drF[k-1]*g.h0FacW[i, j, k-1]
                                             * g.rAw[i, j])
            dWtransV = dWtransV.at[i, j].set(dWtransV[i, j]                                # :145-147
                                             - state.rStarDhSDt[i, j]*g.drF[k-1]*g.h0FacS[i, j, k-1]
                                             * g.rAs[i, j])
            rTransU = rTransU.at[i, j].set(rTransU[i, j]-dWtransU[i, j]                    # :153-154
                                           + (dWtransC[i-1, j]+dWtransC[i, j])*0.5)
            rTransV = rTransV.at[i, j].set(rTransV[i, j]-dWtransV[i, j]                    # :155-156
                                           + (dWtransC[i, j-1]+dWtransC[i, j])*0.5)
        return rTransU, rTransV, state.replace(dWtransC=dWtransC, dWtransU=dWtransU, dWtransV=dWtransV)
    return rTransU, rTransV
