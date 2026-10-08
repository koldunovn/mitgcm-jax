"""pkg/mom_common/mom_calc_hfacz.F: fractional thickness at vorticity points (MOM_CALC_HFACZ)."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MIN
from mitjax.ops.safe import safe_div

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_calc_hfacz.F:1

# mom_calc_hfacz.F:80   PARAMETER ( hZoption = 0 )
hZoption = 0


def mom_calc_hfacz(k, hFacZ, r_hFacZ, *, cfg, grid):
    """MOM_CALC_HFACZ(bi, bj, k, hFacZ, r_hFacZ, myThid)   @63cdc0b pkg/mom_common/mom_calc_hfacz.F:9-377

    C !DESCRIPTION:
    C Calculates the fractional thickness at vorticity points
    C !INPUT PARAMETERS:
    C  k          :: vertical level
    C !OUTPUT PARAMETERS:
    C  hFacZ      :: fractional thickness at vorticity points
    C  r_hFacZ    :: reciprocal

    Returns (hFacZ, r_hFacZ). Ported: the `#else /* not ALLOW_DEPTH_CONTROL */` branch (:158-374), every M1 build;
    hZoption is the PARAMETER 0 (:80), so only the ELSE arm (:215-225) runs and the cubed-sphere block (:229-358,
    `hZoption.GE.1`) never does. #ifdef ALLOW_DEPTH_CONTROL (:86-157) is not ported (raises). `_hFacW`/`_hFacS` are
    hFacW/hFacS without ALLOW_DEPTH_CONTROL (HFACW_MACROS.h:36-38). The point loops run on the whole (i,j) range:
    each point reads only grid fields and writes itself. The IF on hFacZ (:365-369) is a `where` with the
    reciprocal guarded before the division (ops/safe.py); `0.` is the REAL*4 literal zero.
    """
    if cfg.cpp.flag("ALLOW_DEPTH_CONTROL", _OPT):
        raise NotImplementedError("MOM_CALC_HFACZ: ALLOW_DEPTH_CONTROL is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    hFacW, hFacS = grid.hFacW, grid.hFacS

#--   1rst row & column are not computed: fill with zero
    i = loop_i(1-OLx, sNx+OLx)                                      # :170-172
    hFacZ = hFacZ.at[i, 1-OLy].set(0.)
    j = loop_j(2-OLy, sNy+OLy)                                      # :173-175
    hFacZ = hFacZ.at[1-OLx, j].set(0.)

#--   Calculate open water fraction at vorticity points
    if hZoption == 2:                                               # :179-195
        raise NotImplementedError("MOM_CALC_HFACZ: hZoption = 2")
    elif hZoption == 1:                                             # :196-214
        raise NotImplementedError("MOM_CALC_HFACZ: hZoption = 1")
    else:                                                           # :215-225
        j = loop_j(2-OLy, sNy+OLy)
        i = loop_i(2-OLx, sNx+OLx)
        hFacZOpen = MIN(hFacW[i, j, k],                             # :218-219
                        hFacW[i, j-1, k], p="b")
        hFacZOpen = MIN(hFacS[i, j, k], hFacZOpen, p="b")           # :220
        hFacZOpen = MIN(hFacS[i-1, j, k], hFacZOpen, p="b")         # :221
        hFacZ = hFacZ.at[i, j].set(hFacZOpen)

#     Special stuff for Cubed Sphere: IF ( useCubedSphereExchange .AND. hZoption.GE.1 ) (:229) is false (hZoption=0)

#--   Calculate reciprocal:
    j = loop_j(1-OLy, sNy+OLy)                                      # :363-371
    i = loop_i(1-OLx, sNx+OLx)
    hZ = hFacZ[i, j]
    r_hFacZ = r_hFacZ.at[i, j].set(safe_div(1., hZ, hZ != 0., 0.))

    return hFacZ, r_hFacZ
