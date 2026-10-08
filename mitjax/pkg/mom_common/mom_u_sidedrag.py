"""pkg/mom_common/mom_u_sidedrag.F: no-slip side-wall drag on U (MOM_U_SIDEDRAG)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_u_sidedrag.F:1


def mom_u_sidedrag(k, uFld, del2u, hFacZ, viscAh_Z, viscA4_Z, harmonic, biharmonic, useVariableViscosity,
                   uDragTerms, *, cfg, grid, params):
    """MOM_U_SIDEDRAG(bi, bj, k, uFld, del2u, hFacZ, viscAh_Z, viscA4_Z, harmonic, biharmonic, useVariableViscosity,
                      uDragTerms, myThid)   @63cdc0b pkg/mom_common/mom_u_sidedrag.F:3-154

    C Calculates the drag terms due to the no-slip condition on viscous stresses:
    C G^u_{drag} = - \\frac{2}{\\Delta y_u} (A_h u - A_4 \\nabla^2 u)
    C  k                    :: vertical level
    C  uFld                 :: zonal flow
    C  del2u                :: Laplacian of zonal flow
    C  hFacZ                :: fractional open water at vorticity points
    C  uDragTerms           :: drag term

    Returns uDragTerms. `harmonic`, `biharmonic`, `useVariableViscosity` are not read by the routine.
    The IF on the REAL parameter sideDragFactor (:58, a scalar branch on a float namelist value) is a `where` between
    the two branches, both computed: the old branch (:62-98, sideDragFactor <= 0) and the new one (:104-142); every
    M1 variant runs the new one (sideDragFactor = 2, set_defaults.F), the old one is gated in the replay harness with
    sideDragFactor = -1. Both branches are finite on every lane (deltaTMom > 0). `2.` is a REAL*4 literal (exact);
    `x**2` with the integer literal 2 is `lax.integer_pow` (x*x, as gfortran). #ifdef NONLIN_FRSURF (:64-66,
    :106-108) is ported (global_ocean.90x40x15); #ifdef ISOTROPIC_COS_SCALING (:85-91, :117-123, :129-135) follows
    the build (MLAdjust/code/MOM_COMMON_OPTIONS.h:22 defines it), with its #ifdef COSINEMETH_III arms
    (sqcosFacU, else cosFacU). The DIAGNOSTICS_FILL (:147-151) with
    useDiagnostics = .TRUE. raises (pkg/diagnostics is not ported). The point loops run on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    hFacW, recip_hFacW, recip_drF, recip_rAw = grid.hFacW, grid.recip_hFacW, grid.recip_drF, grid.recip_rAw
    dxV, recip_dyU, drF, rAw = grid.dxV, grid.recip_dyU, grid.drF, grid.rAw
    sideDragFactor = params.sideDragFactor
    viscAh, viscAhGrid, viscAhMax, deltaTMom = params.viscAh, params.viscAhGrid, params.viscAhMax, params.deltaTMom
    viscA4, viscA4Grid, viscA4Max = params.viscA4, params.viscA4Grid, params.viscA4Max
    viscA4GridMax, viscA4GridMin = params.viscA4GridMax, params.viscA4GridMin
    h0W = grid.h0FacW if cfg.cpp.flag("NONLIN_FRSURF", _OPT) else hFacW   # :64-70, :106-112
    iso = cfg.cpp.flag("ISOTROPIC_COS_SCALING", _OPT)               # :85, :117, :129
    if iso:
        cosFacU = grid.cosFacU
        cF4 = grid.sqCosFacU if cfg.cpp.flag("COSINEMETH_III", _OPT) else grid.cosFacU   # :87-91, :119-123, :131-135

    j = loop_j(2-OLy, sNy+OLy-1)
    i = loop_i(2-OLx, sNx+OLx-1)

#--   old version (:58-98): sideDragFactor .LE. 0.
    hFacZClosedS = h0W[i, j, k] - hFacZ[i, j]
    hFacZClosedN = h0W[i, j, k] - hFacZ[i, j+1]
    Ahtmp = MIN(viscAh+viscAhGrid*rAw[i, j]/deltaTMom,              # :71-72
                viscAhMax, p="a")
    A4tmp = MIN(viscA4+viscA4Grid*(rAw[i, j]**2)/deltaTMom,          # :73-74
                viscA4Max, p="a")
    A4tmp = MIN(A4tmp, viscA4GridMax*(rAw[i, j]**2)/deltaTMom, p="a")   # :75
    A4tmp = MAX(A4tmp, viscA4GridMin*(rAw[i, j]**2)/deltaTMom, p="a")   # :76
    if iso:                                                         # :85-91
        old = (-recip_hFacW[i, j, k]
               *recip_drF[k]*recip_rAw[i, j]
               *(hFacZClosedS*dxV[i, j]
                 *recip_dyU[i, j]
                 +hFacZClosedN*dxV[i, j+1]
                 *recip_dyU[i, j+1])
               *drF[k]*2.*(
                           viscAh*uFld[i, j]*cosFacU[j]
                          -viscA4*del2u[i, j]*cF4[j]
                          ))
    else:                                                           # :92-95
        old = (-recip_hFacW[i, j, k]                                # :77-96
               *recip_drF[k]*recip_rAw[i, j]
               *(hFacZClosedS*dxV[i, j]
                 *recip_dyU[i, j]
                 +hFacZClosedN*dxV[i, j+1]
                 *recip_dyU[i, j+1])
               *drF[k]*2.*(
                           Ahtmp*uFld[i, j]
                          -A4tmp*del2u[i, j]
                          ))

#--   new version (:100-142): using variable-Viscosity coeff. from MOM_CALC_VISC
    if iso:                                                         # :117-123, :129-135
        new = (-recip_hFacW[i, j, k]                                # :113-140
               *recip_drF[k]*recip_rAw[i, j]
               *(hFacZClosedS*dxV[i, j]*recip_dyU[i, j]
                 *(viscAh_Z[i, j]*uFld[i, j]*cosFacU[j]
                   -viscA4_Z[i, j]*del2u[i, j]*cF4[j])
                 +hFacZClosedN*dxV[i, j+1]*recip_dyU[i, j+1]
                 *(viscAh_Z[i, j+1]*uFld[i, j]*cosFacU[j]
                   -viscA4_Z[i, j+1]*del2u[i, j]*cF4[j])
                 )*drF[k]*sideDragFactor)
    else:                                                           # :124-127, :136-139
        new = (-recip_hFacW[i, j, k]                                # :113-140
               *recip_drF[k]*recip_rAw[i, j]
               *(hFacZClosedS*dxV[i, j]*recip_dyU[i, j]
                 *(viscAh_Z[i, j]*uFld[i, j]
                   -viscA4_Z[i, j]*del2u[i, j])
                 +hFacZClosedN*dxV[i, j+1]*recip_dyU[i, j+1]
                 *(viscAh_Z[i, j+1]*uFld[i, j]
                   -viscA4_Z[i, j+1]*del2u[i, j])
                 )*drF[k]*sideDragFactor)

    uDragTerms = uDragTerms.at[i, j].set(jnp.where(sideDragFactor <= 0., old, new))   # :58, :100, :145

    if cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics:      # :147-151
        raise NotImplementedError("MOM_U_SIDEDRAG: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")
    return uDragTerms
