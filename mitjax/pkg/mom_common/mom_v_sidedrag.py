"""pkg/mom_common/mom_v_sidedrag.F: no-slip side-wall drag on V (MOM_V_SIDEDRAG)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_v_sidedrag.F:1


def mom_v_sidedrag(k, vFld, del2v, hFacZ, viscAh_Z, viscA4_Z, harmonic, biharmonic, useVariableViscosity,
                   vDragTerms, *, cfg, grid, params):
    """MOM_V_SIDEDRAG(bi, bj, k, vFld, del2v, hFacZ, viscAh_Z, viscA4_Z, harmonic, biharmonic, useVariableViscosity,
                      vDragTerms, myThid)   @63cdc0b pkg/mom_common/mom_v_sidedrag.F:3-141

    C Calculates the drag terms due to the no-slip condition on viscous stresses:
    C G^v_{drag} = - \\frac{2}{\\Delta x_v} (A_h v - A_4 \\nabla^2 v)
    C  k                    :: vertical level
    C  vfld                 :: meridional flow
    C  del2v                :: Laplacian of meridional flow
    C  hFacZ                :: fractional open water at vorticity points
    C  vDragTerms           :: drag term

    Returns vDragTerms. As MOM_U_SIDEDRAG: the IF on sideDragFactor (:58) and the IF on viscA4GridMax (:75-77,
    scalar branches on float namelist values) are `where`s with both operands computed (finite everywhere);
    `harmonic`, `biharmonic`, `useVariableViscosity` are not read. Note the Fortran's own u/v asymmetry, kept: the old
    branch here scales by cosFacV and, under COSINEMETH_III (defined in every M1 build), uses viscA4 (not A4tmp) and
    sqcosFacV (:86-93). #ifdef COSINEMETH_III arms (:88-92, :115-119, :122-126) follow the build; #ifdef
    NONLIN_FRSURF (:64-66, :103-105) is ported. The DIAGNOSTICS_FILL (:134-138) with useDiagnostics raises.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    hFacS, recip_hFacS, recip_drF, recip_rAs = grid.hFacS, grid.recip_hFacS, grid.recip_drF, grid.recip_rAs
    dyU, recip_dxV, drF, rAs = grid.dyU, grid.recip_dxV, grid.drF, grid.rAs
    cosFacV, sqCosFacV = grid.cosFacV, grid.sqCosFacV
    sideDragFactor = params.sideDragFactor
    viscAh, viscAhGrid, viscAhMax, deltaTMom = params.viscAh, params.viscAhGrid, params.viscAhMax, params.deltaTMom
    viscA4, viscA4Grid, viscA4Max = params.viscA4, params.viscA4Grid, params.viscA4Max
    viscA4GridMax, viscA4GridMin = params.viscA4GridMax, params.viscA4GridMin
    h0S = grid.h0FacS if cfg.cpp.flag("NONLIN_FRSURF", _OPT) else hFacS   # :64-70, :103-109
    cosmeth3 = cfg.cpp.flag("COSINEMETH_III", _OPT)

    j = loop_j(2-OLy, sNy+OLy-1)
    i = loop_i(2-OLx, sNx+OLx-1)

#--   old version (:58-95): sideDragFactor .LE. 0.
    hFacZClosedW = h0S[i, j, k] - hFacZ[i, j]
    hFacZClosedE = h0S[i, j, k] - hFacZ[i+1, j]
    Ahtmp = MIN(viscAh+viscAhGrid*rAs[i, j]/deltaTMom,              # :71-72
                viscAhMax, p="a")
    A4tmp = MIN(viscA4+viscA4Grid*(rAs[i, j]**2)/deltaTMom,          # :73-74
                viscA4Max, p="a")
    A4tmp = jnp.where(viscA4GridMax > 0.,                           # :75-77
                      MIN(A4tmp, viscA4GridMax*(rAs[i, j]**2)/deltaTMom, p="a"),  # :76
                      A4tmp)
    A4tmp = MAX(A4tmp, viscA4GridMin*(rAs[i, j]**2)/deltaTMom, p="a")   # :78
    if cosmeth3:                                                    # :86-93
        old = (-recip_hFacS[i, j, k]
               *recip_drF[k]*recip_rAs[i, j]
               *(hFacZClosedW*dyU[i, j]
                 *recip_dxV[i, j]
                 +hFacZClosedE*dyU[i+1, j]
                 *recip_dxV[i+1, j])
               *drF[k]*2.*(
                           Ahtmp*vFld[i, j]*cosFacV[j]
                          -viscA4*del2v[i, j]*sqCosFacV[j]
                          ))
    else:
        old = (-recip_hFacS[i, j, k]
               *recip_drF[k]*recip_rAs[i, j]
               *(hFacZClosedW*dyU[i, j]
                 *recip_dxV[i, j]
                 +hFacZClosedE*dyU[i+1, j]
                 *recip_dxV[i+1, j])
               *drF[k]*2.*(
                           Ahtmp*vFld[i, j]*cosFacV[j]
                          -A4tmp*del2v[i, j]*cosFacV[j]
                          ))

#--   new version (:97-129): using variable-Viscosity coeff. from MOM_CALC_VISC
    cF4 = sqCosFacV if cosmeth3 else cosFacV                        # :115-119, :122-126
    new = (-recip_hFacS[i, j, k]                                    # :110-127
           *recip_drF[k]*recip_rAs[i, j]
           *(hFacZClosedW*dyU[i, j]*recip_dxV[i, j]
             *(viscAh_Z[i, j]*vFld[i, j]*cosFacV[j]
               -viscA4_Z[i, j]*del2v[i, j]*cF4[j])
             +hFacZClosedE*dyU[i+1, j]*recip_dxV[i+1, j]
             *(viscAh_Z[i+1, j]*vFld[i, j]*cosFacV[j]
               -viscA4_Z[i+1, j]*del2v[i, j]*cF4[j])
             )*drF[k]*sideDragFactor)

    vDragTerms = vDragTerms.at[i, j].set(jnp.where(sideDragFactor <= 0., old, new))   # :58, :97, :132

    if cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics:      # :134-138
        raise NotImplementedError("MOM_V_SIDEDRAG: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")
    return vDragTerms
