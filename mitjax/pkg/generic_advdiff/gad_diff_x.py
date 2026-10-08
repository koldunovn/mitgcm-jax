"""GAD_DIFF_X: zonal diffusive flux (@63cdc0b pkg/generic_advdiff/gad_diff_x.F)."""

from mitjax.farray import loop_i, loop_j


def gad_diff_x(k, xA, diffKh, tracer, dfx, *, cfg, grid):
    """GAD_DIFF_X( bi,bj,k, xA, diffKh, tracer, dfx, myThid )   @63cdc0b pkg/generic_advdiff/gad_diff_x.F:7-83

    C !DESCRIPTION:
    C Calculates the area integrated zonal flux due to down-gradient diffusion
    C of a tracer:
    C F^x_{diff} = - A^x \\kappa_h \\frac{1}{\\Delta x_c} \\delta_i \\theta
    C !INPUT PARAMETERS:
    C  k                    :: vertical level
    C  xA                   :: area of face at U points
    C  diffKh               :: horizontal diffusivity
    C  tracer               :: tracer field
    C !OUTPUT PARAMETERS:
    C  dfx                  :: zonal diffusive flux

    `diffKh` is a traced float (the caller's diffKhT/diffKhS). GRID.h: recip_dxC (the macro `_recip_dxC` expands to
    recip_dxC(i,j,bi,bj) in every M1 build), recip_deepFacC, cosFacU. Ported: the branch without
    ALLOW_SMAG_3D_DIFFUSIVITY (:69-77); the ALLOW_SMAG_3D_DIFFUSIVITY branch (:53-67, a run-time choice on
    smag3D_diffCoeff with DYNVARS.h smag3D_diffK) raises: no M1 build defines it. The Fortran's unary minus applies to
    the whole product (Fortran precedence; IEEE multiplication is sign-symmetric, so -(a*b) = (-a)*b bit for bit).
    The (i,j) nest runs on the whole range at once: each point reads only inputs. `0.` is a REAL*4 literal, exact.
    """
    if cfg.ALLOW_SMAG_3D_DIFFUSIVITY:
        raise NotImplementedError("GAD_DIFF_X: ALLOW_SMAG_3D_DIFFUSIVITY is not ported")
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    recip_dxC, recip_deepFacC, cosFacU = grid.recip_dxC, grid.recip_deepFacC, grid.cosFacU

    j = loop_j(1-OLy, sNy+OLy)                                      # :69-77
    dfx = dfx.at[1-OLx, j].set(0.)
    i = loop_i(1-OLx+1, sNx+OLx)
    dfx = dfx.at[i, j].set(-(diffKh*xA[i, j]
                *recip_dxC[i, j]*recip_deepFacC[k]
                *(tracer[i, j] - tracer[i-1, j])
                *cosFacU[j]))
    return dfx
