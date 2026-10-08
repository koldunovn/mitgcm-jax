"""GAD_DIFF_Y: meridional diffusive flux (@63cdc0b pkg/generic_advdiff/gad_diff_y.F)."""

from mitjax.farray import loop_i, loop_j


def gad_diff_y(k, yA, diffKh, tracer, dfy, *, cfg, grid):
    """GAD_DIFF_Y( bi,bj,k, yA, diffKh, tracer, dfy, myThid )   @63cdc0b pkg/generic_advdiff/gad_diff_y.F:7-88

    C !DESCRIPTION:
    C Calculates the area integrated meridional flux due to down-gradient diffusion
    C of a tracer:
    C F^y_{diff} = - A^y \\kappa_h \\frac{1}{\\Delta y_c} \\delta_j \\theta
    C !INPUT PARAMETERS:
    C  k                    :: vertical level
    C  yA                   :: area of face at V points
    C  diffKh               :: horizontal diffusivity
    C  tracer               :: tracer field
    C !OUTPUT PARAMETERS:
    C  dfx                  :: meridional diffusive flux

    `diffKh` is a traced float. GRID.h: recip_dyC (`_recip_dyC` expands to recip_dyC(i,j,bi,bj) in every M1 build),
    recip_deepFacC, cosFacV. Ported: the branch without ALLOW_SMAG_3D_DIFFUSIVITY (:73-82), with and without
    ISOTROPIC_COS_SCALING (:78-80; defined in advect_xy's GAD_OPTIONS.h, undefined in the other M1 builds); the
    ALLOW_SMAG_3D_DIFFUSIVITY branch (:56-71) raises: no M1 build defines it. Unary minus as in GAD_DIFF_X. The (i,j)
    nest runs on the whole range at once: each point reads only inputs. `0.` is a REAL*4 literal, exact.
    """
    if cfg.ALLOW_SMAG_3D_DIFFUSIVITY:
        raise NotImplementedError("GAD_DIFF_Y: ALLOW_SMAG_3D_DIFFUSIVITY is not ported")
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    recip_dyC, recip_deepFacC, cosFacV = grid.recip_dyC, grid.recip_deepFacC, grid.cosFacV

    i = loop_i(1-OLx, sNx+OLx)                                      # :53-55
    dfy = dfy.at[i, 1-OLy].set(0.)
    j = loop_j(1-OLy+1, sNy+OLy)                                    # :73-82
    if cfg.ISOTROPIC_COS_SCALING:
        dfy = dfy.at[i, j].set(-(diffKh*yA[i, j]
                    *recip_dyC[i, j]*recip_deepFacC[k]
                    *(tracer[i, j] - tracer[i, j-1])
                    *cosFacV[j]))
    else:
        dfy = dfy.at[i, j].set(-(diffKh*yA[i, j]
                    *recip_dyC[i, j]*recip_deepFacC[k]
                    *(tracer[i, j] - tracer[i, j-1])))
    return dfy
