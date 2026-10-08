"""GMREDI_CALC_QGLEITH: pkg/gmredi/gmredi_calc_qgleith.F @63cdc0b (M3 lane MLAdjust, input.QGLthGM)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import halfRL
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.scan_k import scan_levels
from mitjax.pkg.mom_common.mom_calc_hdiv import mom_calc_hdiv
from mitjax.pkg.mom_common.mom_calc_hfacz import mom_calc_hfacz
from mitjax.pkg.mom_common.mom_calc_relvort3 import mom_calc_relvort3
from mitjax.pkg.mom_common.mom_calc_visc import PI, _sqrt_ad, _sqrt_plain
from mitjax.pkg.mom_common.mom_visc_qgl import mom_visc_qgl_limit, mom_visc_qgl_stretch

_OPT = "GMREDI_OPTIONS.h"           # gmredi_calc_qgleith.F:1


def gmredi_calc_qgleith(leithQG_K, myTime, myIter, *, cfg, grid, params, state, visc):
    """GMREDI_CALC_QGLEITH( leithQG_K, bi, bj, myTime, myIter, myThid )
    @63cdc0b pkg/gmredi/gmredi_calc_qgleith.F:7-276

    C     | Calculate QG Leith contribution to GMRedi tensor.
    C     | leithQG_K is located at the cell centre.

    leithQG_K: the GMREDI.h array GM_LeithQG_K (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr), tiled; returned with the points the
    routine writes (:216-264: i 2-OLx..sNx+OLx-1, j 2-OLy..sNy+OLy-1, every k) replaced, every other point as it came
    in. `state`: DYNVARS.h (uVel, vVel; sigmaRfield for MOM_VISC_QGL_STRETCH); `visc`: MOM_VISC.h (L3_D); `params`:
    PARAMS.h (useFullLeith, viscC2LeithQG, useCubedSphereExchange static). Under #ifdef ALLOW_GM_LEITH_QG and
    ALLOW_MOM_COMMON (:43-44); without either the routine has no body and leithQG_K comes back unchanged.

    The locals (:47-58) are zero at every point before the k loop (:78-93) and carry from level to level, as in the
    Fortran: points a level's loops do not write keep the previous level's value (or the zero). The k loop is a level
    scan (KERNEL_GUIDE §4): the body, written for one level, calls the per-level MOM_COMMON kernels in order; k = 1
    and k = Nr are peeled (MOM_VISC_QGL_STRETCH selects its arms on them), k = 2 .. Nr-1 run in `scan_levels`.
    #ifdef ALLOW_MOM_VECINV (:115-124) masks vort3 with hFacZ (the MLAdjust builds compile pkg/mom_vecinv). Not
    ported (raise): the cube corner fills
    FILL_CS_CORNER_TR_RL (:142-147, :156-161; useCubedSphereExchange), ALLOW_OBCS (:172-174, :184-186, :197-199,
    :209-211). The DIAGNOSTICS_FILL 'GM_LTHQG' (:265-270) is output only (useDiagnostics is .FALSE. for every kernel).
    SQRT (:236-242): the #ifdef ALLOW_AUTODIFF arm and the plain arm as mom_calc_visc's `_sqrt_ad` / `_sqrt_plain`
    (SQRT's value on every lane, finite derivative). MAX (:248-255, approximate gradients) from the build's minmax
    sites (p="a" at each of the six sites).
    """
    if not cfg.cpp.flag("ALLOW_GM_LEITH_QG", _OPT) or not cfg.cpp.ALLOW_MOM_COMMON:     # :43-44
        return leithQG_K
    if params.useCubedSphereExchange:
        raise NotImplementedError("GMREDI_CALC_QGLEITH: FILL_CS_CORNER_TR_RL (useCubedSphereExchange) is not ported")
    if cfg.cpp.flag("ALLOW_OBCS", _OPT):
        raise NotImplementedError("GMREDI_CALC_QGLEITH: ALLOW_OBCS is not ported")
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    g = grid
    kw = dict(cfg=cfg, grid=grid)
    sqrt_ = _sqrt_ad if cfg.cpp.ALLOW_AUTODIFF else _sqrt_plain

#--   Initialise terms
    if params.useFullLeith:                                         # :64-75
#     Uses correct calculation for gradients, but might not work on cube sphere
        leithQG2fac = (params.viscC2LeithQG/PI)**6                  # :70
    else:
#     Uses approximate gradients, but works on cube sphere.
        leithQG2fac = (params.viscC2LeithQG/PI)**3                  # :74

    z2 = g.R_low.local                                              # a (i,j) local of this program's tiles
    jA = loop_j(1-OLy, sNy+OLy)                                     # :78-93
    iA = loop_i(1-OLx, sNx+OLx)
    hFacZ = z2("hFacZ").at[iA, jA].set(0.)                          # 0. _d 0
    r_hFacZ = z2("r_hFacZ").at[iA, jA].set(0.)
    uFld = z2("uFld").at[iA, jA].set(0.)
    vFld = z2("vFld").at[iA, jA].set(0.)
    stretching = z2("stretching").at[iA, jA].set(0.)
    Nsquare = z2("Nsquare").at[iA, jA].set(0.)
    vort3 = z2("vort3").at[iA, jA].set(0.)
    hDiv = z2("hDiv").at[iA, jA].set(0.)
    divDx = z2("divDx").at[iA, jA].set(0.)
    divDy = z2("divDy").at[iA, jA].set(0.)
    vrtDx = z2("vrtDx").at[iA, jA].set(0.)
    vrtDy = z2("vrtDy").at[iA, jA].set(0.)

    L3_D = visc.L3_D

    # DO k=1,Nr (:96): the level body below, k = 1 and k = Nr as static calls (MOM_VISC_QGL_STRETCH branches on
    # them) and k = 2 .. Nr-1 in a level scan (KERNEL_GUIDE §4: mitjax/ops/scan_k.scan_levels; k is then a traced
    # level index). `c` carries the locals the Fortran carries from level to level and the output leithQG_K.
    def level_k(k, c):
        (hFacZ, r_hFacZ, uFld, vFld, stretching, Nsquare, vort3, hDiv, divDx, divDy, vrtDx, vrtDy,
         leithQG_K) = c
        rdf = g.recip_deepFacC[k]
        deepFac3 = g.deepFac2C[k]*g.deepFacC[k]                     # :98

#--     Calculate open water fraction at vorticity points
        hFacZ, r_hFacZ = mom_calc_hfacz(k, hFacZ, r_hFacZ, **kw)   # :101

#       Make local copies of horizontal flow field
        uFld = uFld.at[iA, jA].set(state.uVel[iA, jA, k])          # :104-109
        vFld = vFld.at[iA, jA].set(state.vVel[iA, jA, k])

        vort3 = mom_calc_relvort3(k, uFld, vFld, hFacZ, vort3, params=params, **kw)   # :111
        hDiv = mom_calc_hdiv(k, 2, uFld, vFld, hDiv, **kw)        # :112

        if cfg.cpp.ALLOW_MOM_VECINV:                                # :115-124  mask vort3
            vort3 = vort3.at[iA, jA].set(jnp.where(hFacZ[iA, jA] == 0., 0., vort3[iA, jA]))   # zeroRS; 0.

#     Having calculated the quantitites, use them to calculate LeithQG coefficient
        stretching, Nsquare = mom_visc_qgl_stretch(k, stretching, Nsquare, myTime, myIter,   # :129-131
                                                   params=params, state=state, **kw)
        stretching = mom_visc_qgl_limit(k, stretching, Nsquare, uFld, vFld, vort3,          # :132-135
                                        myTime, myIter, **kw)

#--     horizontal gradient of horizontal divergence:
        j = loop_j(2-OLy, sNy+OLy-1)                                # :148-153  gradient in x direction
        i = loop_i(2-OLx, sNx+OLx-1)
        divDx = divDx.at[i, j].set((hDiv[i, j]-hDiv[i-1, j])
                                   * g.recip_dxC[i, j]*rdf)
        divDy = divDy.at[i, j].set((hDiv[i, j]-hDiv[i, j-1])        # :162-167  gradient in y direction
                                   * g.recip_dyC[i, j]*rdf)

#       horizontal gradient of vorticity and vortex stretching:
        j = loop_j(2-OLy, sNy+OLy)                                  # :172-191  gradient in x direction
        i = loop_i(2-OLx, sNx+OLx-1)
        vrtDx = vrtDx.at[i, j].set(
            (vort3[i+1, j]-vort3[i, j])
            * g.recip_dxG[i, j]*rdf
            * g.maskS[i, j, k]
#        Average d/dx of stretching onto V-points to match vrtDX
            + halfRL*halfRL
            * ((stretching[i+1, j]-stretching[i, j])
               * g.recip_dxC[i+1, j]*rdf
               + (stretching[i, j]-stretching[i-1, j])
               * g.recip_dxC[i, j]*rdf
               + (stretching[i+1, j-1]-stretching[i, j-1])
               * g.recip_dxC[i+1, j-1]*rdf
               + (stretching[i, j-1]-stretching[i-1, j-1])
               * g.recip_dxC[i, j-1]*rdf
               )*g.maskS[i, j, k])
        j = loop_j(2-OLy, sNy+OLy-1)                                # :193-214  gradient in y direction
        i = loop_i(2-OLx, sNx+OLx)
        vrtDy = vrtDy.at[i, j].set(
            (vort3[i, j+1]-vort3[i, j])
            * g.recip_dyG[i, j]*rdf
            * g.maskW[i, j, k]
#        Average d/dy of stretching onto U-points to match vrtDy
            + halfRL*halfRL
            * ((stretching[i, j+1]-stretching[i, j])
               * g.recip_dyC[i, j+1]*rdf
               + (stretching[i, j]-stretching[i, j-1])
               * g.recip_dyC[i, j]*rdf
               + (stretching[i-1, j+1]-stretching[i-1, j])
               * g.recip_dyC[i-1, j+1]*rdf
               + (stretching[i-1, j]-stretching[i-1, j-1])
               * g.recip_dyC[i-1, j]*rdf
               )*g.maskW[i, j, k])

        j = loop_j(2-OLy, sNy+OLy-1)                                # :216-264
        i = loop_i(2-OLx, sNx+OLx-1)
#     These are (powers of) length scales
        L3 = L3_D[i, j]*deepFac3                                    # :220
        if params.useFullLeith:                                     # :222-244
#     This is the vector magnitude of the vorticity gradient squared
            grdVrt = 0.25*((vrtDx[i, j+1]*vrtDx[i, j+1]             # 0.25 _d 0
                            + vrtDx[i, j]*vrtDx[i, j])
                           + (vrtDy[i+1, j]*vrtDy[i+1, j]
                              + vrtDy[i, j]*vrtDy[i, j]))
#     This is the vector magnitude of grad (div.v) squared
            grdDiv = 0.25*((divDx[i+1, j]*divDx[i+1, j]
                            + divDx[i, j]*divDx[i, j])
                           + (divDy[i, j+1]*divDy[i, j+1]
                              + divDy[i, j]*divDy[i, j]))
            sqarg = leithQG2fac*(grdVrt+grdDiv)                     # :235
            sqarg = sqrt_(sqarg)                                    # :236-242
            leithQG_K = leithQG_K.at[i, j, k].set(sqarg*L3)         # :244
        else:                                                       # :246-258
#     but this approximation will work on cube (and differs by as much as 4X)
            grdVrt = MAX(jnp.abs(vrtDx[i, j+1]), jnp.abs(vrtDx[i, j]), p="a")   # :248
            grdVrt = MAX(grdVrt, jnp.abs(vrtDy[i+1, j]), p="a")                 # :249
            grdVrt = MAX(grdVrt, jnp.abs(vrtDy[i, j]), p="a")                   # :250
#     This approximation is good to the same order as above...
            grdDiv = MAX(jnp.abs(divDx[i+1, j]), jnp.abs(divDx[i, j]), p="a")   # :253
            grdDiv = MAX(grdDiv, jnp.abs(divDy[i, j+1]), p="a")                 # :254
            grdDiv = MAX(grdDiv, jnp.abs(divDy[i, j]), p="a")                   # :255
            leithQG_K = leithQG_K.at[i, j, k].set(leithQG2fac*(grdVrt + grdDiv)*L3)   # :257
        return (hFacZ, r_hFacZ, uFld, vFld, stretching, Nsquare, vort3, hDiv, divDx, divDy, vrtDx, vrtDy,
                leithQG_K)

    c = (hFacZ, r_hFacZ, uFld, vFld, stretching, Nsquare, vort3, hDiv, divDx, divDy, vrtDx, vrtDy, leithQG_K)
    c = level_k(1, c)
    if Nr > 1:
        c = scan_levels(level_k, c, 2, Nr-1)
        c = level_k(Nr, c)
    leithQG_K = c[-1]
    return leithQG_K
