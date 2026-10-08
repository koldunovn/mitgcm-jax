"""pkg/mom_common/mom_calc_visc.F: variable horizontal viscosities, Leith / Smagorinsky / grid-scale, with limiters
(MOM_CALC_VISC)."""

import jax.numpy as jnp
import numpy as np

from mitjax.eesupp.fill_cs_corner_tr_rl import fill_cs_corner_tr_rl
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.grid import halfRL
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.safe import safe_div, safe_sqrt

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_calc_visc.F:1
_AD_OPT = "AUTODIFF_OPTIONS.h"      # mom_calc_visc.F:2-4 (#ifdef ALLOW_AUTODIFF)
PI = 3.14159265358979323844         # PARAMS.h:15-16  PARAMETER ( PI = 3.14159265358979323844D0 )
_SQRT_NEG = float(np.copysign(np.nan, -1.))   # SQRT of a negative REAL*8: the x86 default NaN (0xFFF8000000000000)


def _sqrt_ad(x):
    """`IF (x .GT. 0. _d 0) x = SQRT(x)` (#ifdef ALLOW_AUTODIFF arms, :430-436, :489, :582-588, :639): SQRT on the
    x > 0 lanes, x kept elsewhere; the SQRT is guarded (safe_sqrt) so the kept lanes have a finite derivative."""
    pos = x > 0.
    return jnp.where(pos, safe_sqrt(x, pos), x)


def _sqrt_plain(x):
    """`x = SQRT(x)` (#else arms, :438-444, :491, :590-596, :641) with the derivative guarded: SQRT on the x > 0 lanes
    (safe_sqrt: the infinite derivative of SQRT at 0 is never formed, where a plain jnp.sqrt under the caller's
    selections would give 0*inf = NaN); SQRT(x) = x for x = +0, -0 and NaN, so x is kept there; x < 0 gives the NaN
    gfortran's sqrtsd returns (the x86 default NaN, sign bit set), a constant. Every lane has SQRT's value."""
    pos = x > 0.
    return jnp.where(x < 0., _SQRT_NEG, jnp.where(pos, safe_sqrt(x, pos), x))


def _fill(fill4dir, trFld, cfg, params, w2):
    """CALL FILL_CS_CORNER_TR_RL( fill4dir, .FALSE., trFld, bi,bj, myThid ) on every tile (mitjax/eesupp, lane B):
    the corner flags of each tile from the W2 tile view, as fill_cs_corner_tr_rl.F:75-82 derives them."""
    if not cfg.cpp.flag("ALLOW_EXCH2", _OPT):
        raise NotImplementedError("MOM_CALC_VISC: FILL_CS_CORNER_TR_RL on the cube without ALLOW_EXCH2 is not ported")
    isW, isE = jnp.asarray(w2.exch2_isWedge) == 1, jnp.asarray(w2.exch2_isEedge) == 1
    isS, isN = jnp.asarray(w2.exch2_isSedge) == 1, jnp.asarray(w2.exch2_isNedge) == 1
    corners = jnp.stack([isW & isS, isE & isS, isW & isN, isE & isN], axis=1)       # SW, SE, NW, NE
    sz = cfg.size
    new = fill_cs_corner_tr_rl(fill4dir, False, trFld.data, corners, params.useCubedSphereExchange,
                               sNx=sz.sNx, sNy=sz.sNy, OLx=sz.OLx, OLy=sz.OLy)
    return FArray(new, trFld.name, tiled=trFld.tiled, _dims=trFld.dims)


def mom_calc_visc(k, viscAh_Z, viscAh_D, viscA4_Z, viscA4_D, hDiv, vort3, tension, strain, stretching, KE, hFacZ,
                  *, cfg, grid, params, visc, w2=None):
    """MOM_CALC_VISC(bi,bj,k, viscAh_Z,viscAh_D,viscA4_Z,viscA4_D, hDiv,vort3,tension,strain,stretching,KE,hFacZ,
                     myThid)   @63cdc0b pkg/mom_common/mom_calc_visc.F:11-778

    C     Calculate horizontal viscosities (L is typical grid width)
    C     harmonic viscosity=
    C       viscAh (or viscAhD on div pts and viscAhZ on zeta pts)
    C       +0.25*L**2*viscAhGrid/deltaT
    C       +sqrt((viscC2leith/pi)**6*grad(Vort3)**2
    C             +(viscC2leithD/pi)**6*grad(hDiv)**2)*L**3
    C       +(viscC2smag/pi)**2*L**2*sqrt(Tension**2+Strain**2)
    C     biharmonic viscosity=
    C       viscA4 (or viscA4D on div pts and viscA4Z on zeta pts)
    C       +0.25*0.125*L**4*viscA4Grid/deltaT (approx)
    C       +0.125*L**5*sqrt((viscC4leith/pi)**6*grad(Vort3)**2
    C                        +(viscC4leithD/pi)**6*grad(hDiv)**2)
    C       +0.125*L**4*(viscC4smag/pi)**2*sqrt(Tension**2+Strain**2)
    C     LIMITERS -- limit min and max values of viscosities
    C     viscAhReMax is min value for grid point harmonic Reynolds num
    C      harmonic viscosity>sqrt(2*KE)*L/viscAhReMax
    C     viscA4ReMax is min value for grid point biharmonic Reynolds num
    C      biharmonic viscosity>sqrt(2*KE)*L**3/8/viscA4ReMax
    C     viscAhgridmax is CFL stability limiter for harmonic viscosity
    C      harmonic viscosity<0.25*viscAhgridmax*L**2/deltaT
    C     viscA4gridmax is CFL stability limiter for biharmonic viscosity
    C      biharmonic viscosity<viscA4gridmax*L**4/32/deltaT (approx)
    C     viscAhgridmin and viscA4gridmin are lower limits for viscosity:
    C       harmonic viscosity>0.25*viscAhgridmin*L**2/deltaT
    C       biharmonic viscosity>viscA4gridmin*L**4/32/deltaT (approx)

    Returns (viscAh_Z, viscAh_D, viscA4_Z, viscA4_D, hDiv): the four outputs on the points of the main loop
    (DO j=2-OLy,sNy+OLy-1 / DO i=2-OLx,sNx+OLx-1, :371-680; other points keep their prior values) and hDiv, which
    FILL_CS_CORNER_TR_RL overwrites on the cube (:271, :284; returned as the KERNEL_GUIDE says for arguments the
    Fortran overwrites by reference).

    Configuration: `visc` holds the MOM_VISC.h arrays (L2_D, L2_Z, L3_D, L3_Z, L4rdt_D, L4rdt_Z); `params` the PARAMS.h REALs
    (traced: viscAhD/Z, viscA4D/Z, viscAhGrid, viscA4Grid, viscAhMax, viscA4Max, viscAhGridMax/Min,
    viscA4GridMax/Min, viscAhReMax, viscA4ReMax, viscC2leith(D), viscC2LeithQG, viscC4leith(D), viscC2smag,
    viscC4smag, deltaTMom) and the static LOGICALs useHarmonicVisc, useBiharmonicVisc (MOM_VISC.h), useFullLeith,
    nonHydrostatic, useDiagnostics, useCubedSphereExchange. The REAL tests that select code (calcLeith :184-189,
    calcLeithQG :183, calcSmag :191-193) are decided on the host as static named flags (`viscC2leith_ne_0`,
    `viscC2leithD_ne_0`, `viscC2LeithQG_ne_0`, `viscC4leith_ne_0`, `viscC4leithD_ne_0`, `viscC2smag_ne_0`,
    `viscC4smag_ne_0`, KERNEL_GUIDE "REAL-IF rule"); a perturbation of those coefficients must not cross 0. The REAL
    tests that choose between two values stay traced (`jnp.where`, both arms finite): deltaTMom.NE.0 (:167),
    viscAhReMax.NE.0 (:171), viscA4ReMax.NE.0 (:177), viscAhRe_max.GT.0 / viscA4Re_max.GT.0 with KE.GT.0 (:393-402,
    :543-556). The divisions and SQRTs of those arms are guarded before the operation (mitjax/ops/safe.py).

    Literals: `1. _d 0`, `2. _d 0`, `0.125 _d 0`, `0.015625 _d 0`, `0.25 _d 0` are double; `0.` REAL*4 (exact).
    `(x/pi)**6`, `**3`, `**2` with integer literals are lax.integer_pow. MAX/MIN winners per site from the oracle's
    tables (`p=`).

    Ported: the useFullLeith (:406-453) and non-full (:454-472) Leith arms, Smagorinsky (:483-499, :633-646), the
    Reynolds-number limits (:391-403, :541-557), #ifdef ALLOW_AUTODIFF SQRT guards (code_ad builds) and the plain
    SQRTs (code builds); M3 lane MLAdjust: the ALLOW_LEITH_QG arms (viscAh_D/ZLthQG, the stretching gradients of
    calcLeithQG :318-365, halfRL = 0.5 _d 0 of EEPARAMS.h:73). Not ported, raise: ALLOW_3D_VISCAH, ALLOW_3D_VISCA4,
    ALLOW_OBCS,
    AUTODIFF_DISABLE_LEITH, AUTODIFF_DISABLE_REYNOLDS_SCALE (all undefined in the M2 builds), nonHydrostatic under
    ALLOW_NONHYDROSTATIC (:682-719; viscAh_W/viscA4_W of MOM_VISC.h) and DIAGNOSTICS_FILL with useDiagnostics. On the
    cube the two FILL_CS_CORNER_TR_RL calls (:269-273, :282-286) are lane B's eesupp port, the corner flags from the
    W2 tile view `w2` (as in mom_calc_relvort3; the one addition to the Fortran signature). `stretching` is read
    only under ALLOW_LEITH_QG with calcLeithQG; `hFacZ` is not read by the ported code (only by the ALLOW_3D_VISC*
    and OBCS arms). The kkey/ijkkey computations (:154-157, :375-382) only feed
    TAF directives. The point loops run on the whole (i,j) range: each point reads only inputs and the arrays
    divDx/divDy/vrtDx/vrtDy completed before the loop.
    """
    for name in ("ALLOW_3D_VISCAH", "ALLOW_3D_VISCA4", "ALLOW_OBCS"):
        if cfg.cpp.flag(name, _OPT):
            raise NotImplementedError(f"MOM_CALC_VISC: {name} is not ported")
    AD = cfg.cpp.flag("ALLOW_AUTODIFF", _OPT)
    if AD:
        for name in ("AUTODIFF_DISABLE_LEITH", "AUTODIFF_DISABLE_REYNOLDS_SCALE"):
            if cfg.cpp.flag(name, _AD_OPT):
                raise NotImplementedError(f"MOM_CALC_VISC: {name} is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    recip_dxC, recip_dyC, recip_dxG, recip_dyG = grid.recip_dxC, grid.recip_dyC, grid.recip_dxG, grid.recip_dyG
    maskW, maskS, recip_deepFacC = grid.maskW, grid.maskS, grid.recip_deepFacC
    deepFacC, deepFac2C = grid.deepFacC, grid.deepFac2C
    L2_D, L2_Z, L3_D, L3_Z, L4rdt_D, L4rdt_Z = (visc.L2_D, visc.L2_Z, visc.L3_D, visc.L3_Z, visc.L4rdt_D,
                                                 visc.L4rdt_Z)
    p = params
    sqrt_ = _sqrt_ad if AD else _sqrt_plain
    QG = cfg.cpp.flag("ALLOW_LEITH_QG", _OPT)                      # M3 lane MLAdjust: the #ifdef ALLOW_LEITH_QG arms

    recip_dt = safe_div(1., p.deltaTMom, p.deltaTMom != 0., fill=1.)   # :166-167
    deepFac3 = deepFac2C[k]*deepFacC[k]                             # :168
    deepFac4 = deepFac2C[k]*deepFac2C[k]                            # :169

    if p.useHarmonicVisc:                                           # :171-175
        viscAhRe_max = safe_div(jnp.sqrt(2.), p.viscAhReMax, p.viscAhReMax != 0., fill=0.)
    else:
        viscAhRe_max = 0.

    if p.useBiharmonicVisc:                                         # :177-181
        viscA4Re_max = safe_div(0.125*jnp.sqrt(2.), p.viscA4ReMax, p.viscA4ReMax != 0., fill=0.)
    else:
        viscA4Re_max = 0.

    calcLeithQG = p.viscC2LeithQG_ne_0                              # :183
    calcLeith = (p.viscC2leith_ne_0                                 # :184-189
                 or p.viscC2leithD_ne_0
                 or p.viscC4leith_ne_0
                 or p.viscC4leithD_ne_0
                 or calcLeithQG)

    calcSmag = (p.viscC2smag_ne_0                                   # :191-193
                or p.viscC4smag_ne_0)

    if calcSmag:                                                    # :195-201
        smag2fac = (p.viscC2smag/PI)**2
        smag4fac = 0.125*(p.viscC4smag/PI)**2
    else:
        smag2fac = 0.
        smag4fac = 0.

    if calcLeith:                                                   # :203-226
        if p.useFullLeith:
#         Uses correct calculation for gradients
            leith2fac = (p.viscC2leith/PI)**6
            leithD2fac = (p.viscC2leithD/PI)**6
            leithQG2fac = (p.viscC2LeithQG/PI)**6                   # :208
            leith4fac = 0.015625*(p.viscC4leith/PI)**6
            leithD4fac = 0.015625*(p.viscC4leithD/PI)**6
        else:
#         Uses approximate gradients
            leith2fac = (p.viscC2leith/PI)**3
            leithD2fac = (p.viscC2leithD/PI)**3
            leithQG2fac = (p.viscC2LeithQG/PI)**3                   # :216
            leith4fac = 0.125*(p.viscC4leith/PI)**3
            leithD4fac = 0.125*(p.viscC4leithD/PI)**3
    else:
        leith2fac = 0.
        leith4fac = 0.
        leithQG2fac = 0.                                            # :220
        leithD2fac = 0.
        leithD4fac = 0.

    jA = loop_j(1-OLy, sNy+OLy)                                     # :228-254
    iA = loop_i(1-OLx, sNx+OLx)
    viscAh_DLth = viscAh_D.local("viscAh_DLth").at[iA, jA].set(0.)
    viscAh_ZLth = viscAh_D.local("viscAh_ZLth").at[iA, jA].set(0.)
    viscA4_DLth = viscAh_D.local("viscA4_DLth").at[iA, jA].set(0.)
    viscA4_ZLth = viscAh_D.local("viscA4_ZLth").at[iA, jA].set(0.)
    viscAh_DLthD = viscAh_D.local("viscAh_DLthD").at[iA, jA].set(0.)
    viscAh_ZLthD = viscAh_D.local("viscAh_ZLthD").at[iA, jA].set(0.)
    viscA4_DLthD = viscAh_D.local("viscA4_DLthD").at[iA, jA].set(0.)
    viscA4_ZLthD = viscAh_D.local("viscA4_ZLthD").at[iA, jA].set(0.)
    viscAh_DSmg = viscAh_D.local("viscAh_DSmg").at[iA, jA].set(0.)
    viscAh_ZSmg = viscAh_D.local("viscAh_ZSmg").at[iA, jA].set(0.)
    viscA4_DSmg = viscAh_D.local("viscA4_DSmg").at[iA, jA].set(0.)
    viscA4_ZSmg = viscAh_D.local("viscA4_ZSmg").at[iA, jA].set(0.)
    if QG:                                                          # :244-247
        viscAh_DLthQG = viscAh_D.local("viscAh_DLthQG").at[iA, jA].set(0.)
        viscAh_ZLthQG = viscAh_D.local("viscAh_ZLthQG").at[iA, jA].set(0.)

#     - Initialise gradient arrays (:257-264)
    divDx = viscAh_D.local("divDx").at[iA, jA].set(0.)
    divDy = viscAh_D.local("divDy").at[iA, jA].set(0.)
    vrtDx = viscAh_D.local("vrtDx").at[iA, jA].set(0.)
    vrtDy = viscAh_D.local("vrtDy").at[iA, jA].set(0.)

    if calcLeith:                                                   # :266-369
#     horizontal gradient of horizontal divergence:
#-     in X direction:
        if p.useCubedSphereExchange:                                # :269-273
            hDiv = _fill(1, hDiv, cfg, p, w2)
        j = loop_j(2-OLy, sNy+OLy-1)                                # :274-279
        i = loop_i(2-OLx, sNx+OLx-1)
        divDx = divDx.at[i, j].set((hDiv[i, j]-hDiv[i-1, j])
                                   *recip_dxC[i, j]*recip_deepFacC[k])

#-     in Y direction:
        if p.useCubedSphereExchange:                                # :282-286
            hDiv = _fill(2, hDiv, cfg, p, w2)
        divDy = divDy.at[i, j].set((hDiv[i, j]-hDiv[i, j-1])        # :287-292
                                   *recip_dyC[i, j]*recip_deepFacC[k])

#     horizontal gradient of vertical vorticity:
#-     in X direction:
        j = loop_j(2-OLy, sNy+OLy)                                  # :296-305
        i = loop_i(2-OLx, sNx+OLx-1)
        vrtDx = vrtDx.at[i, j].set((vort3[i+1, j]-vort3[i, j])
                                   *recip_dxG[i, j]*recip_deepFacC[k]
                                   *maskS[i, j, k])
#-     in Y direction:
        j = loop_j(2-OLy, sNy+OLy-1)                                # :307-316
        i = loop_i(2-OLx, sNx+OLx)
        vrtDy = vrtDy.at[i, j].set((vort3[i, j+1]-vort3[i, j])
                                   *recip_dyG[i, j]*recip_deepFacC[k]
                                   *maskW[i, j, k])

        if QG and calcLeithQG:                                      # :318-365 (M3 lane MLAdjust)
#     horizontal gradient of vorticity and vortex stretching: d/dx of stretching averaged onto V-points
            j = loop_j(2-OLy, sNy+OLy)                              # :324-325
            i = loop_i(2-OLx, sNx+OLx-1)
            vrtDx = vrtDx.at[i, j].set(vrtDx[i, j]
                                       + halfRL*halfRL
                                       * ((stretching[i+1, j]-stretching[i, j])
                                          * recip_dxC[i+1, j]*recip_deepFacC[k]
                                          + (stretching[i, j]-stretching[i-1, j])
                                          * recip_dxC[i, j]*recip_deepFacC[k]
                                          + (stretching[i+1, j-1]-stretching[i, j-1])
                                          * recip_dxC[i+1, j-1]*recip_deepFacC[k]
                                          + (stretching[i, j-1]-stretching[i-1, j-1])
                                          * recip_dxC[i, j-1]*recip_deepFacC[k]
                                          )*maskS[i, j, k])         # :328-338
#     d/dy of stretching averaged onto U-points
            j = loop_j(2-OLy, sNy+OLy-1)                            # :344-345
            i = loop_i(2-OLx, sNx+OLx)
            vrtDy = vrtDy.at[i, j].set(vrtDy[i, j]
                                       + halfRL*halfRL
                                       * ((stretching[i, j+1]-stretching[i, j])
                                          * recip_dyC[i, j+1]*recip_deepFacC[k]
                                          + (stretching[i, j]-stretching[i, j-1])
                                          * recip_dyC[i, j]*recip_deepFacC[k]
                                          + (stretching[i-1, j+1]-stretching[i-1, j])
                                          * recip_dyC[i-1, j+1]*recip_deepFacC[k]
                                          + (stretching[i-1, j]-stretching[i-1, j-1])
                                          * recip_dyC[i-1, j]*recip_deepFacC[k]
                                          )*maskW[i, j, k])         # :348-358

    j = loop_j(2-OLy, sNy+OLy-1)                                    # :371-680
    i = loop_i(2-OLx, sNx+OLx-1)
#CCCCCCCCCCCCCCC Divergence Point CalculationsCCCCCCCCCCCCCCCCCCCC
#     Harmonic on Div.u points
    L2 = L2_D[i, j]*deepFac2C[k]                                    # :385-389
    L2rdt = 0.25*recip_dt*L2
    L3 = L3_D[i, j]*deepFac3
    L4rdt = L4rdt_D[i, j]*deepFac4
    L5 = (L2*L3)

#     These are (powers of) length scales
    KEij = KE[i, j]                                                 # :393-402
    cU = (viscAhRe_max > 0.) & (KEij > 0.)
    argU = KEij*L2
    Uscl = jnp.where(cU, safe_sqrt(argU, cU & (argU > 0.))*viscAhRe_max, 0.)
    c4 = (viscA4Re_max > 0.) & (KEij > 0.)
    U4scl = jnp.where(c4, safe_sqrt(KEij, c4)*L3*viscA4Re_max, 0.)

    if p.useFullLeith and calcLeith:                                # :406-453
#     This is the vector magnitude of the vorticity gradient squared
        grdVrt = 0.25*((vrtDx[i, j+1]*vrtDx[i, j+1]
                        + vrtDx[i, j]*vrtDx[i, j])
                       + (vrtDy[i+1, j]*vrtDy[i+1, j]
                          + vrtDy[i, j]*vrtDy[i, j]))
#     This is the vector magnitude of grad (div.v) squared
#     Using it in Leith serves to damp instabilities in w.
        grdDiv = 0.25*((divDx[i+1, j]*divDx[i+1, j]
                        + divDx[i, j]*divDx[i, j])
                       + (divDy[i, j+1]*divDy[i, j+1]
                          + divDy[i, j]*divDy[i, j]))

        sqargAh = leith2fac*grdVrt+leithD2fac*grdDiv               # :420-423
        sqargA4 = leith4fac*grdVrt+leithD4fac*grdDiv
        sqargAhD = leithD2fac*grdDiv
        sqargA4D = leithD4fac*grdDiv
        if QG:
            sqargQG = leithQG2fac*(grdVrt+grdDiv)                   # :424-426

        sqargAh = sqrt_(sqargAh)                                    # :428-445
        sqargA4 = sqrt_(sqargA4)
        sqargAhD = sqrt_(sqargAhD)
        sqargA4D = sqrt_(sqargA4D)
        if QG:
            sqargQG = sqrt_(sqargQG)                                # :434-436 / :442-444
        viscAh_DLth = viscAh_DLth.at[i, j].set(sqargAh * L3)        # :446-449
        viscA4_DLth = viscA4_DLth.at[i, j].set(sqargA4 * L5)
        viscAh_DLthD = viscAh_DLthD.at[i, j].set(sqargAhD * L3)
        viscA4_DLthD = viscA4_DLthD.at[i, j].set(sqargA4D * L5)
        if QG:
            viscAh_DLthQG = viscAh_DLthQG.at[i, j].set(sqargQG * L3)   # :450-452

    elif calcLeith:                                                 # :454-472
#     but this approximation will work on cube (and differs by as much as 4X)
        grdVrt = MAX(jnp.abs(vrtDx[i, j+1]), jnp.abs(vrtDx[i, j]), p="a")   # :456
        grdVrt = MAX(grdVrt, jnp.abs(vrtDy[i+1, j]), p="a")                 # :457
        grdVrt = MAX(grdVrt, jnp.abs(vrtDy[i, j]), p="a")                   # :458

        grdDiv = MAX(jnp.abs(divDx[i+1, j]), jnp.abs(divDx[i, j]), p="a")   # :461
        grdDiv = MAX(grdDiv, jnp.abs(divDy[i, j+1]), p="a")                 # :462
        grdDiv = MAX(grdDiv, jnp.abs(divDy[i, j]), p="a")                   # :463

        viscAh_DLth = viscAh_DLth.at[i, j].set((leith2fac*grdVrt+(leithD2fac*grdDiv))*L3)   # :465-468
        viscA4_DLth = viscA4_DLth.at[i, j].set((leith4fac*grdVrt+(leithD4fac*grdDiv))*L5)
        viscAh_DLthD = viscAh_DLthD.at[i, j].set(((leithD2fac*grdDiv))*L3)
        viscA4_DLthD = viscA4_DLthD.at[i, j].set(((leithD4fac*grdDiv))*L5)
        if QG:
            viscAh_DLthQG = viscAh_DLthQG.at[i, j].set(leithQG2fac*(grdVrt + grdDiv)*L3)   # :469-471

    else:                                                           # :473-481
        viscAh_DLth = viscAh_DLth.at[i, j].set(0.)
        viscA4_DLth = viscA4_DLth.at[i, j].set(0.)
        viscAh_DLthD = viscAh_DLthD.at[i, j].set(0.)
        viscA4_DLthD = viscA4_DLthD.at[i, j].set(0.)
        if QG:
            viscAh_DLthQG = viscAh_DLthQG.at[i, j].set(0.)          # :478-480

    if calcSmag:                                                    # :483-499
        sqargSmag = (tension[i, j]**2
                     + 0.25*(strain[i+1, j]**2+strain[i, j+1]**2
                             + strain[i, j]**2+strain[i+1, j+1]**2))
        sqargSmag = sqrt_(sqargSmag)                                # :487-492
        viscAh_DSmg = viscAh_DSmg.at[i, j].set(L2*sqargSmag)        # :493-495
        viscA4_DSmg = viscA4_DSmg.at[i, j].set(smag4fac*L2*viscAh_DSmg[i, j])
        viscAh_DSmg = viscAh_DSmg.at[i, j].set(smag2fac*viscAh_DSmg[i, j])
    else:
        viscAh_DSmg = viscAh_DSmg.at[i, j].set(0.)                  # :496-498
        viscA4_DSmg = viscA4_DSmg.at[i, j].set(0.)

#     Harmonic on Div.u points
    Alin = (p.viscAhD+p.viscAhGrid*L2rdt                            # :503-507
            + viscAh_DLth[i, j]+viscAh_DSmg[i, j])
    if QG:
        Alin = (p.viscAhD+p.viscAhGrid*L2rdt                        # :503-507 with :505-507
                + viscAh_DLth[i, j]+viscAh_DSmg[i, j]
                + viscAh_DLthQG[i, j])
    viscAh_DMin = MAX(p.viscAhGridMin*L2rdt, Uscl, p="b")           # :514
    viscAh_D = viscAh_D.at[i, j].set(MAX(viscAh_DMin, Alin, p="b"))   # :515
    viscAh_DMax = MIN(p.viscAhGridMax*L2rdt, p.viscAhMax, p="a")    # :516
    viscAh_D = viscAh_D.at[i, j].set(MIN(viscAh_DMax, viscAh_D[i, j], p="b"))   # :517

#     BiHarmonic on Div.u points
    Alin = (p.viscA4D+p.viscA4Grid*L4rdt                            # :520-521
            + viscA4_DLth[i, j]+viscA4_DSmg[i, j])
    viscA4_DMin = MAX(p.viscA4GridMin*L4rdt, U4scl, p="b")          # :528
    viscA4_D = viscA4_D.at[i, j].set(MAX(viscA4_DMin, Alin, p="b"))   # :529
    viscA4_DMax = MIN(p.viscA4GridMax*L4rdt, p.viscA4Max, p="a")    # :530
    viscA4_D = viscA4_D.at[i, j].set(MIN(viscA4_DMax, viscA4_D[i, j], p="b"))   # :531

#CCCCCCCCCCCCC Vorticity Point CalculationsCCCCCCCCCCCCCCCCCC
#     These are (powers of) length scales
    L2 = L2_Z[i, j]*deepFac2C[k]                                    # :535-539
    L2rdt = 0.25*recip_dt*L2
    L3 = L3_Z[i, j]*deepFac3
    L4rdt = L4rdt_Z[i, j]*deepFac4
    L5 = (L2*L3)

#     Velocity Reynolds Scale (Pb here at CS-grid corners !)
    cRe = (viscAhRe_max > 0.) | (viscA4Re_max > 0.)                 # :543-556
    keZpt = 0.25*((KE[i, j]+KE[i-1, j-1])
                  + (KE[i-1, j]+KE[i, j-1]))
    cK = cRe & (keZpt > 0.)
    argZ = keZpt*L2
    Uscl = jnp.where(cK, safe_sqrt(argZ, cK & (argZ > 0.))*viscAhRe_max, 0.)
    U4scl = jnp.where(cK, safe_sqrt(keZpt, cK)*L3*viscA4Re_max, 0.)

    if p.useFullLeith and calcLeith:                                # :561-605
#     This is the vector magnitude of the vorticity gradient squared
        grdVrt = 0.25*((vrtDx[i-1, j]*vrtDx[i-1, j]
                        + vrtDx[i, j]*vrtDx[i, j])
                       + (vrtDy[i, j-1]*vrtDy[i, j-1]
                          + vrtDy[i, j]*vrtDy[i, j]))
#     This is the vector magnitude of grad(div.v) squared
        grdDiv = 0.25*((divDx[i, j-1]*divDx[i, j-1]
                        + divDx[i, j]*divDx[i, j])
                       + (divDy[i-1, j]*divDy[i-1, j]
                          + divDy[i, j]*divDy[i, j]))

        sqargAh = leith2fac*grdVrt+leithD2fac*grdDiv               # :573-576
        sqargA4 = leith4fac*grdVrt+leithD4fac*grdDiv
        sqargAhD = leithD2fac*grdDiv
        sqargA4D = leithD4fac*grdDiv
        if QG:
            sqargQG = leithQG2fac*(grdVrt+grdDiv)                   # :577-579
        sqargAh = sqrt_(sqargAh)                                    # :580-597
        sqargA4 = sqrt_(sqargA4)
        sqargAhD = sqrt_(sqargAhD)
        sqargA4D = sqrt_(sqargA4D)
        if QG:
            sqargQG = sqrt_(sqargQG)                                # :586-588 / :594-596
        viscAh_ZLth = viscAh_ZLth.at[i, j].set(sqargAh * L3)        # :598-601
        viscA4_ZLth = viscA4_ZLth.at[i, j].set(sqargA4 * L5)
        viscAh_ZLthD = viscAh_ZLthD.at[i, j].set(sqargAhD * L3)
        viscA4_ZLthD = viscA4_ZLthD.at[i, j].set(sqargA4D * L5)
        if QG:
            viscAh_ZLthQG = viscAh_ZLthQG.at[i, j].set(sqargQG * L3)   # :602-604

    elif calcLeith:                                                 # :606-622
#     but this approximation will work on cube (and differs by as much as 4X)
        grdVrt = MAX(jnp.abs(vrtDx[i-1, j]), jnp.abs(vrtDx[i, j]), p="a")   # :608
        grdVrt = MAX(grdVrt, jnp.abs(vrtDy[i, j-1]), p="a")                 # :609
        grdVrt = MAX(grdVrt, jnp.abs(vrtDy[i, j]), p="a")                   # :610

        grdDiv = MAX(jnp.abs(divDx[i, j]), jnp.abs(divDx[i, j-1]), p="a")   # :612
        grdDiv = MAX(grdDiv, jnp.abs(divDy[i, j]), p="a")                   # :613
        grdDiv = MAX(grdDiv, jnp.abs(divDy[i-1, j]), p="a")                 # :614

        viscAh_ZLth = viscAh_ZLth.at[i, j].set((leith2fac*grdVrt+(leithD2fac*grdDiv))*L3)   # :616-619
        viscA4_ZLth = viscA4_ZLth.at[i, j].set((leith4fac*grdVrt+(leithD4fac*grdDiv))*L5)
        viscAh_ZLthD = viscAh_ZLthD.at[i, j].set((leithD2fac*grdDiv)*L3)
        viscA4_ZLthD = viscA4_ZLthD.at[i, j].set((leithD4fac*grdDiv)*L5)
        if QG:
            viscAh_ZLthQG = viscAh_ZLthQG.at[i, j].set(leithQG2fac*(grdVrt + grdDiv)*L3)   # :620-622
    else:                                                           # :623-631
        viscAh_ZLth = viscAh_ZLth.at[i, j].set(0.)
        viscA4_ZLth = viscA4_ZLth.at[i, j].set(0.)
        viscAh_ZLthD = viscAh_ZLthD.at[i, j].set(0.)
        viscA4_ZLthD = viscA4_ZLthD.at[i, j].set(0.)
        if QG:
            viscAh_ZLthQG = viscAh_ZLthQG.at[i, j].set(0.)          # :628-630

    if calcSmag:                                                    # :633-646 (no ELSE: the :249-252 zeros stay)
        sqargSmag = (strain[i, j]**2
                     + 0.25*(tension[i, j]**2+tension[i, j-1]**2
                             + tension[i-1, j]**2+tension[i-1, j-1]**2))
        sqargSmag = sqrt_(sqargSmag)                                # :637-642
        viscAh_ZSmg = viscAh_ZSmg.at[i, j].set(L2*sqargSmag)        # :643-645
        viscA4_ZSmg = viscA4_ZSmg.at[i, j].set(smag4fac*L2*viscAh_ZSmg[i, j])
        viscAh_ZSmg = viscAh_ZSmg.at[i, j].set(smag2fac*viscAh_ZSmg[i, j])

#     Harmonic on Zeta points
    Alin = (p.viscAhZ+p.viscAhGrid*L2rdt                            # :650-654
            + viscAh_ZLth[i, j]+viscAh_ZSmg[i, j])
    if QG:
        Alin = (p.viscAhZ+p.viscAhGrid*L2rdt                        # :650-654 with :652-654
                + viscAh_ZLth[i, j]+viscAh_ZSmg[i, j]
                + viscAh_ZLthQG[i, j])
    viscAh_ZMin = MAX(p.viscAhGridMin*L2rdt, Uscl, p="b")           # :661
    viscAh_Z = viscAh_Z.at[i, j].set(MAX(viscAh_ZMin, Alin, p="b"))   # :662
    viscAh_ZMax = MIN(p.viscAhGridMax*L2rdt, p.viscAhMax, p="a")    # :663
    viscAh_Z = viscAh_Z.at[i, j].set(MIN(viscAh_ZMax, viscAh_Z[i, j], p="b"))   # :664

#     BiHarmonic on Zeta points
    Alin = (p.viscA4Z+p.viscA4Grid*L4rdt                            # :667-668
            + viscA4_ZLth[i, j]+viscA4_ZSmg[i, j])
    viscA4_ZMin = MAX(p.viscA4GridMin*L4rdt, U4scl, p="b")          # :675
    viscA4_Z = viscA4_Z.at[i, j].set(MAX(viscA4_ZMin, Alin, p="b"))   # :676
    viscA4_ZMax = MIN(p.viscA4GridMax*L4rdt, p.viscA4Max, p="a")    # :677
    viscA4_Z = viscA4_Z.at[i, j].set(MIN(viscA4_ZMax, viscA4_Z[i, j], p="b"))   # :678

    if cfg.cpp.flag("ALLOW_NONHYDROSTATIC", _OPT) and p.nonHydrostatic:   # :682-719
        raise NotImplementedError("MOM_CALC_VISC: nonHydrostatic (viscAh_W, viscA4_W of MOM_VISC.h) is not ported")
    if cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and p.useDiagnostics:      # :734-775
        raise NotImplementedError("MOM_CALC_VISC: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")
    return viscAh_Z, viscAh_D, viscA4_Z, viscA4_D, hDiv
