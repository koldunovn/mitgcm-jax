"""GGL90_CALC: pkg/ggl90/ggl90_calc.F @63cdc0b."""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.model.grid import PI
from mitjax.model.src.solve_tridiagonal import solve_tridiagonal
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.libm import glibc_exp
from mitjax.ops.safe import safe_div, safe_sqrt
from mitjax.pkg.ggl90.ggl90_h import GGL90eps, SQRTTWO
from mitjax.pkg.ggl90.ggl90_idemix import ggl90_idemix
from mitjax.pkg.ggl90.ggl90_mixinglength import ggl90_mixinglength


def kvals(k):
    """The Fortran level numbers of a k-vectorised nest `k` (loops_kji) as an int32 array [1, nk, 1, 1]."""
    return jnp.arange(k.first, k.last+1, dtype=jnp.int32)[None, :, None, None]


def at_level(A, kidx):
    """A(i,j,kidx(i,j)) of an FArray A declared (i, j, k) (k = 1..Nr), kidx a traced int32 array [tile, j, i] over
    A's whole i, j range: the gather of a per-point level index (e.g. kBot = MAX(kLowC,1))."""
    (_, klo, _), = [d for d in A.dims if d[0] == "k"]
    return jnp.take_along_axis(A.data, (kidx - klo)[:, None], axis=1)[:, 0]


def ggl90_calc(sigmaR, myTime, myIter, *, cfg, grid, params, ggl, state, gm=None):
    """GGL90_CALC( bi, bj, sigmaR, myTime, myIter, myThid )   @63cdc0b pkg/ggl90/ggl90_calc.F:13-1179

    C     | SUBROUTINE GGL90_CALC                                    |
    C     | o Compute all GGL90 fields defined in GGL90.h            |
    C     | Equation numbers refer to                                |
    C     |  Gaspar et al. (1990), JGR 95 (C9), pp 16,179            |
    C     | Some parts of the implementation follow Blanke and       |
    C     |  Delecuse (1993), JPO, and OPA code, in particular the   |
    C     |  computation of the                                      |
    C     |  mixing length = max(min(lk,depth),lkmin)                |
    C     | Note: Only call this S/R if Nr > 1 (no use if Nr=1)      |
    C global parameters updated by ggl90_calc
    C     GGL90TKE     :: sub-grid turbulent kinetic energy          (m^2/s^2)
    C     GGL90viscAz  :: GGL90 eddy viscosity coefficient             (m^2/s)
    C     GGL90diffKzT :: GGL90 diffusion coefficient for temperature  (m^2/s)
    C     sigmaR :: Vertical gradient of iso-neutral density

    sigmaR: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr); `ggl` GGL90.h (Ggl90), `state` DYNVARS.h/FFIELDS.h (uVel, vVel,
    surfaceForcingU, surfaceForcingV), `gm` GMREDI.h (Kwz; read by GGL90_IDEMIX only with useGMRedi). Returns `ggl`
    with GGL90TKE, GGL90viscArU, GGL90viscArV, GGL90diffKr (and IDEMIX_E from GGL90_IDEMIX) as the routine leaves
    them; every point it does not write keeps its value.

    Ported: z coordinates (kSrf = 1, kTop = 2, coordFac = 1: :200-211; every usingPCoords arm raises), the default
    and calcMeanVertShear shears (:474-504), the plain Prandtl number (:579-587) and the IDEMIX one
    (ALLOW_GGL90_IDEMIX with useIDEMIX, :564-576, GGL90_IDEMIX :264-267, :616-626), the Langmuir branch
    (ALLOW_GGL90_LANGMUIR with useLANGMUIR, :318-335, :443-458, :510-561, :628-638), GGL90_MISSING_HFAC_BUG
    (:677-685), both bottom boundary conditions (GGL90_dirichlet, :724-741, :918-939), the non-AD and the
    ALLOW_AUTODIFF forms of the SQRT guards (:298-307, :341-350, :868-873). Not ported (raise): pressure coordinates,
    ALLOW_GGL90_HORIZDIFF with GGL90diffTKEh > 0 (:387-440, :640-650), ALLOW_GGL90_SMOOTH (:995-1003 and the
    smoothed viscosities), ALLOW_SHELFICE with useShelfIce (:798-862, :903-908), SOLVE_DIAGONAL_LOWMEMORY (the
    matrix is not initialised there, :279-283). The ALLOW_DIAGNOSTICS blocks (:221-239, :955-989, :1076-1174) only
    fill diagnostics (pkg/diagnostics is not ported) and change no output; the TAF directives are not code.

    Loops: the "proper" k loop (:382-653) has independent iterations (level k reads levels k and k-1 of inputs and
    writes level k only; the 2-D temporaries KappaM, verticalShear, stokesterm, dstokesUdR/VdR are rewritten at every
    level before they are read and not read after the loop), so it is vectorised with 3-D temporaries; so are the
    matrix, boundary-condition and viscosity loops. The per-point levels kp1 = MAX(1,MIN(kLowC,k+1)) (:711) and
    kBot = MAX(kLowC,1) (:736, :932) are gathers / masked writes along k. The tridiagonal solve is SOLVE_TRIDIAGONAL
    (scan_k).

    Differentiability: SQRT(GGL90TKE) (:302, :345), SQRT(uStarSquare) (:872) and SQRT(ABS(surfaceForcingU/V))
    (:328-331) are guarded on an argument > 0 (the value is Fortran's for every argument >= +0: GGL90TKE >= +0 after
    GGL90_INIT_VARIA and every clipping, uStarSquare is a sum of squares); 1/hFac (:252-253) is guarded on the
    Fortran's own `hFac .NE. 0` test. MAX/MIN as written (JAX's derivative at a tie).
    """
    opt = "GGL90_OPTIONS.h"
    sz = cfg.size
    Nr = sz.Nr
    autodiff = cfg.cpp.flag("ALLOW_AUTODIFF", opt)
    idemix_c = cfg.cpp.flag("ALLOW_GGL90_IDEMIX", opt)
    langmuir_c = cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", opt)
    useIDEMIX = idemix_c and ggl.useIDEMIX
    useLANGMUIR = langmuir_c and ggl.useLANGMUIR
    if params.usingPCoords:                                             # :200-202
        raise NotImplementedError("GGL90_CALC: pressure coordinates are not ported")
    if cfg.cpp.flag("ALLOW_GGL90_HORIZDIFF", opt) and ggl.static_float("GGL90diffTKEh") > 0.:
        raise NotImplementedError("GGL90_CALC: ALLOW_GGL90_HORIZDIFF with GGL90diffTKEh > 0 is not ported")
    if cfg.cpp.flag("ALLOW_GGL90_SMOOTH", opt):
        raise NotImplementedError("GGL90_CALC: ALLOW_GGL90_SMOOTH is not ported")
    if cfg.cpp.flag("ALLOW_SHELFICE", opt) and cfg.use_flag("useShelfIce"):
        raise NotImplementedError("GGL90_CALC: ALLOW_SHELFICE with useShelfIce is not ported")
    if cfg.cpp.flag("SOLVE_DIAGONAL_LOWMEMORY", opt):
        raise NotImplementedError("GGL90_CALC: SOLVE_DIAGONAL_LOWMEMORY is not ported")

    iMin, iMax = 2-sz.OLx, sz.sNx+sz.OLx-1                             # :192
    jMin, jMax = 2-sz.OLy, sz.sNy+sz.OLy-1                             # :193
    fj, fi = (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx)
    kSrf, kTop = 1, 2                                                   # :204-205
    deltaTloc = params.dTtracerLev[kSrf]                                # :207
    coordFac = 1.0                                                      # :209  1. _d 0
    recip_coordFac = 1./coordFac                                        # :211
    explDissFac = 0.0                                                   # :218  0. _d 0
    implDissFac = 1.0 - explDissFac                                     # :219  1. _d 0 - explDissFac
    halfRS = halfRL = 0.5                                               # EEPARAMS.h:70, :73
    oneRS = oneRL = 1.0                                                 # EEPARAMS.h:69, :72
    maskC, hFacC, recip_hFacC = grid.maskC, grid.hFacC, grid.recip_hFacC
    TKE = ggl.GGL90TKE
    uVel, vVel = state.uVel, state.vVel

    # :244-259  hFac of the w-cells; km1 = MAX(k-1,1): 1 at k = 1, else k-1
    recip_hFacI = TKE.local("recip_hFacI")
    hFacI = TKE.local("hFacI")
    for klo, khi in ((1, 1), (2, Nr)):
        k, j, i = loops_kji((klo, khi), fj, fi)
        hFacC_km1 = hFacC[i, j, 1] if klo == 1 else hFacC[i, j, k-1]
        hFac = (MIN(halfRS, hFacC_km1, p="a")                           # :248-250
                + MIN(halfRS, hFacC[i, j, k], p="b"))
        nz = hFac != 0.0                                                # :252  IF ( hFac .NE. 0. _d 0 )
        recip_hFacI = recip_hFacI.at[i, j, k].set(safe_div(1.0, hFac, nz, fill=0.0))    # :251-253
        if idemix_c:
            hFacI = hFacI.at[i, j, k].set(hFac)                         # :255

    IDEMIX_gTKE = None
    if useIDEMIX:                                                       # :264-267
        IDEMIX_gTKE, ggl = ggl90_idemix(hFacI, recip_hFacI, sigmaR, TKE.local("gTKE"), myTime, myIter,
                                        cfg=cfg, grid=grid, params=params, ggl=ggl, gm=gm)

    # :271-288  initialize local fields
    k, j, i = loops_kji((1, Nr), fj, fi)
    rMixingLength = TKE.local("rMixingLength").at[i, j, k].set(0.0)
    GGL90visctmp = TKE.local("GGL90visctmp").at[i, j, k].set(0.0)
    KappaE = TKE.local("KappaE").at[i, j, k].set(0.0)
    TKEPrandtlNumber = TKE.local("TKEPrandtlNumber").at[i, j, k].set(1.0)
    GGL90mixingLength = TKE.local("GGL90mixingLength").at[i, j, k].set(ggl.GGL90mixingLengthMin)
    a3d = TKE.local("a3d").at[i, j, k].set(0.0)
    b3d = TKE.local("b3d").at[i, j, k].set(1.0)
    c3d = TKE.local("c3d").at[i, j, k].set(0.0)
    Nsquare = TKE.local("Nsquare").at[i, j, k].set(0.0)
    SQRTTKE = TKE.local("SQRTTKE").at[i, j, k].set(0.0)
    # :292-316  (KappaM, verticalShear are per-level 2-D temporaries in the Fortran: 3-D here, see docstring)
    j, i = loop_j(*fj), loop_i(*fi)
    uStarSquare = grid.Ro_surf.local("uStarSquare").at[i, j].set(0.0)  # :295
    t1 = TKE[i, j, 1]
    if autodiff:                                                        # :298-307 (usingZCoords holds)
        SQRTTKE = SQRTTKE.at[i, j, 1].set(safe_sqrt(t1, (maskC[i, j, 1] == oneRS) & (t1 > 0.0)))
    else:
        SQRTTKE = SQRTTKE.at[i, j, 1].set(safe_sqrt(t1, t1 > 0.0))     # :302

    if useLANGMUIR:                                                     # :318-335
        recip_Lasq = 1.0/ggl.LC_num                                     # :320  1. _d 0 / LC_num
        recip_Lasq = recip_Lasq*recip_Lasq                              # :321
        recip_LD = 4.0*PI/ggl.LC_lambda                                 # :322  4. _d 0 * PI / LC_lambda
        sfU, sfV = state.surfaceForcingU[i, j], state.surfaceForcingV[i, j]
        uStar = uStarSquare.local("uStar").at[i, j].set(                # :328-329
            jnp.copysign(safe_sqrt(jnp.abs(sfU), jnp.abs(sfU) > 0.0), sfU))
        vStar = uStarSquare.local("vStar").at[i, j].set(                # :330-331
            jnp.copysign(safe_sqrt(jnp.abs(sfV), jnp.abs(sfV) > 0.0), sfV))

    # :337-363
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    mskLoc = maskC[i, j, k]*maskC[i, j, k-1]                            # :340
    tk = TKE[i, j, k]
    if autodiff:                                                        # :341-350
        SQRTTKE = SQRTTKE.at[i, j, k].set(safe_sqrt(tk, (mskLoc == oneRS) & (tk > 0.0)))
    else:
        SQRTTKE = SQRTTKE.at[i, j, k].set(safe_sqrt(tk, tk > 0.0))     # :345
    Nsquare = Nsquare.at[i, j, k].set(params.gravity*grid.gravitySign*params.recip_rhoConst      # :353-354
                                      * sigmaR[i, j, k] * coordFac)
    GGL90mixingLength = GGL90mixingLength.at[i, j, k].set(              # :358-360
        SQRTTWO * SQRTTKE[i, j, k]/jnp.sqrt(MAX(Nsquare[i, j, k], GGL90eps, p="b")) * mskLoc)

    LCmixingLength = TKE.local("LCmixingLength") if langmuir_c else None
    GGL90mixingLength, LCmixingLength, rMixingLength = ggl90_mixinglength(   # :365-372
        GGL90mixingLength, LCmixingLength, rMixingLength, iMin, iMax, jMin, jMax, myTime, myIter,
        cfg=cfg, grid=grid, params=params, ggl=ggl)

    # :381-653  the "proper" k loop, vectorised (docstring); km1 = k-1
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    KappaM3 = TKE.local("KappaM")
    if useLANGMUIR:                                                     # :444-449
        KappaM3 = KappaM3.at[i, j, k].set(ggl.GGL90ck*LCmixingLength[i, j, k]*SQRTTKE[i, j, k])
    else:                                                               # :452-454
        KappaM3 = KappaM3.at[i, j, k].set(ggl.GGL90ck*GGL90mixingLength[i, j, k]*SQRTTKE[i, j, k])
    m2 = maskC[i, j, k]*maskC[i, j, k-1]
    GGL90visctmp = GGL90visctmp.at[i, j, k].set(                        # :462-464
        MAX(KappaM3[i, j, k], params.diffKrNrS[k] * recip_coordFac*recip_coordFac, p="b")
        * maskC[i, j, k]*maskC[i, j, k-1])
    KappaM3 = KappaM3.at[i, j, k].set(                                  # :467-469
        MAX(KappaM3[i, j, k], params.viscArNr[k] * recip_coordFac*recip_coordFac, p="b")
        * maskC[i, j, k]*maskC[i, j, k-1])
    KappaM = KappaM3[i, j, k]
    if ggl.calcMeanVertShear:                                           # :474-488
        tempU = (uVel[i, j, k-1] - uVel[i, j, k])
        tempUp = (uVel[i+1, j, k-1] - uVel[i+1, j, k])
        tempV = (vVel[i, j, k-1] - vVel[i, j, k])
        tempVp = (vVel[i, j+1, k-1] - vVel[i, j+1, k])
        verticalShear = (((tempU*tempU + tempUp*tempUp)                 # :482-486
                          + (tempV*tempV + tempVp*tempVp))*halfRL*grid.recip_drC[k]*grid.recip_drC[k]
                         * coordFac*coordFac)
    else:                                                               # :491-503
        tempU = (((uVel[i, j, k-1] + uVel[i+1, j, k-1])
                  - (uVel[i, j, k] + uVel[i+1, j, k]))*halfRL*grid.recip_drC[k]
                 * coordFac)
        tempV = (((vVel[i, j, k-1] + vVel[i, j+1, k-1])
                  - (vVel[i, j, k] + vVel[i, j+1, k]))*halfRL*grid.recip_drC[k]
                 * coordFac)
        verticalShear = tempU*tempU + tempV*tempV                       # :501

    if useLANGMUIR:                                                     # :510-557
        kk = loops_kji((2, Nr), fj, fi)
        depthFac = recip_Lasq*glibc_exp(recip_LD*grid.rF[kk[0]])         # :513  EXP( recip_LD*rF(k) )
        dsU = TKE.local("dstokesUdR").at[kk[2], kk[1], kk[0]].set(recip_LD * uStar[kk[2], kk[1]] * depthFac)
        dsV = TKE.local("dstokesVdR").at[kk[2], kk[1], kk[0]].set(recip_LD * vStar[kk[2], kk[1]] * depthFac)
        if ggl.calcMeanVertShear:                                       # :524-537
            stokesterm = ((tempU*dsU[i, j, k] + tempUp*dsU[i+1, j, k])
                          + (tempV*dsV[i, j, k] + tempVp*dsV[i, j+1, k])
                          )*halfRL*grid.recip_drC[k]*coordFac*coordFac
        else:                                                           # :540-555
            stokesterm = halfRL*coordFac*(tempU*(dsU[i, j, k]+dsU[i+1, j, k])
                                          + tempV*(dsV[i, j, k]+dsV[i, j+1, k]))

    N2 = Nsquare[i, j, k]
    if useIDEMIX:                                                       # :565-576
        RiNumber = MAX(N2, 0.0, p="a")/(verticalShear+GGL90eps)        # :568-569
        IDEMIX_RiNumber = (MAX(KappaM*N2, 0.0, p="a")                   # :570-571
                           / (GGL90eps + IDEMIX_gTKE[i, j, k]))
        prTemp = 6.6*MIN(RiNumber, IDEMIX_RiNumber, p="b")              # :572  6.6 _d 0 * MIN(...)
        Pr = MIN(10.0, prTemp, p="b")                                   # :573
        Pr = MAX(oneRL, Pr, p="b")                                      # :574
    else:                                                               # :579-587
        RiNumber = MAX(N2, 0.0, p="a")/(verticalShear+GGL90eps)        # :581-582
        prTemp = jnp.where(RiNumber >= 0.2, 5.0 * RiNumber, 1.0)        # :583-584  0.2 _d 0, 5. _d 0, 1. _d 0
        Pr = MIN(10.0, prTemp, p="b")                                   # :585
    TKEPrandtlNumber = TKEPrandtlNumber.at[i, j, k].set(Pr)
    Pr = TKEPrandtlNumber[i, j, k]
    KappaH = KappaM/Pr                                                  # :598
    KappaE = KappaE.at[i, j, k].set(ggl.GGL90alpha * KappaM            # :599-600
                                    * maskC[i, j, k]*maskC[i, j, k-1])
    TKEdissipation = (explDissFac*ggl.GGL90ceps                         # :603-605
                      * SQRTTKE[i, j, k]*rMixingLength[i, j, k]
                      * TKE[i, j, k])
    TKE = TKE.at[i, j, k].set(TKE[i, j, k]                              # :607-612
                              + deltaTloc*(
                                  + KappaM*verticalShear
                                  - KappaH*N2
                                  - TKEdissipation))
    if useIDEMIX:                                                       # :617-625
        TKE = TKE.at[i, j, k].set(TKE[i, j, k] + deltaTloc*IDEMIX_gTKE[i, j, k])
    if useLANGMUIR:                                                     # :629-637
        TKE = TKE.at[i, j, k].set(TKE[i, j, k] + deltaTloc*(KappaM*stokesterm))

    # :669-700  lower diagonal (GGL90_MISSING_HFAC_BUG: :677-685 inside the same k loop, before a3d(k))
    jj, ii = loop_j(jMin, jMax), loop_i(iMin, iMax)
    a3d = a3d.at[ii, jj, 1].set(0.0)                                    # :671-675
    if cfg.cpp.flag("GGL90_MISSING_HFAC_BUG", opt) and not ggl.useIDEMIX:
        k, j, i = loops_kji((2, Nr), fj, fi)
        recip_hFacI = recip_hFacI.at[i, j, k].set(oneRS)                # :681
    for klo, khi in ((2, 2), (3, Nr)):                                  # km1 = MAX(2,k-1): 2 at k = 2, else k-1
        if khi < klo:
            continue
        k, j, i = loops_kji((klo, khi), (jMin, jMax), (iMin, iMax))
        KappaE_km1 = KappaE[i, j, 2] if klo == 2 else KappaE[i, j, k-1]
        a3d = a3d.at[i, j, k].set(-deltaTloc                            # :693-697
                                  * grid.recip_drF[k-1]*recip_hFacC[i, j, k-1]
                                  * .5*(KappaE[i, j, k]+KappaE_km1)
                                  * grid.recip_drC[k]*maskC[i, j, k]*recip_hFacI[i, j, k]
                                  * coordFac*coordFac)
    # :701-722  upper diagonal; z coordinates: kp1 = MAX(1,MIN(klowC(i,j),k+1)) (:711)
    c3d = c3d.at[ii, jj, 1].set(0.0)                                    # :702-706
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    kp1 = jnp.maximum(1, jnp.minimum(grid.kLowC[i, j], kvals(k)+1))   # MINMAX-RAW: integer level index (no tie/NaN)
    KappaE_kp1 = jnp.take_along_axis(KappaE.data[:, :, jMin-fj[0]:jMax-fj[0]+1, iMin-fi[0]:iMax-fi[0]+1],
                                     kp1 - 1, axis=1)                   # KappaE(i,j,kp1): storage index kp1-1
    c3d = c3d.at[i, j, k].set(-deltaTloc                                # :715-719
                              * grid.recip_drF[k] * recip_hFacC[i, j, k]
                              * .5*(KappaE[i, j, k]+KappaE_kp1)
                              * grid.recip_drC[k]*maskC[i, j, k-1]*recip_hFacI[i, j, k]
                              * coordFac*coordFac)

    k, j, i = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))
    kBot = jnp.maximum(grid.kLowC[i, j], 1)                            # MINMAX-RAW: integer level index (no tie/NaN)
    if not ggl.GGL90_dirichlet:                                         # :724-741  c3d(i,j,kBot) = 0
        c3d = c3d.at[i, j, k].set(jnp.where(kvals(k) == kBot, 0.0, c3d[i, j, k]))   # :736-737

    # :744-754  center diagonal, km1 = MAX(k-1,1)
    for klo, khi in ((1, 1), (2, Nr)):
        k, j, i = loops_kji((klo, khi), (jMin, jMax), (iMin, iMax))
        mkm1 = maskC[i, j, 1] if klo == 1 else maskC[i, j, k-1]
        b3d = b3d.at[i, j, k].set(1.0 - c3d[i, j, k] - a3d[i, j, k]    # :748-751
                                  + implDissFac*deltaTloc*ggl.GGL90ceps*SQRTTKE[i, j, k]
                                  * rMixingLength[i, j, k]
                                  * maskC[i, j, k]*mkm1)

    # :766-797  boundary condition: friction velocity
    j, i = loop_j(jMin, jMax), loop_i(iMin, iMax)
    sfU, sfV = state.surfaceForcingU, state.surfaceForcingV
    if ggl.calcMeanVertShear:                                           # :768-785
        tempU, tempUp = sfU[i, j], sfU[i+1, j]
        tempV, tempVp = sfV[i, j], sfV[i, j+1]
        uStarSquare = uStarSquare.at[i, j].set(
            (tempU*tempU + tempUp*tempUp
             + tempV*tempV + tempVp*tempVp
             )*halfRL)
    else:                                                               # :787-796
        uStarSquare = uStarSquare.at[i, j].set(
            (.5*(sfU[i, j] + sfU[i+1, j]))**2
            + (.5*(sfV[i, j] + sfV[i, j+1]))**2)
    us = uStarSquare[i, j]
    if autodiff:                                                        # :868-870
        uStarSquare = uStarSquare.at[i, j].set(jnp.where(us > 0.0, safe_sqrt(us, us > 0.0)*recip_coordFac, us))
    else:                                                               # :872
        uStarSquare = uStarSquare.at[i, j].set(safe_sqrt(us, us > 0.0)*recip_coordFac)

    # :901-915  Dirichlet surface boundary condition (z coordinates)
    TKE = TKE.at[i, j, kSrf].set(maskC[i, j, kSrf]                      # :909-910
                                 * MAX(ggl.GGL90TKEsurfMin, ggl.GGL90m2*uStarSquare[i, j], p="a"))
    TKE = TKE.at[i, j, kTop].set(TKE[i, j, kTop]                        # :911-912
                                 - a3d[i, j, kTop]*TKE[i, j, kSrf])
    a3d = a3d.at[i, j, kTop].set(0.0)                                   # :913

    if ggl.GGL90_dirichlet:                                             # :918-939  (z coordinates :930-937)
        k, j, i = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))
        at_bot = kvals(k) == kBot
        TKE = TKE.at[i, j, k].set(jnp.where(at_bot, TKE[i, j, k] - ggl.GGL90TKEbottom*c3d[i, j, k], TKE[i, j, k]))
        c3d = c3d.at[i, j, k].set(jnp.where(at_bot, 0.0, c3d[i, j, k]))

    TKE, _ = solve_tridiagonal(iMin, iMax, jMin, jMax, a3d, b3d, c3d, TKE, -1, cfg=cfg)    # :945-950

    # :955-989  impose minimum TKE
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    TKE = TKE.at[i, j, k].set(maskC[i, j, k]*maskC[i, j, k-1]           # :972-973
                              * MAX(TKE[i, j, k], ggl.GGL90TKEmin, p="b"))

    # :994-1074  viscosity and diffusivity
    k, j, i = loops_kji((2, Nr), (1, sz.sNy), (1, sz.sNx))
    tmpVisc = GGL90visctmp[i, j, k]                                     # :1021
    tmpVisc = (MIN(tmpVisc/TKEPrandtlNumber[i, j, k], ggl.GGL90diffMax, p="a")   # :1023-1024
               * coordFac*coordFac)
    diffKr = ggl.GGL90diffKr.at[i, j, k].set(MAX(tmpVisc, params.diffKrNrS[k], p="b"))       # :1025
    k, j, i = loops_kji((2, Nr), (1, sz.sNy), (1, sz.sNx+1))
    tmpVisc = (grid.maskW[i, j, k-1] * grid.maskW[i, j, k] * halfRL    # :1042-1044
               * (GGL90visctmp[i-1, j, k]
                  + GGL90visctmp[i, j, k]))
    tmpVisc = (MIN(tmpVisc, ggl.GGL90viscMax, p="a")                    # :1046-1047
               * coordFac*coordFac)
    viscArU = ggl.GGL90viscArU.at[i, j, k].set(MAX(tmpVisc, params.viscArNr[k], p="b"))     # :1048
    k, j, i = loops_kji((2, Nr), (1, sz.sNy+1), (1, sz.sNx))
    tmpVisc = (grid.maskS[i, j, k-1] * grid.maskS[i, j, k] * halfRL    # :1065-1067
               * (GGL90visctmp[i, j-1, k]
                  + GGL90visctmp[i, j, k]))
    tmpVisc = (MIN(tmpVisc, ggl.GGL90viscMax, p="a")                    # :1069-1070
               * coordFac*coordFac)
    viscArV = ggl.GGL90viscArV.at[i, j, k].set(MAX(tmpVisc, params.viscArNr[k], p="b"))     # :1071
    return ggl.replace(GGL90TKE=TKE, GGL90diffKr=diffKr, GGL90viscArU=viscArU, GGL90viscArV=viscArV)

