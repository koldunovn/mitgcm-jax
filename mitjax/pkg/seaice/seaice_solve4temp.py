"""SEAICE_SOLVE4TEMP: pkg/seaice/seaice_solve4temp.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.libm import glibc_exp
from mitjax.ops.safe import div, safe_div
from mitjax.pkg.seaice.seaice_params_h import ZERO



# seaice_solve4temp.F:267, :276 (#ifndef EXF_LWDOWN_WITH_EMISSIVITY): "use the old hard wired inconsistent value"
# 0.97 _d 0 (lane M4ADCOL)
LW_HARDWIRED = 0.97
# MAYKUT CONSTANTS FOR SAT. VAP. PRESSURE TEMP. POLYNOMIAL (:172-177; REAL*8 literals; lane M4ADLAB session 2)
C1 = 2.7798202e-06                                                             # :172  2.7798202  _d -06
C2 = -2.6913393e-03                                                            # :173 -2.6913393  _d -03
C3 = 0.97920849                                                                # :174  0.97920849 _d +00
C4 = -158.63779                                                                # :175 -158.63779  _d +00
C5 = 9653.1925                                                                 # :176 9653.1925   _d +00
QS1 = 0.622/1013.0                                                             # :177 0.622 _d +00/1013.0 _d +00

def seaice_solve4temp(UG, HICE_ACTUAL, HSNOW_ACTUAL, TSURFin, myTime, myIter, *, cfg, sp, op, exf, grid, state):
    """SEAICE_SOLVE4TEMP( UG, HICE_ACTUAL, HSNOW_ACTUAL, TSURFin, TSURFout, F_ia, IcePenetSW, FWsublim, bi, bj,
    myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_solve4temp.F:12-630

    C     | o Calculate ice growth rate, surface fluxes and
    C     |   temperature of ice surface.
    C     |   see Hibler, MWR, 108, 1943-1973, 1980

    Arguments (1:sNx,1:sNy) as interior arrays [tile, j, i] (the values of the caller's (i,j) loop): UG, HICE_ACTUAL,
    HSNOW_ACTUAL, TSURFin. `exf` EXF_FIELDS.h (LWDOWN, ATEMP, AQH, SWDOWN), `state` (salt), `grid` (yC). Returns the
    outputs (TSURFout, F_ia, IcePenetSW, FWsublim), each [tile, j, i]; on lanes without ice (`iceOrNot` .FALSE.)
    TSURFout = TSURFin and the fluxes are 0 (:232-237).
    Ported: ALLOW_ATM_TEMP + ALLOW_DOWNWARD_RADIATION with or (lane M4ADCOL) without EXF_LWDOWN_WITH_EMISSIVITY,
    useMaykutSatVapPoly .FALSE. or (lane M4ADLAB) .TRUE., postSolvTempIter = 2 or (lane M4ADLAB) 0, no
    SEAICE_CAP_SUBLIM / SEAICE_USE_GROWTH_ADX / SEAICE_MODIFY_GROWTH_ADJ (each raises; postSolvTempIter = 1 raises). The (I,J) loops are independent:
    vectorised; the ITER loop (IMAX_TICE, a static count) is a Python loop (6 passes of a short point-wise body).
    `IF ( iceOrNot )` is a where: the arms not taken are computed with guarded divisions (effConduct, safe_div on
    iceOrNot) so that their derivatives stay finite; t1 there is TSURFin (the previous ice temperature, > 0).
    `IF ( HCUT.GT.0. )` (:217) on the traced SEAICE_snowThick is a where with safe_div. The Maykut constants C1..C5,
    QS1 (:172-177) are module constants. F_io_net, F_ia_net (:534-546) are locals only GROWTH_ADX reads:
    computed as the Fortran does and not returned. EXP is libm's (glibc_exp); LOG is XLA's (bitwise glibc)."""
    for o in ("SEAICE_CAP_SUBLIM", "SEAICE_USE_GROWTH_ADX", "SEAICE_MODIFY_GROWTH_ADJ"):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h"):
            raise NotImplementedError(f"SEAICE_SOLVE4TEMP: {o} is not ported")
    if not (cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h")
            and cfg.cpp.flag("ALLOW_DOWNWARD_RADIATION", "EXF_OPTIONS.h")):
        raise NotImplementedError("SEAICE_SOLVE4TEMP: only ALLOW_ATM_TEMP + ALLOW_DOWNWARD_RADIATION is ported")
    # lane M4ADCOL: 1D_ocean_ice_column/code_ad's EXF_OPTIONS.h leaves EXF_LWDOWN_WITH_EMISSIVITY undefined: the
    # "old hard wired inconsistent value" 0.97 _d 0 (:267, :276) multiplies lwdownLoc in both arms
    lw_emiss = cfg.cpp.flag("EXF_LWDOWN_WITH_EMISSIVITY", "EXF_OPTIONS.h")
    # lane M4ADLAB session 2 (lab_sea/input_ad): useMaykutSatVapPoly (:385-388, :447-458, :513-514) and
    # postSolvTempIter = 0 (:563-564: the fluxes of the last iteration stand); postSolvTempIter = 1 raises
    maykut = bool(sp.useMaykutSatVapPoly)
    if sp.postSolvTempIter not in (0, 2):
        raise NotImplementedError("SEAICE_SOLVE4TEMP: postSolvTempIter = 1 (:547-561) is not ported")
    sz = cfg.size
    J = loop_j(1, sz.sNy)
    I = loop_i(1, sz.sNx)
    lnTEN = jnp.log(jnp.float64(10.0))                                         # :179
    aa1 = 2663.5                                                               # :180
    aa2 = 12.537                                                               # :181
    bb1 = 0.622                                                                # :182
    bb2 = 1.0 - bb1                                                            # :183
    Ppascals = 100000.                                                         # :184
    cc0 = glibc_exp(aa2*lnTEN)                                                 # :186
    cc1 = cc0*aa1*bb1*Ppascals*lnTEN                                           # :187
    cc2 = cc0*bb2                                                              # :188
    if op.usingPCoords:                                                        # :190-194
        kSurface = sz.Nr
    else:
        kSurface = 1
    D1 = sp.SEAICE_dalton*sp.SEAICE_cpAir*sp.SEAICE_rhoAir                    # :197
    lhSublim = sp.SEAICE_lhEvap + sp.SEAICE_lhFusion                           # :200
    D1I = sp.SEAICE_dalton*lhSublim*sp.SEAICE_rhoAir                           # :201
    TMELT = op.celsius2K                                                       # :204
    XKI = sp.SEAICE_iceConduct                                                 # :207
    XKS = sp.SEAICE_snowConduct                                                # :210
    HCUT = sp.SEAICE_snowThick                                                 # :215
    recip_HCUT = jnp.where(HCUT > 0., safe_div(1.0, HCUT, HCUT > 0.), 0.)    # :216-217
    XIO = sp.SEAICE_shortwave                                                  # :220
    SurfMeltTemp = TMELT + sp.SEAICE_wetAlbTemp                                # :223
    # :229-280 initialise
    zero = jnp.zeros_like(TSURFin)
    TSURFout = TSURFin                                                         # :232
    F_ia = zero                                                                # :233
    F_ia_net = zero                                                            # :234
    F_io_net = zero                                                            # :235
    IcePenetSW = zero                                                          # :236
    FWsublim = zero                                                            # :237
    iceOrNot = HICE_ACTUAL > 0.                                                # :241
    absorbedSW = zero                                                          # :242
    qhice = zero                                                               # :243
    dqh_dTs = zero                                                             # :244
    F_lh = zero                                                                # :245
    F_lwu = zero                                                               # :246
    F_sens = zero                                                              # :247
    tsurfLoc = TSURFin                                                         # :249
    lwdownLoc = MAX(sp.MIN_LWDOWN, exf["lwdown"][I, J], p="b")                 # :251
    atempLoc = MAX(op.celsius2K+sp.MIN_ATEMP, exf["atemp"][I, J], p="b")       # :252
    tempFrz = (sp.SEAICE_dTempFrz_dS*state.salt[I, J, kSurface]               # :255-256
               + sp.SEAICE_tempFrz0 + op.celsius2K)
    snow = HSNOW_ACTUAL > 0.0                                                  # :260
    D3 = jnp.where(snow, sp.SEAICE_snow_emiss*sp.SEAICE_boltzmann,             # :262, :271
                   sp.SEAICE_ice_emiss*sp.SEAICE_boltzmann)
    if lw_emiss:
        lwdownLoc = jnp.where(snow, sp.SEAICE_snow_emiss*lwdownLoc,            # :265, :274
                              sp.SEAICE_ice_emiss*lwdownLoc)
    else:
        lwdownLoc = jnp.where(snow, LW_HARDWIRED*lwdownLoc,                    # :267  0.97 _d 0
                              LW_HARDWIRED*lwdownLoc)                          # :276  0.97 _d 0
    # :282-364 albedo, shortwave, conductivity (iceOrNot)
    wet = tsurfLoc >= SurfMeltTemp                                             # :289, :297
    south = grid.yC[I, J] < 0.0                                                # :288
    ALB_ICE = jnp.where(south,
                        jnp.where(wet, sp.SEAICE_wetIceAlb_south, sp.SEAICE_dryIceAlb_south),   # :290, :293
                        jnp.where(wet, sp.SEAICE_wetIceAlb, sp.SEAICE_dryIceAlb))               # :298, :301
    ALB_SNOW = jnp.where(south,
                         jnp.where(wet, sp.SEAICE_wetSnowAlb_south, sp.SEAICE_drySnowAlb_south),  # :291, :294
                         jnp.where(wet, sp.SEAICE_wetSnowAlb, sp.SEAICE_drySnowAlb))              # :299, :302
    ALB = jnp.where(HSNOW_ACTUAL > HCUT, ALB_SNOW,                             # :307-316
                    jnp.where(HCUT <= ZERO, ALB_ICE,
                              MIN(ALB_ICE + HSNOW_ACTUAL*recip_HCUT
                                  * (ALB_SNOW - ALB_ICE),
                                  ALB_SNOW, p="b")))                           # :313-315
    penetSWFrac = jnp.where(HSNOW_ACTUAL > 0.0, 0.0,                           # :321-325
                            XIO*glibc_exp(-1.5*HICE_ACTUAL))
    IcePenetSW = jnp.where(iceOrNot, -(1.0 - ALB)                              # :327-328
                           * penetSWFrac * exf["swdown"][I, J], IcePenetSW)
    absorbedSW = jnp.where(iceOrNot, (1.0 - ALB)                               # :330-331
                           * (1.0 - penetSWFrac) * exf["swdown"][I, J], absorbedSW)
    effConduct = safe_div(XKI * XKS,                                           # :338-339
                          (XKS * HICE_ACTUAL + XKI * HSNOW_ACTUAL), iceOrNot)
    AQH = exf["aqh"][I, J]

    def fluxes(t1):
        """:379-422 (and :507-536): the fluxes at surface temperature t1 (every lane; selected by iceOrNot)."""
        t2 = t1*t1                                                             # :380
        t3 = t2*t1                                                             # :381
        t4 = t2*t2                                                             # :382
        if maykut:                                                             # :385-388 (:513-514)
            qh = QS1*(C1*t4+C2*t3 + C3*t2+C4*t1+C5)                            # :387 (:514)
            dqh = jnp.zeros_like(t1)                                           # :388 dqh_dTs = 0. _d 0
        else:
            mm_log10pi = div(-aa1, t1) + aa2                                   # :392
            mm_pi = glibc_exp(mm_log10pi*lnTEN)                                # :397
            qh = div(bb1*mm_pi, (Ppascals - (1.0 - bb1)*mm_pi))                # :398
            cc3t = glibc_exp(div(aa1, t1) * lnTEN)                             # :402
            dqh = div(cc1*cc3t, ((cc2-cc3t*Ppascals)**2 * t2))                 # :404
        Fc = effConduct*(tempFrz-t1)                                           # :408
        Flh = D1I*UG*(qh-AQH)                                                  # :409
        Flwu = t4 * D3                                                         # :419
        Fsens = D1 * UG * (t1 - atempLoc)                                      # :420
        Fia = (-lwdownLoc - absorbedSW + Flwu                                  # :421-422
               + Fsens + Flh)
        return t3, qh, dqh, Fc, Flh, Flwu, Fsens, Fia
    # :367-480 iterations
    for ITER in range(1, sp.IMAX_TICE + 1):
        tsurfPrev = tsurfLoc                                                   # :376
        t1 = tsurfLoc                                                          # :379
        t3, qh, dqh, F_c, Flh, Flwu, Fsens, Fia = fluxes(t1)
        qhice = jnp.where(iceOrNot, qh, qhice)
        dqh_dTs = jnp.where(iceOrNot, dqh, dqh_dTs)
        F_lh = jnp.where(iceOrNot, Flh, F_lh)
        F_lwu = jnp.where(iceOrNot, Flwu, F_lwu)
        F_sens = jnp.where(iceOrNot, Fsens, F_sens)
        F_ia = jnp.where(iceOrNot, Fia, F_ia)
        dFia_dTs = (4.0*D3*t3 + D1*UG                                          # :424-425
                    + D1I*UG*dqh_dTs)
        tsurfLoc = jnp.where(iceOrNot, tsurfLoc                                # :439-440
                             + div((F_c-F_ia), (effConduct+jnp.where(iceOrNot, dFia_dTs, 1.0))), tsurfLoc)
        if maykut:                                                             # :447-458 (every point)
            tsurfLoc = MAX(op.celsius2K+sp.MIN_TICE, tsurfLoc, p="b")          # :451
        tsurfLoc = MIN(tsurfLoc, TMELT, p="b")                                 # :466
    del tsurfPrev
    if sp.postSolvTempIter == 2:                                               # :502-551 (else :563-564: none)
        t1 = tsurfLoc                                                          # :507
        _, qh, _, F_c, Flh, Flwu, Fsens, Fia = fluxes(t1)
        qhice = jnp.where(iceOrNot, qh, qhice)                                 # :517-523
        F_lh = jnp.where(iceOrNot, Flh, F_lh)                                  # :526
        F_lwu = jnp.where(iceOrNot, Flwu, F_lwu)                               # :532
        F_sens = jnp.where(iceOrNot, Fsens, F_sens)                            # :533
        F_ia = jnp.where(iceOrNot, Fia, F_ia)                                  # :535-536
        out = -F_c < ZERO                                                      # :539
        F_io_net = jnp.where(iceOrNot, jnp.where(out, F_c, ZERO), F_io_net)    # :541, :544
        F_ia_net = jnp.where(iceOrNot, jnp.where(out, ZERO, F_ia), F_ia_net)   # :542, :545
    del F_io_net, F_ia_net
    # :570-626
    TSURFout = jnp.where(iceOrNot, tsurfLoc, TSURFout)                         # :575
    FWsublim = jnp.where(iceOrNot, F_lh/lhSublim, FWsublim)                    # :579
    return TSURFout, F_ia, IcePenetSW, FWsublim
