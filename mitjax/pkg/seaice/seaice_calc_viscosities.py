"""SEAICE_CALC_VISCOSITIES: pkg/seaice/seaice_calc_viscosities.F @63cdc0b (lane M4OFF session 3, the C-grid build)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.libm import glibc_exp, powi
from mitjax.ops.safe import safe_sqrt
from mitjax.pkg.seaice.seaice_params_h import HALF, ONE


def seaice_calc_viscosities(e11, e22, e12, zMin, zMax, HEFFM, press0, tnsFac, eta, etaZ, zeta, zetaZ, press, deltaC,
                            iStep, myTime, myIter, *, cfg, sp, grid):
    """SEAICE_CALC_VISCOSITIES( e11, e22, e12, zMin, zMax, HEFFM, press0, tnsFac, eta, etaZ, zeta, zetaZ, press,
    deltaC, iStep, myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_calc_viscosities.F:8-482

    C     | o compute shear and bulk viscositites eta, zeta and the
    C     |   corrected ice strength P

    Returns (eta, etaZ, zeta, zetaZ, press, deltaC): the points of 1-OLx+1..sNx+OLx-1, 1-OLy+1..sNy+OLy-1 written,
    every other point as given (the Fortran arguments are SEAICE.h fields). Ported: the build's options
    (SEAICE_ALLOW_TEARDROP, SEAICE_ALLOW_MCE/TEM, SEAICE_ALLOW_MCS, SEAICE_ZETA_SMOOTHREG defined; SEAICE_DELTA_SMOOTHREG,
    ALLOW_AUTODIFF not; lane M4LAB session 3: lab_sea's build without SEAICE_ZETA_SMOOTHREG and without the TD / MC /
    TEM options) and the default rheology: the ELSE arm :334-447 (elliptical yield curve with the smooth
    regularisation :365-372, or without SEAICE_ZETA_SMOOTHREG zeta = HALF*press0*(1+tnsFac)/deltaCreg capped by MIN
    zMax, :373-379, MIN site :378 (b)) with SEAICEuseMCE .AND. SEAICEuseTEM .FALSE. (:435-442: eta = zeta/e**2);
    SEAICEuseTD / SEAICEusePL / SEAICEuseMCS / SEAICEuseMCE / SEAICEuseTEM raise in SEAICE_READPARMS (:161-332,
    :397-434). SEAICEetaZmethod 0 or 3 (:117-138); 1 and 2 raise there. The (i,j) loops are independent per point:
    vectorised; MAX sites :155 (b), :363 (a), :380 (b) from the build's table. EXP is glibc's; `**2` and `**4` are
    gfortran -O0's __powidf2 (libm.powi).
    Lane M4ADLAB session 2 (lab_sea/code_ad): ALLOW_AUTODIFF. Its arms: the local initialisations :106-114
    (deltaCreg = SEAICE_deltaMin, e12Csq = recip_shear = 0 on every point: the port forms these locals only on the
    points the loops below write and reads nothing else, so no value changes), the shearDef guard :148-151 (in
    recip_shear, read only by the TD / PL / MCS / MCE arms that raise: not formed) and the deltaC guard :350-353
    (deltaC = 0 unless deltaCsq > 0, then SQRT: safe_sqrt on that condition, the same values as the plain SQRT for
    deltaCsq >= +0 and a finite derivative at 0)."""
    del iStep, myTime, myIter
    for o in ("SEAICE_DELTA_SMOOTHREG",):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h") or cfg.cpp.flag(o):
            raise NotImplementedError(f"SEAICE_CALC_VISCOSITIES: {o} is not ported")
    autodiff = cfg.cpp.flag("ALLOW_AUTODIFF")
    zeta_smoothreg = cfg.cpp.flag("SEAICE_ZETA_SMOOTHREG", "SEAICE_OPTIONS.h")   # :365-379
    if sp.SEAICEuseTD or sp.SEAICEusePL or sp.SEAICEuseMCS or sp.SEAICEuseMCE or sp.SEAICEuseTEM:
        raise NotImplementedError("SEAICE_CALC_VISCOSITIES: TD / PL / MCS / MCE / TEM rheologies are not ported")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    recip_rA, rAz = grid.recip_rA, grid.rAz
    smallNbr = 1.0e-20                                                         # :91
    recip_e2 = 0.0                                                             # :94
    recip_e2 = jnp.where(sp.SEAICE_eccen != 0.0, ONE/powi(sp.SEAICE_eccen, 2), recip_e2)   # :95
    recip_efr2 = 0.0                                                           # :96
    recip_efr4 = 0.0                                                           # :97
    recip_efr2 = jnp.where(sp.SEAICE_eccfr != 0.0, ONE/powi(sp.SEAICE_eccfr, 2), recip_efr2)   # :98-101
    recip_efr4 = jnp.where(sp.SEAICE_eccfr != 0.0, powi(sp.SEAICE_eccen, 2) / powi(sp.SEAICE_eccfr, 4), recip_efr4)
    del recip_e2                       # read only by the TD/PL arms and the CML comment (:345-348)
    j = loop_j(1-OLy+1, sNy+OLy-1)                                             # :118-119, :127-128
    i = loop_i(1-OLx+1, sNx+OLx-1)
    if sp.SEAICEetaZmethod == 0:                                               # :117-125
        # tmp = 0.25 * (...) with the REAL*4 literal 0.25 (exact in REAL*8)
        tmp = 0.25 * (e12[i, j] + e12[i+1, j] + e12[i, j+1] + e12[i+1, j+1])
        e12Csq = tmp*tmp
    elif sp.SEAICEetaZmethod == 3:                                             # :126-137
        e12Csq = 0.25 * recip_rA[i, j] * (
            rAz[i, j]*powi(e12[i, j], 2)
            + rAz[i+1, j]*powi(e12[i+1, j], 2)
            + rAz[i, j+1]*powi(e12[i, j+1], 2)
            + rAz[i+1, j+1]*powi(e12[i+1, j+1], 2))
    else:
        raise NotImplementedError(f"SEAICE_CALC_VISCOSITIES: SEAICEetaZmethod {sp.SEAICEetaZmethod}")
    # :139-158 recip_shear: read only by the TD / PL / MCS / MCE arms (MAX site :155 b); not formed
    # :334-394 the ELSE arm
    ep = e11[i, j]+e22[i, j]                                                   # :339
    em = e11[i, j]-e22[i, j]                                                   # :340
    shearDefSq = em*em + 4.0*e12Csq                                            # :341
    deltaCsq = ep*ep + recip_efr4*shearDefSq                                   # :342
    if autodiff:                                                               # :350-353 (ALLOW_AUTODIFF)
        deltaC = deltaC.at[i, j].set(safe_sqrt(deltaCsq, deltaCsq > 0.0))      # :351-353 (0. _d 0, :351)
    else:
        deltaC = deltaC.at[i, j].set(jnp.sqrt(deltaCsq))                       # :355
    deltaCreg = MAX(deltaC[i, j], sp.SEAICE_deltaMin, p="a")                   # :363
    if zeta_smoothreg:                                                         # :365-372
        argTmp = glibc_exp(-1.0/(deltaCreg*sp.SEAICE_zetaMaxFac))              # :369
        zeta = zeta.at[i, j].set(zMax[i, j]                                    # :370-372
                                 * (1.0 - argTmp)/(1.0 + argTmp)
                                 * (1.0 + tnsFac[i, j]))
    else:                                                                      # :373-379 (lane M4LAB session 3)
        zeta = zeta.at[i, j].set(HALF*(press0[i, j]                            # :374-376
                                       * (1.0 + tnsFac[i, j])
                                       )/deltaCreg)
        zeta = zeta.at[i, j].set(MIN(zMax[i, j], zeta[i, j], p="b"))           # :378
    zeta = zeta.at[i, j].set(MAX(zMin[i, j], zeta[i, j], p="b"))               # :380
    zeta = zeta.at[i, j].set(zeta[i, j]*HEFFM[i, j])                           # :382
    press = press.at[i, j].set(                                                # :384-388
        (press0[i, j]*(1.0 - sp.SEAICEpressReplFac)
         + 2.0*zeta[i, j]*deltaC[i, j]
         * sp.SEAICEpressReplFac/(1.0 + tnsFac[i, j])
         ) * (1.0 - tnsFac[i, j]))
    eta = eta.at[i, j].set(zeta[i, j] * recip_efr2)                            # :440 (not MCE, not TEM)
    sumNorm = HEFFM[i, j]+HEFFM[i-1, j] + HEFFM[i, j-1]+HEFFM[i-1, j-1]        # :453-454
    sumNorm = jnp.where(sumNorm > 0.0, 1.0 / jnp.where(sumNorm > 0.0, sumNorm, 1.0), sumNorm)   # :455
    etaZ = etaZ.at[i, j].set(sumNorm *                                         # :456-458
                             (eta[i, j] + eta[i-1, j]
                              + eta[i, j-1] + eta[i-1, j-1]))
    zetaZ = zetaZ.at[i, j].set(sumNorm *                                       # :459-461
                               (zeta[i, j] + zeta[i-1, j]
                                + zeta[i, j-1] + zeta[i-1, j-1]))
    if not sp.SEAICE_no_slip:                                                  # :467-476
        maskZ = (HEFFM[i, j]*HEFFM[i-1, j]
                 * HEFFM[i, j-1]*HEFFM[i-1, j-1])
        etaZ = etaZ.at[i, j].set(etaZ[i, j] * maskZ)
        zetaZ = zetaZ.at[i, j].set(zetaZ[i, j] * maskZ)
    return eta, etaZ, zeta, zetaZ, press, deltaC
