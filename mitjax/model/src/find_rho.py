"""model/src/find_rho.F @63cdc0b: routines to compute density (FIND_RHO_2D, FIND_RHOP0, FIND_BULKMOD,
FIND_RHO_SCALAR).

Ported for the M1 equations of state (eosType in each experiment's `data`): 'LINEAR' (tutorial_baroclinic_gyre),
'JMD95Z' (tutorial_global_oce_optim/input_ad, global_ocean.90x40x15/input_ad*), 'JMD95P'
(global_ocean.90x40x15/input), and (lane COLMIX, M3) 'MDJWF' (vermix; FIND_RHONUM, FIND_RHODEN). The JMD95
polynomials are the non-factorized forms: USE_FACTORIZED_EOS is defined only under TARGET_NEC_SX (find_rho.F:3-6),
which no M1 build defines (each `#ifdef USE_FACTORIZED_EOS` arm raises). 'POLY3', 'TEOS10', 'IDEALG' raise
NotImplementedError (FIND_RHOTEOS is not ported); LOOK_FOR_NEG_SALINITY is not called by any M1 build
(CHECK_SALINITY_FOR_NEGATIVE_VALUES) and not ported.

Common blocks, by their Fortran names: `params` (PARAMS.h: tRef, sRef, rhoNil, rhoConst, surf_pRef; traced floats),
`eos` (EOS.h as INI_EOS leaves it, mitjax/model/src/ini_eos.py: equationOfState static, the coefficient vectors as
FArrays with their declared bounds), and for PRESSURE_FOR_EOS `grid`, `state` (mitjax/model/src/pressure_for_eos.py).
"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.src.pressure_for_eos import pressure_for_eos
from mitjax.ops.safe import safe_sqrt

# model/inc/EOS.h:18-20  PARAMETER ( SItoBar = 1.D-05 ), PARAMETER ( SItodBar = 1.D-04 )
SItoBar = 1.0e-05
SItodBar = 1.0e-04
# find_rho.F:698 (FIND_RHODEN) and :870 (FIND_RHO_SCALAR)  PARAMETER ( epsln = 0. _d 0 )
epsln = 0.0


def _no_factorized_eos(cfg, routine):
    # find_rho.F:3-6: USE_FACTORIZED_EOS is defined iff TARGET_NEC_SX
    if cfg.cpp.TARGET_NEC_SX:
        raise NotImplementedError(f"{routine}: USE_FACTORIZED_EOS (TARGET_NEC_SX) polynomials are not ported")


def _is_jmd95_or_unesco(equationOfState):
    # equationOfState(1:5).EQ.'JMD95' .OR. equationOfState.EQ.'UNESCO'  (CHARACTER*(6), blank padded)
    return equationOfState[:5] == "JMD95" or equationOfState.rstrip() == "UNESCO"


def find_rho_2d(iMin, iMax, jMin, jMax, kRef, tFld, sFld, rhoLoc, k, *, cfg, grid, params, eos, state):
    """FIND_RHO_2D( iMin, iMax, jMin, jMax, kRef, tFld, sFld, rhoLoc, k, bi, bj, myThid )
    @63cdc0b model/src/find_rho.F:22-269

    C     *==========================================================*
    C     | o SUBROUTINE FIND_RHO_2D
    C     |   Calculates [rho(S,T,z)-rhoConst] of a 2-D slice
    C     *==========================================================*
    C     | kRef - determines pressure reference level
    C     |        (not used in 'LINEAR' mode)
    C     | Note:  k is not used ; keep it for debugging.
    C     *==========================================================*

    tFld, sFld, rhoLoc: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy); kRef, k Python ints. Returns rhoLoc: under
    ALLOW_AUTODIFF (code_ad builds) every point is first set to 0 (:75-83), otherwise points outside iMin:iMax,
    jMin:jMax keep their prior values. Ported: 'LINEAR' (:92-110) and 'JMD95Z'/'JMD95P' (:147-183; the 'UNESCO'
    coefficients are refused by INI_EOS); the other equations of state raise; CHECK_SALINITY_FOR_NEGATIVE_VALUES
    (:85-90) raises.
    """
    eqs = eos.equationOfState
    sz = cfg.size
    locPres = rhoLoc.local("locPres")
    rhoP0 = rhoLoc.local("rhoP0")
    bulkMod = rhoLoc.local("bulkMod")
    if cfg.cpp.ALLOW_AUTODIFF:                                          # :75-83  (a forward branch of code_ad builds)
        j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
        i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        rhoLoc = rhoLoc.at[i, j].set(0.0)
        rhoP0 = rhoP0.at[i, j].set(0.0)
        bulkMod = bulkMod.at[i, j].set(0.0)
    if cfg.cpp.CHECK_SALINITY_FOR_NEGATIVE_VALUES:                      # :85-90
        raise NotImplementedError("FIND_RHO_2D: CHECK_SALINITY_FOR_NEGATIVE_VALUES (LOOK_FOR_NEG_SALINITY) is not "
                                  "ported")
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if eqs.rstrip() == "LINEAR":                                       # :92
        refTemp = params.tRef[kRef]                                     # :98
        refSalt = params.sRef[kRef]                                     # :99
        dRho = params.rhoNil-params.rhoConst                            # :101
        rhoLoc = rhoLoc.at[i, j].set(                                   # :103-110
            params.rhoNil*(
                eos.sBeta*(sFld[i, j]-refSalt)
                - eos.tAlpha*(tFld[i, j]-refTemp))
            + dRho)
    elif eqs.rstrip() == "POLY3":                                      # :112-145
        raise NotImplementedError("FIND_RHO_2D: equationOfState = 'POLY3' is not ported")
    elif _is_jmd95_or_unesco(eqs):                                      # :147-183
        dp0 = params.surf_pRef - eos.eosRefP0                           # :150
        locPres = pressure_for_eos(iMin, iMax, jMin, jMax, kRef, dp0, locPres,      # :152-155
                                   cfg=cfg, grid=grid, params=params, state=state)
        rhoP0 = find_rhop0(iMin, iMax, jMin, jMax, tFld, sFld, rhoP0,               # :157-161
                           cfg=cfg, eos=eos)
        bulkMod = find_bulkmod(iMin, iMax, jMin, jMax, locPres, tFld, sFld, bulkMod,  # :163-167
                               cfg=cfg, eos=eos)
        rhoLoc = rhoLoc.at[i, j].set(                                   # :173-183
            rhoP0[i, j]
            / (1.0 -
               locPres[i, j]*SItoBar/bulkMod[i, j])
            - params.rhoConst)
    elif eqs.rstrip() == "MDJWF":                                      # :185-213 (lane COLMIX, M3 vermix)
        dp0 = params.surf_pRef - eos.eosRefP0                           # :186
        locPres = pressure_for_eos(iMin, iMax, jMin, jMax, kRef, dp0, locPres,      # :188-191
                                   cfg=cfg, grid=grid, params=params, state=state)
        rhoNum = find_rhonum(iMin, iMax, jMin, jMax, locPres, tFld, sFld,           # :193-197
                             rhoLoc.local("rhoNum"), eos=eos)
        rhoDen = find_rhoden(iMin, iMax, jMin, jMax, locPres, tFld, sFld,           # :199-203
                             rhoLoc.local("rhoDen"), eos=eos)
        rhoLoc = rhoLoc.at[i, j].set(rhoNum[i, j]*rhoDen[i, j] - params.rhoConst)  # :208-212
    elif eqs.rstrip() in ("TEOS10", "IDEALG"):                         # :215-259
        raise NotImplementedError(f"FIND_RHO_2D: equationOfState = '{eqs.rstrip()}' is not ported")
    else:                                                               # :261-266
        raise ValueError(f' FIND_RHO_2D: equationOfState = "{eqs}"\nABNORMAL END: S/R FIND_RHO_2D')
    return rhoLoc


def find_rhop0(iMin, iMax, jMin, jMax, tFld, sFld, rhoP0, *, cfg, eos):
    """FIND_RHOP0( iMin, iMax, jMin, jMax, tFld, sFld, rhoP0, myThid )   @63cdc0b model/src/find_rho.F:271-406

    C     *==========================================================*
    C     | o SUBROUTINE FIND_RHOP0
    C     |   Calculates rho(S,T,0) of a slice
    C     *==========================================================*

    Returns rhoP0 (points outside iMin:iMax, jMin:jMax keep their prior values). Non-factorized polynomials
    (:363-370, :383-397). The salinity branch `IF ( s .GT. 0. _d 0 )` (:347-352) is a pointwise where; SQRT is taken
    only where s > 0 (guarded: finite derivatives on the s <= 0 lanes).
    """
    _no_factorized_eos(cfg, "FIND_RHOP0")
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    # :337-340 abbreviations
    t = tFld[i, j]
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    s = sFld[i, j]                                                      # :346
    pos = s > 0.0                                                       # :347  IF ( s .GT. 0. _d 0 ) THEN
    s3o2 = jnp.where(pos, s*safe_sqrt(s, pos), 0.0)                     # :348 s*SQRT(s) / :351 0. _d 0
    s = jnp.where(pos, s, 0.0)                                          # :350  s = 0. _d 0
    Fw, Sw = eos.eosJMDCFw, eos.eosJMDCSw
    # :364-370 density of freshwater at the surface
    rfresh = (
        Fw[1]
        + Fw[2]*t
        + Fw[3]*t2
        + Fw[4]*t3
        + Fw[5]*t4
        + Fw[6]*t4*t)
    # :384-397 density of sea water at the surface
    rsalt = (
        s*(
            Sw[1]
            + Sw[2]*t
            + Sw[3]*t2
            + Sw[4]*t3
            + Sw[5]*t4)
        + s3o2*(
            Sw[6]
            + Sw[7]*t
            + Sw[8]*t2)
        + Sw[9]*s*s)
    return rhoP0.at[i, j].set(rfresh + rsalt)                          # :400


def find_bulkmod(iMin, iMax, jMin, jMax, locPres, tFld, sFld, bulkMod, *, cfg, eos):
    """FIND_BULKMOD( iMin, iMax, jMin, jMax, locPres, tFld, sFld, bulkMod, myThid )
    @63cdc0b model/src/find_rho.F:408-588

    C     *==========================================================*
    C     | o SUBROUTINE FIND_BULKMOD
    C     |   Calculates the secant bulk modulus K(S,T,p) of a slice
    C     *==========================================================*
    C     | k    - is the level of Theta/Salt slice
    C     *==========================================================*

    Returns bulkMod (points outside iMin:iMax, jMin:jMax keep their prior values). Non-factorized polynomials
    (:517-522, :550-579); the salinity branch as in FIND_RHOP0 (:500-505).
    """
    _no_factorized_eos(cfg, "FIND_BULKMOD")
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    # :490-493 abbreviations
    t = tFld[i, j]
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    s = sFld[i, j]                                                      # :499
    pos = s > 0.0                                                       # :500
    s3o2 = jnp.where(pos, s*safe_sqrt(s, pos), 0.0)                     # :501 / :504
    s = jnp.where(pos, s, 0.0)                                          # :503
    p = locPres[i, j]*SItoBar                                           # :507
    p2 = p*p                                                            # :508
    KFw, KSw, KP = eos.eosJMDCKFw, eos.eosJMDCKSw, eos.eosJMDCKP
    # :517-522 secant bulk modulus of fresh water at the surface
    bMfresh = (
        KFw[1]
        + KFw[2]*t
        + KFw[3]*t2
        + KFw[4]*t3
        + KFw[5]*t4)
    # :550-559 secant bulk modulus of sea water at the surface
    bMsalt = (
        s*(KSw[1]
           + KSw[2]*t
           + KSw[3]*t2
           + KSw[4]*t3)
        + s3o2*(KSw[5]
                + KSw[6]*t
                + KSw[7]*t2))
    # :561-579 secant bulk modulus of sea water at pressure p
    bMpres = (
        p*(KP[1]
           + KP[2]*t
           + KP[3]*t2
           + KP[4]*t3)
        + p*s*(KP[5]
               + KP[6]*t
               + KP[7]*t2)
        + p*s3o2*KP[8]
        + p2*(KP[9]
              + KP[10]*t
              + KP[11]*t2)
        + p2*s*(KP[12]
                + KP[13]*t
                + KP[14]*t2))
    return bulkMod.at[i, j].set(bMfresh + bMsalt + bMpres)             # :582


def find_rhonum(iMin, iMax, jMin, jMax, locPres, tFld, sFld, rhoNum, *, eos):
    """FIND_RHONUM( iMin, iMax, jMin, jMax, locPres, tFld, sFld, rhoNum, myThid )   @63cdc0b model/src/find_rho.F:594-654

    C     *==========================================================*
    C     | o SUBROUTINE FIND_RHONUM
    C     |   Calculates the numerator of the McDougall et al.
    C     |   equation of state
    C     |   - the code is more or less a copy of MOM4
    C     *==========================================================*

    Returns rhoNum (points outside iMin:iMax, jMin:jMax keep their prior values). The salinity is used as it is
    (no clipping here, unlike FIND_RHODEN). Pointwise: vectorised over (i, j). Lane COLMIX (M3, vermix's MDJWF).
    """
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    t1 = tFld[i, j]                                                     # :635-637 abbreviations
    t2 = t1*t1
    s1 = sFld[i, j]
    p1 = locPres[i, j]*SItodBar                                         # :639
    num = eos.eosMDJWFnum
    return rhoNum.at[i, j].set(                                         # :641-648
        num[0]
        + t1*(num[1]
              + t1*(num[2] + num[3]*t1))
        + s1*(num[4]
              + num[5]*t1 + num[6]*s1)
        + p1*(num[7] + num[8]*t2
              + num[9]*s1
              + p1*(num[10] + num[11]*t2)))


def find_rhoden(iMin, iMax, jMin, jMax, locPres, tFld, sFld, rhoDen, *, eos):
    """FIND_RHODEN( iMin, iMax, jMin, jMax, locPres, tFld, sFld, rhoDen, myThid )   @63cdc0b model/src/find_rho.F:660-737

    C     *==========================================================*
    C     | o SUBROUTINE FIND_RHODEN
    C     |   Calculates the denominator of the McDougall et al.
    C     |   equation of state
    C     |   - the code is more or less a copy of MOM4
    C     *==========================================================*

    Returns rhoDen = 1/den (points outside iMin:iMax, jMin:jMax keep their prior values). `IF ( s1 .GT. 0. _d 0 )`
    (:710-716) is a pointwise where; SQRT only where s1 > 0 (masked-lane rule, mitjax/ops/safe.py: the s1 <= 0 lanes
    get 0 with a zero derivative, as the Fortran's ELSE sets sp5 = 0). The `#if (defined ALLOW_AUTODIFF && defined
    TARGET_NEC_SX)` line (:706-709) is not compiled by any build (TARGET_NEC_SX). epsln = 0 (:698 PARAMETER).
    Lane COLMIX (M3, vermix's MDJWF).
    """
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    t1 = tFld[i, j]                                                     # :703-705 abbreviations
    t2 = t1*t1
    s1 = sFld[i, j]
    pos = s1 > 0.0                                                      # :710
    sp5 = safe_sqrt(s1, pos)                                            # :711 SQRT(s1) / :714 0. _d 0
    s1 = jnp.where(pos, s1, 0.0)                                        # :713  s1 = 0. _d 0
    p1 = locPres[i, j]*SItodBar                                         # :717
    p1t1 = p1*t1                                                        # :718
    den_ = eos.eosMDJWFden
    den = (den_[0]                                                      # :720-729
           + t1*(den_[1]
                 + t1*(den_[2]
                       + t1*(den_[3] + t1*den_[4])))
           + s1*(den_[5]
                 + t1*(den_[6]
                       + den_[7]*t2)
                 + sp5*(den_[8] + den_[9]*t2))
           + p1*(den_[10]
                 + p1t1*(den_[11]*t2 + den_[12]*p1)))
    return rhoDen.at[i, j].set(1.0/(epsln+den))                         # :731  (1.0 REAL*4, exact)


def find_rho_scalar(tLoc, sLoc, pLoc, *, cfg, params, eos):
    """FIND_RHO_SCALAR( tLoc, sLoc, pLoc, rhoLoc, myThid )   @63cdc0b model/src/find_rho.F:830-1183

    C     *==========================================================*
    C     | o SUBROUTINE FIND_RHO_SCALAR
    C     |   Calculates rho(S,T,p)
    C     | Note: for seawater EOS, p is pressure (Pa) relative to
    C     |   reference surface pressure "surf_pRef"
    C     *==========================================================*

    A pointwise routine: tLoc, sLoc, pLoc are arrays of any (equal) shape, one Fortran call per element; returns
    rhoLoc. Ported: 'LINEAR' (:941-947) and 'JMD95Z'/'JMD95P' (:963-1096). The initialisations :913-923 are
    overwritten in each ported branch; the negative-salinity WARNING (:932-939) clips s1 as the Fortran does, its
    message is not printed (no STDOUT in the port).
    """
    _no_factorized_eos(cfg, "FIND_RHO_SCALAR")
    eqs = eos.equationOfState
    dp0 = params.surf_pRef - eos.eosRefP0                               # :924
    t1 = tLoc                                                           # :926-929
    t2 = t1*t1
    t3 = t2*t1
    t4 = t3*t1
    s1 = sLoc                                                           # :931
    s1 = jnp.where(s1 < 0.0, 0.0, s1)                                   # :932-939  IF (s1 .LT. 0) s1 = 0. _d 0
    if eqs.rstrip() == "LINEAR":                                       # :941-947
        rhoLoc = params.rhoNil*(
            eos.sBeta*(sLoc-params.sRef[1])
            - eos.tAlpha*(tLoc-params.tRef[1])
        ) + params.rhoNil
    elif eqs.rstrip() == "POLY3":                                      # :949-961
        raise NotImplementedError("FIND_RHO_SCALAR: equationOfState = 'POLY3' is not ported")
    elif _is_jmd95_or_unesco(eqs):                                      # :963-1096
        # :967  s3o2 = s1*SQRT(s1); s1 >= 0 here. SQRT(+-0) = +-0, so the guard's fill on s1 <= 0 is s1 itself
        # (bitwise the IEEE square root of a zero) with a finite derivative.
        s3o2 = s1*safe_sqrt(s1, s1 > 0.0, fill=s1)
        p1 = (pLoc + dp0)*SItoBar                                       # :969
        p2 = p1*p1                                                      # :970
        Fw, Sw = eos.eosJMDCFw, eos.eosJMDCSw
        KFw, KSw, KP = eos.eosJMDCKFw, eos.eosJMDCKSw, eos.eosJMDCKP
        rfresh = (                                                      # :982-988
            Fw[1]
            + Fw[2]*t1
            + Fw[3]*t2
            + Fw[4]*t3
            + Fw[5]*t4
            + Fw[6]*t4*t1)
        rsalt = (                                                       # :1002-1015
            s1*(
                Sw[1]
                + Sw[2]*t1
                + Sw[3]*t2
                + Sw[4]*t3
                + Sw[5]*t4)
            + s3o2*(
                Sw[6]
                + Sw[7]*t1
                + Sw[8]*t2)
            + Sw[9]*s1*s1)
        rhoP0 = rfresh + rsalt                                          # :1018
        bMfresh = (                                                     # :1028-1033
            KFw[1]
            + KFw[2]*t1
            + KFw[3]*t2
            + KFw[4]*t3
            + KFw[5]*t4)
        bMsalt = (                                                      # :1061-1070
            s1*(KSw[1]
                + KSw[2]*t1
                + KSw[3]*t2
                + KSw[4]*t3)
            + s3o2*(KSw[5]
                    + KSw[6]*t1
                    + KSw[7]*t2))
        bMpres = (                                                      # :1072-1090
            p1*(KP[1]
                + KP[2]*t1
                + KP[3]*t2
                + KP[4]*t3)
            + p1*s1*(KP[5]
                     + KP[6]*t1
                     + KP[7]*t2)
            + p1*s3o2*KP[8]
            + p2*(KP[9]
                  + KP[10]*t1
                  + KP[11]*t2)
            + p2*s1*(KP[12]
                     + KP[13]*t1
                     + KP[14]*t2))
        bulkMod = bMfresh + bMsalt + bMpres                             # :1093
        rhoLoc = rhoP0/(1.0 - p1/bulkMod)                               # :1096
    elif eqs.rstrip() == "MDJWF":                                      # :1098-1127 (lane COLMIX, M3 vermix)
        # :1100  sp5 = SQRT(s1); s1 >= 0 here (clipped at :932-939). SQRT(+-0) = +-0: the guard's fill on s1 <= 0
        # is s1 itself, with a finite derivative (masked-lane rule, mitjax/ops/safe.py).
        sp5 = safe_sqrt(s1, s1 > 0.0, fill=s1)
        p1 = (pLoc + dp0)*SItodBar                                      # :1102
        p1t1 = p1*t1                                                    # :1103
        num, den_ = eos.eosMDJWFnum, eos.eosMDJWFden
        rhoNum = (num[0]                                                # :1105-1112
                  + t1*(num[1]
                        + t1*(num[2] + num[3]*t1))
                  + s1*(num[4]
                        + num[5]*t1 + num[6]*s1)
                  + p1*(num[7] + num[8]*t2
                        + num[9]*s1
                        + p1*(num[10] + num[11]*t2)))
        den = (den_[0]                                                  # :1114-1123
               + t1*(den_[1]
                     + t1*(den_[2]
                           + t1*(den_[3] + t1*den_[4])))
               + s1*(den_[5]
                     + t1*(den_[6]
                           + den_[7]*t2)
                     + sp5*(den_[8] + den_[9]*t2))
               + p1*(den_[10]
                     + p1t1*(den_[11]*t2 + den_[12]*p1)))
        rhoDen = 1.0/(epsln+den)                                        # :1125  (1.0 REAL*4, exact)
        rhoLoc = rhoNum*rhoDen                                          # :1127
    elif eqs.rstrip() in ("TEOS10", "IDEALG"):                         # :1129-1172
        raise NotImplementedError(f"FIND_RHO_SCALAR: equationOfState = '{eqs.rstrip()}' is not ported")
    else:                                                               # :1174-1180
        raise ValueError(f' FIND_RHO_SCALAR : equationOfState = "{eqs}"\nABNORMAL END: S/R FIND_RHO_SCALAR')
    return rhoLoc
