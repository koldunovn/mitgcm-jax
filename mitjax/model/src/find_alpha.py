"""FIND_ALPHA, FIND_BETA: model/src/find_alpha.F @63cdc0b (FIND_ALPHA called by CALC_OCE_MXLAYER; both by the KPP
lane's STATEKPP, pkg/kpp/kpp_routines.F:1859-1865, 1918-1924).

Ported arms: 'LINEAR' (lane COL; FIND_BETA's by lane KPP), 'MDJWF' (lane EOSAB, M3: vermix's eosType in input,
input.ggl90, .gglLC, .my82, .opps, .pp81) and 'JMD95Z' / 'JMD95P' (lane M4COL: 1D_ocean_ice_column/input's
eosType = 'JMD95Z', KPP's STATEKPP). The MDJWF arms call PRESSURE_FOR_EOS (mitjax/model/src/pressure_for_eos.py)
and lane COLMIX's FIND_RHONUM / FIND_RHODEN, the JMD95 arms PRESSURE_FOR_EOS and FIND_RHOP0 / FIND_BULKMOD
(mitjax/model/src/find_rho.py); both read theta, salt of level k from DYNVARS.h (`state`); 'LINEAR' needs neither,
so `cfg`, `grid`, `state` are required only by the MDJWF and JMD95 arms.
"""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.src.find_rho import SItoBar, SItodBar, find_bulkmod, find_rhoden, find_rhonum, find_rhop0
from mitjax.model.src.pressure_for_eos import pressure_for_eos
from mitjax.ops.libm import powi
from mitjax.ops.safe import safe_sqrt


def _level(fld, k):
    """`fld(1-OLx,1-OLy,k,bi,bj)` passed to a 2-D dummy (sequence association, KERNEL_GUIDE §4): level k of a
    DYNVARS.h field as an FArray declared (i, j); k a Python int or a traced level (KIdx)."""
    (_, ilo, ihi), (_, jlo, jhi), (_, klo, _) = fld.dims
    kk = k - klo
    return FArray(fld.data[:, getattr(kk, "value", kk)], fld.name, i=(ilo, ihi), j=(jlo, jhi))


def _need_common_blocks(routine, cfg, grid, state, eqs="MDJWF"):
    if cfg is None or grid is None or state is None:
        raise ValueError(f"{routine}: equationOfState = '{eqs}' reads theta, salt (DYNVARS.h) and PRESSURE_FOR_EOS's "
                         "fields: cfg=, grid=, state= are required")


def _is_jmd95_or_unesco(eqs):
    # equationOfState(1:5).EQ.'JMD95' .OR. equationOfState.EQ.'UNESCO'  (find_alpha.F:112-113, :443-444)
    return eqs[:5] == "JMD95" or eqs == "UNESCO"


def find_alpha(iMin, iMax, jMin, jMax, k, kRef, alphaLoc, *, params, eos, cfg=None, grid=None, state=None):
    """FIND_ALPHA( bi, bj, iMin, iMax, jMin, jMax, k, kRef, alphaLoc, myThid )
    @63cdc0b model/src/find_alpha.F:10-341

    C     *==========================================================*
    C     | o SUBROUTINE FIND_ALPHA
    C     |   Calculates [drho(S,T,z) / dT] of a horizontal slice
    C     *==========================================================*
    C     | k        - is the Theta/Salt level
    C     | kRef     - determines pressure reference level
    C     |            (not used in 'LINEAR' mode)
    C     | alphaLoc - drho / dT (kg/m^3/C)
    C     *==========================================================*

    Returns alphaLoc (points outside iMin:iMax, jMin:jMax keep their values). Ported: 'LINEAR' (:75-81; CALC_OCE_MXLAYER
    method 1 in tutorial_baroclinic_gyre), 'MDJWF' (:222-279, lane EOSAB; vermix: CALC_OCE_MXLAYER, KPP's
    STATEKPP) and 'JMD95Z' / 'JMD95P' (:112-221, lane M4COL; 1D_ocean_ice_column: KPP's STATEKPP; 'UNESCO', the
    other name of the same ELSEIF, raises: FIND_BULKMOD's UNESCO arm is not ported). JMD95: as the MDJWF arm, with
    FIND_RHOP0 / FIND_BULKMOD; `x**2` is libgcc's __powidf2 (`powi`, gfortran -O0), REAL*4 literals 2., 3., 4.,
    5. exact; SQRT(s1*s1*s1) only where s1 > 0 (as MDJWF's SQRT); the division by (bulkmod - p1)**2 is the
    Fortran's (bulkmod >> p1 on every lane FIND_BULKMOD writes, halos included). MDJWF: theta, salt of level k
    from `state` (DYNVARS.h), PRESSURE_FOR_EOS at kRef (`cfg`, `grid`, `state.totPhiHyd`), FIND_RHONUM /
    FIND_RHODEN of lane COLMIX; k and kRef are Python ints or traced levels (KIdx, inside a caller's
    scan_levels). Pointwise: vectorised over (i, j). `IF ( s1 .GT. 0. _d 0 )` (:251-256) is a pointwise where; SQRT
    only where s1 > 0 (masked-lane rule, mitjax/ops/safe.py: the s1 <= 0 lanes get sp5 = 0 and s1 = 0 with a zero
    derivative, as the Fortran's ELSE); the only division, 1/den, is FIND_RHODEN's. The other branches ('POLY3',
    'UNESCO', 'TEOS10') raise.
    """
    eqs = eos.equationOfState.rstrip()
    if eqs == "LINEAR":                                                 # :75-81
        j = loop_j(jMin, jMax)
        i = loop_i(iMin, iMax)
        return alphaLoc.at[i, j].set(-params.rhoNil * eos.tAlpha)
    if eqs == "MDJWF":                                                  # :222-279 (lane EOSAB, M3 vermix)
        _need_common_blocks("FIND_ALPHA", cfg, grid, state)
        locPres = alphaLoc.local("locPres")
        rhoLoc = alphaLoc.local("rhoLoc")
        rhoDen = alphaLoc.local("rhoDen")
        dp0 = params.surf_pRef - eos.eosRefP0                           # :223
        locPres = pressure_for_eos(iMin, iMax, jMin, jMax, kRef, dp0, locPres,      # :225-228
                                   cfg=cfg, grid=grid, params=params, state=state)
        rhoLoc = find_rhonum(iMin, iMax, jMin, jMax, locPres,                       # :230-234
                             _level(state.theta, k), _level(state.salt, k), rhoLoc, eos=eos)
        rhoDen = find_rhoden(iMin, iMax, jMin, jMax, locPres,                       # :236-240
                             _level(state.theta, k), _level(state.salt, k), rhoDen, eos=eos)
        num, den_ = eos.eosMDJWFnum, eos.eosMDJWFden
        j = loop_j(jMin, jMax)                                          # :242-243
        i = loop_i(iMin, iMax)
        t1 = state.theta[i, j, k]                                       # :244
        t2 = t1*t1                                                      # :245
        s1 = state.salt[i, j, k]                                        # :246
        # :247-250  #if (defined ALLOW_AUTODIFF && defined TARGET_NEC_SX): no build defines TARGET_NEC_SX
        pos = s1 > 0.0                                                  # :251  IF ( s1 .GT. 0. _d 0 )
        sp5 = safe_sqrt(s1, pos)                                        # :252 SQRT(s1) / :255 0. _d 0
        s1 = jnp.where(pos, s1, 0.0)                                    # :254  s1 = 0. _d 0
        p1 = locPres[i, j]*SItodBar                                     # :258
        p1t1 = p1*t1                                                    # :259
        dnum_dtheta = (num[1]                                           # :261-264
                       + t1*(2.0*num[2] + 3.0*num[3]*t1)
                       + num[5]*s1
                       + p1t1*(2.0*num[8] + 2.0*num[11]*p1))
        dden_dtheta = (den_[1]                                          # :266-273
                       + t1*(2.0*den_[2]
                             + t1*(3.0*den_[3]
                                   + 4.0*den_[4]*t1))
                       + s1*(den_[6]
                             + t1*(3.0*den_[7]*t1
                                   + 2.0*den_[9]*sp5))
                       + p1*p1*(3.0*den_[11]*t2 + den_[12]*p1))
        return alphaLoc.at[i, j].set(rhoDen[i, j]*(dnum_dtheta          # :275-276
                                                   - (rhoLoc[i, j]*rhoDen[i, j])*dden_dtheta))
    if _is_jmd95_or_unesco(eqs) and eqs != "UNESCO":                    # :112-221 (lane M4COL: 'JMD95Z')
        _need_common_blocks("FIND_ALPHA", cfg, grid, state, eqs)
        locPres = alphaLoc.local("locPres")
        rhoP0 = alphaLoc.local("rhoP0")
        bulkMod = alphaLoc.local("bulkMod")
        dp0 = params.surf_pRef - eos.eosRefP0                           # :115
        locPres = pressure_for_eos(iMin, iMax, jMin, jMax, kRef, dp0, locPres,      # :117-120
                                   cfg=cfg, grid=grid, params=params, state=state)
        rhoP0 = find_rhop0(iMin, iMax, jMin, jMax,                                  # :122-126
                           _level(state.theta, k), _level(state.salt, k), rhoP0, cfg=cfg, eos=eos)
        bulkMod = find_bulkmod(iMin, iMax, jMin, jMax, locPres,                     # :128-132
                               _level(state.theta, k), _level(state.salt, k), bulkMod, cfg=cfg, eos=eos)
        Fw, Sw, KFw, KSw, KP = eos.eosJMDCFw, eos.eosJMDCSw, eos.eosJMDCKFw, eos.eosJMDCKSw, eos.eosJMDCKP
        j = loop_j(jMin, jMax)                                          # :134-135
        i = loop_i(iMin, iMax)
        t1 = state.theta[i, j, k]                                       # :138
        t2 = t1*t1                                                      # :139
        t3 = t2*t1                                                      # :140
        # :142-145  #if (defined ALLOW_AUTODIFF && defined TARGET_NEC_SX): no build defines TARGET_NEC_SX
        s1 = state.salt[i, j, k]                                        # :146
        pos = s1 > 0.0                                                  # :147  IF ( s1 .GT. 0. _d 0 )
        s3o2 = safe_sqrt(s1*s1*s1, pos)                                 # :148 SQRT(s1*s1*s1) / :151 0. _d 0
        s1 = jnp.where(pos, s1, 0.0)                                    # :150  s1 = 0. _d 0
        p1 = locPres[i, j]*SItoBar                                      # :154
        p2 = p1*p1                                                      # :155
        drhoP0dthetaFresh = (Fw[2]                                      # :159-164
                             + 2.*Fw[3]*t1
                             + 3.*Fw[4]*t2
                             + 4.*Fw[5]*t3
                             + 5.*Fw[6]*t3*t1)
        drhoP0dthetaSalt = (s1*(Sw[2]                                   # :166-176
                                + 2.*Sw[3]*t1
                                + 3.*Sw[4]*t2
                                + 4.*Sw[5]*t3)
                            + s3o2*(+Sw[7]
                                    + 2.*Sw[8]*t1))
        dKdthetaFresh = (KFw[2]                                         # :179-183
                         + 2.*KFw[3]*t1
                         + 3.*KFw[4]*t2
                         + 4.*KFw[5]*t3)
        dKdthetaSalt = (s1*(KSw[2]                                      # :185-192
                            + 2.*KSw[3]*t1
                            + 3.*KSw[4]*t2)
                        + s3o2*(KSw[6]
                                + 2.*KSw[7]*t1))
        dKdthetaPres = (p1*(KP[2]                                       # :194-207
                            + 2.*KP[3]*t1
                            + 3.*KP[4]*t2)
                        + p1*s1*(KP[6]
                                 + 2.*KP[7]*t1)
                        + p2*(KP[10]
                              + 2.*KP[11]*t1)
                        + p2*s1*(KP[13]
                                 + 2.*KP[14]*t1))
        drhoP0dtheta = (drhoP0dthetaFresh                               # :209-210
                        + drhoP0dthetaSalt)
        dKdtheta = (dKdthetaFresh                                       # :211-213
                    + dKdthetaSalt
                    + dKdthetaPres)
        bm = bulkMod[i, j]
        return alphaLoc.at[i, j].set(                                   # :214-218
            (powi(bm, 2)*drhoP0dtheta
             - bm*p1*drhoP0dtheta
             - rhoP0[i, j]*p1*dKdtheta)
            / powi(bm - p1, 2))
    if eqs in ("POLY3", "UNESCO", "TEOS10"):
        raise NotImplementedError(f"FIND_ALPHA: equationOfState = '{eqs}' is not ported")
    raise ValueError('FIND_ALPHA: "equationOfState" has illegal value')    # :335-337


def find_beta(iMin, iMax, jMin, jMax, k, kRef, betaLoc, *, params, eos, cfg=None, grid=None, state=None):
    """FIND_BETA( bi, bj, iMin, iMax, jMin, jMax, k, kRef, betaLoc, myThid )
    @63cdc0b model/src/find_alpha.F:347-645

    C     | o SUBROUTINE FIND_BETA
    C     |   Calculates [drho(S,T,z) / dS] of a horizontal slice
    C     | k        - is the Theta/Salt level
    C     | kRef     - determines pressure reference level
    C     |            (not used in 'LINEAR' mode)
    C     | betaLoc  - drho / dS (kg/m^3/PSU)

    KPP lane (M3, marked arm; STATEKPP calls it). Returns betaLoc (points outside iMin:iMax, jMin:jMax keep their
    values). Ported: 'LINEAR' (:408-414, `betaLoc(i,j) = rhonil * sBeta`, vermix/input.dd) and 'MDJWF' (:539-589,
    lane EOSAB; vermix/input: STATEKPP), written as FIND_ALPHA's MDJWF arm (same arguments, same masked-lane rule;
    `p1t1` (:576) is computed and not used, as in the Fortran), and 'JMD95Z' / 'JMD95P' (:443-538, lane M4COL,
    written as FIND_ALPHA's JMD95 arm; `2*eosJMDCSw(9)` an INTEGER 2, exact; `0. _d 0 + eosJMDCSw(1)` kept); the
    other equations of state raise."""
    eqs = eos.equationOfState.rstrip()
    if eqs == "LINEAR":                                                 # :408-414
        j = loop_j(jMin, jMax)
        i = loop_i(iMin, iMax)
        return betaLoc.at[i, j].set(params.rhoNil * eos.sBeta)
    if eqs == "MDJWF":                                                  # :539-589 (lane EOSAB, M3 vermix)
        _need_common_blocks("FIND_BETA", cfg, grid, state)
        locPres = betaLoc.local("locPres")
        rhoLoc = betaLoc.local("rhoLoc")
        rhoDen = betaLoc.local("rhoDen")
        dp0 = params.surf_pRef - eos.eosRefP0                           # :540
        locPres = pressure_for_eos(iMin, iMax, jMin, jMax, kRef, dp0, locPres,      # :542-545
                                   cfg=cfg, grid=grid, params=params, state=state)
        rhoLoc = find_rhonum(iMin, iMax, jMin, jMax, locPres,                       # :547-551
                             _level(state.theta, k), _level(state.salt, k), rhoLoc, eos=eos)
        rhoDen = find_rhoden(iMin, iMax, jMin, jMax, locPres,                       # :553-557
                             _level(state.theta, k), _level(state.salt, k), rhoDen, eos=eos)
        num, den_ = eos.eosMDJWFnum, eos.eosMDJWFden
        j = loop_j(jMin, jMax)                                          # :559-560
        i = loop_i(iMin, iMax)
        t1 = state.theta[i, j, k]                                       # :561
        t2 = t1*t1                                                      # :562
        s1 = state.salt[i, j, k]                                        # :563
        # :564-567  #if (defined ALLOW_AUTODIFF && defined TARGET_NEC_SX): no build defines TARGET_NEC_SX
        pos = s1 > 0.0                                                  # :568  IF ( s1 .GT. 0. _d 0 )
        sp5 = safe_sqrt(s1, pos)                                        # :569 SQRT(s1) / :572 0. _d 0
        s1 = jnp.where(pos, s1, 0.0)                                    # :571  s1 = 0. _d 0
        p1 = locPres[i, j]*SItodBar                                     # :575
        p1t1 = p1*t1                                                    # :576
        dnum_dsalt = (num[4]                                            # :578-580
                      + num[5]*t1
                      + 2.0*num[6]*s1 + num[9]*p1)
        dden_dsalt = (den_[5]                                           # :581-583
                      + t1*(den_[6] + den_[7]*t2)
                      + 1.5*sp5*(den_[8] + den_[9]*t2))
        return betaLoc.at[i, j].set(rhoDen[i, j]*(dnum_dsalt            # :585-586
                                                  - (rhoLoc[i, j]*rhoDen[i, j])*dden_dsalt))
    if _is_jmd95_or_unesco(eqs) and eqs != "UNESCO":                    # :443-538 (lane M4COL: 'JMD95Z')
        _need_common_blocks("FIND_BETA", cfg, grid, state, eqs)
        locPres = betaLoc.local("locPres")
        rhoP0 = betaLoc.local("rhoP0")
        bulkMod = betaLoc.local("bulkMod")
        dp0 = params.surf_pRef - eos.eosRefP0                           # :446
        locPres = pressure_for_eos(iMin, iMax, jMin, jMax, kRef, dp0, locPres,      # :448-451
                                   cfg=cfg, grid=grid, params=params, state=state)
        rhoP0 = find_rhop0(iMin, iMax, jMin, jMax,                                  # :453-457
                           _level(state.theta, k), _level(state.salt, k), rhoP0, cfg=cfg, eos=eos)
        bulkMod = find_bulkmod(iMin, iMax, jMin, jMax, locPres,                     # :459-463
                               _level(state.theta, k), _level(state.salt, k), bulkMod, cfg=cfg, eos=eos)
        Sw, KSw, KP = eos.eosJMDCSw, eos.eosJMDCKSw, eos.eosJMDCKP
        j = loop_j(jMin, jMax)                                          # :465-466
        i = loop_i(iMin, iMax)
        t1 = state.theta[i, j, k]                                       # :469
        t2 = t1*t1                                                      # :470
        t3 = t2*t1                                                      # :471
        s1 = state.salt[i, j, k]                                        # :473
        # :474-477  #if (defined ALLOW_AUTODIFF && defined TARGET_NEC_SX): no build defines TARGET_NEC_SX
        pos = s1 > 0.0                                                  # :478  IF ( s1 .GT. 0. _d 0 )
        s3o2 = 1.5*safe_sqrt(s1, pos)                                   # :479 1.5*SQRT(s1) / :482 0. _d 0
        s1 = jnp.where(pos, s1, 0.0)                                    # :481  s1 = 0. _d 0
        p1 = locPres[i, j]*SItoBar                                      # :485
        drhoP0dS = 0.0                                                  # :489  0. _d 0
        drhoP0dS = (drhoP0dS                                            # :491-502
                    + Sw[1]
                    + Sw[2]*t1
                    + Sw[3]*t2
                    + Sw[4]*t3
                    + Sw[5]*t3*t1
                    + s3o2*(Sw[6]
                            + Sw[7]*t1
                            + Sw[8]*t2)
                    + 2*Sw[9]*s1)
        # :505  dKdS = 0. _d 0 (overwritten at :529 before any read)
        dKdSSalt = (KSw[1]                                              # :507-515
                    + KSw[2]*t1
                    + KSw[3]*t2
                    + KSw[4]*t3
                    + s3o2*(KSw[5]
                            + KSw[6]*t1
                            + KSw[7]*t2))
        dKdSPres = (p1*(KP[5]                                           # :518-527
                        + KP[6]*t1
                        + KP[7]*t2)
                    + s3o2*p1*KP[8]
                    + p1*p1*(KP[12]
                             + KP[13]*t1
                             + KP[14]*t2))
        dKdS = dKdSSalt + dKdSPres                                      # :529
        bm = bulkMod[i, j]
        return betaLoc.at[i, j].set(                                    # :531-535
            (powi(bm, 2)*drhoP0dS
             - bm*p1*drhoP0dS
             - rhoP0[i, j]*p1*dKdS)
            / powi(bm - p1, 2))
    if eqs in ("POLY3", "UNESCO", "TEOS10"):
        raise NotImplementedError(f"FIND_BETA: equationOfState = '{eqs}' is not ported")
    raise ValueError('FIND_BETA: "equationOfState" has illegal value')    # :639-642
