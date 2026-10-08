"""INI_FORCING: model/src/ini_forcing.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_UV_XY_RS, EXCH_XY_RS
from mitjax.farray import loop_i, loop_j


def READ_FLD_XY_RS(pref, suff, field, myIter, *, rw):
    """READ_FLD_XY_RS( pref, suff, field, myIter, myThid )   @63cdc0b pkg/rw/read_fld_xy_rs.F

    fullName = pref [// suff] (blanks trimmed, suff ignored if blank); MDS_READ_FIELD( fullName, readBinaryPrec,
    .FALSE., 'RS', nNz=1, 1, 1, dummyRL, field, iRec=1, myThid ): the interior of `field` from record 1, halos kept.
    The 'RS' twin of READ_FLD_XY_RL (mitjax/pkg/rw/read_rec.py, core lane): with _RS = Real*8 (REAL4_IS_SLOW, every
    M1 build) the two read the same values into the same storage. Kept here until pkg/rw carries it (core lane)."""
    full = str(pref).strip() + ("" if str(suff).strip() == "" else str(suff).strip())
    return rw.mds_read_field(full, rw.readBinaryPrec, 1, field, 1)


def ini_forcing(ff, *, cfg, grid, fp, rw, ex, latBandClimRelax):
    """INI_FORCING( myThid )   @63cdc0b model/src/ini_forcing.F:7-253

    C     | SUBROUTINE INI_FORCING
    C     | o Set model initial forcing fields.

    `ff` FFIELDS.h as INI_FFIELDS left it; `latBandClimRelax` (PARAMS.h, set by INI_GRID: mitjax ini_grid returns
    it). Returns the new FFields.

    Ported: lambdaThetaClimRelax / lambdaSaltClimRelax (:46-65), the PARM05 files read with READ_FLD_XY_RS (:71-113;
    EmPmR scaled by rhoConstFresh :82-96), ATMOSPHERIC_LOADING pLoadFile (:191-195), the exchanges :233-240 and
    :245-248 (pLoad). PTRACERS lane: SHORTWAVE_HEATING without surfQswFile (:130-188, SWFrac3D with SWFRAC;
    `_swfrac3d`); surfQswFile raises. ALLOW_GEOTHERMAL_FLUX raises (FFIELDS.h fields_of). Lane B (Task 25,
    global_ocean.cs32x15): ALLOW_ADDFLUID (:196-201: addMassFile raises), ALLOW_BALANCE_FLUXES (:213-231:
    selectBalanceEmPmR = 2 sets weight2BalanceFlx = oneRS; wghtBalanceFile raises)."""
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    yC = grid.yC[i, j]
    # :50-55
    if fp.doThetaClimRelax:
        lt = jnp.where(jnp.abs(yC) <= latBandClimRelax, 1.0/fp.tauThetaClimRelax, 0.)
    else:
        lt = jnp.zeros_like(yC)
    # :56-61
    if fp.doSaltClimRelax:
        ls = jnp.where(jnp.abs(yC) <= latBandClimRelax, 1.0/fp.tauSaltClimRelax, 0.)
    else:
        ls = jnp.zeros_like(yC)
    g = dict((n, getattr(ff, n)) for n in ff.names())
    g["lambdaThetaClimRelax"] = g["lambdaThetaClimRelax"].at[i, j].set(lt)
    g["lambdaSaltClimRelax"] = g["lambdaSaltClimRelax"].at[i, j].set(ls)

    if fp.zonalWindFile.strip():                                               # :71-73
        g["fu"] = READ_FLD_XY_RS(fp.zonalWindFile, " ", g["fu"], 0, rw=rw)
    if fp.meridWindFile.strip():                                               # :74-76
        g["fv"] = READ_FLD_XY_RS(fp.meridWindFile, " ", g["fv"], 0, rw=rw)
    if fp.surfQFile.strip():                                                   # :77-81
        g["Qnet"] = READ_FLD_XY_RS(fp.surfQFile, " ", g["Qnet"], 0, rw=rw)
    elif fp.surfQnetFile.strip():
        g["Qnet"] = READ_FLD_XY_RS(fp.surfQnetFile, " ", g["Qnet"], 0, rw=rw)
    if fp.EmPmRfile.strip():                                                   # :82-96
        g["EmPmR"] = READ_FLD_XY_RS(fp.EmPmRfile, " ", g["EmPmR"], 0, rw=rw)
        g["EmPmR"] = g["EmPmR"].at[i, j].set(g["EmPmR"][i, j]*fp.rhoConstFresh)        # :90
    if fp.saltFluxFile.strip():                                                # :97-99
        g["saltFlux"] = READ_FLD_XY_RS(fp.saltFluxFile, " ", g["saltFlux"], 0, rw=rw)
    if fp.thetaClimFile.strip():                                               # :100-102
        g["SST"] = READ_FLD_XY_RS(fp.thetaClimFile, " ", g["SST"], 0, rw=rw)
    if fp.saltClimFile.strip():                                                # :103-105
        g["SSS"] = READ_FLD_XY_RS(fp.saltClimFile, " ", g["SSS"], 0, rw=rw)
    if fp.lambdaThetaFile.strip():                                             # :106-109
        g["lambdaThetaClimRelax"] = READ_FLD_XY_RS(fp.lambdaThetaFile, " ", g["lambdaThetaClimRelax"], 0, rw=rw)
    if fp.lambdaSaltFile.strip():                                              # :110-113
        g["lambdaSaltClimRelax"] = READ_FLD_XY_RS(fp.lambdaSaltFile, " ", g["lambdaSaltClimRelax"], 0, rw=rw)
    if cfg.cpp.SHORTWAVE_HEATING:                                              # :114-190 (PTRACERS lane)
        if fp.surfQswFile.strip():                                             # :115-129
            raise NotImplementedError("INI_FORCING: surfQswFile (Qsw read, Qnet + Qsw) is not ported")
        g["SWFrac3D"] = _swfrac3d(g["SWFrac3D"], cfg=cfg, grid=grid, fp=fp)
    if cfg.cpp.ATMOSPHERIC_LOADING and fp.pLoadFile.strip():                  # :191-195
        g["pLoad"] = READ_FLD_XY_RS(fp.pLoadFile, " ", g["pLoad"], 0, rw=rw)
    if cfg.cpp.ALLOW_ADDFLUID and fp.addMassFile.strip():                      # :196-201 (lane B)
        raise NotImplementedError("INI_FORCING: addMassFile (READ_FLD_XYZ_RL, :197-200) is not ported")
    if cfg.cpp.ALLOW_BALANCE_FLUXES:                                            # :213-231 (lane B)
        if fp.selectBalanceEmPmR == 2:                                          # :214-224
            g["weight2BalanceFlx"] = g["weight2BalanceFlx"].at[i, j].set(1.)   # :220  oneRS
        if fp.wghtBalanceFile.strip():                                          # :226-230
            raise NotImplementedError("INI_FORCING: wghtBalanceFile (:226-230) is not ported")

    g["fu"], g["fv"] = EXCH_UV_XY_RS(g["fu"], g["fv"], True, ex=ex)            # :233
    g["Qnet"] = EXCH_XY_RS(g["Qnet"], ex=ex)                                    # :234
    g["EmPmR"] = EXCH_XY_RS(g["EmPmR"], ex=ex)                                  # :235
    g["saltFlux"] = EXCH_XY_RS(g["saltFlux"], ex=ex)                            # :236
    g["SST"] = EXCH_XY_RS(g["SST"], ex=ex)                                      # :237
    g["SSS"] = EXCH_XY_RS(g["SSS"], ex=ex)                                      # :238
    g["lambdaThetaClimRelax"] = EXCH_XY_RS(g["lambdaThetaClimRelax"], ex=ex)    # :239
    g["lambdaSaltClimRelax"] = EXCH_XY_RS(g["lambdaSaltClimRelax"], ex=ex)      # :240
    # :241-244 SHORTWAVE_HEATING: EXCH_XY_RS( Qsw ) only with surfQswFile (raises above)
    if cfg.cpp.ATMOSPHERIC_LOADING:                                             # :245-248
        g["pLoad"] = EXCH_XY_RS(g["pLoad"], ex=ex)                              # :246
    return ff.replace(**{k: v for k, v in g.items() if v is not getattr(ff, k)})


def _swfrac3d(SWFrac3D, *, cfg, grid, fp):
    """INI_FORCING's SWFrac3D (ini_forcing.F:130-188, SHORTWAVE_HEATING; PTRACERS lane, tutorial_tracer_adjsens).
    :130-145 the no-penetration profile: swfac = 0. _d 0, 1. _d 0 at k = 1 (usingZCoords) or k = Nr+1, on every point
    of every level k = 1..Nr+1. With selectPenetratingSW > 0 (:146-187): SWFracK(k) = rF(k) - rF(1) (usingZCoords,
    :149-150; the pressure-coordinate arm :152-155 raises), SWFRAC( Nr+1, oneRL, SWFracK, zeroRL, 0 ) (:160-163),
    then SWFrac3D(i,j,k) = SWFracK(k)*swfac*maskC(i,j,km) (left to right) with km = MIN(k,Nr) and swfac = 0. _d 0
    at k = Nr+1, else 1. _d 0 (:166-176; MIN on loop integers: # MINMAX-INT). SWFracK is a local vector (Nr+1)."""
    from mitjax.model.src.swfrac import swfrac
    sz = cfg.size
    Nr = sz.Nr
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    if not fp.usingZCoords:                                                     # :135-138, :151-155
        raise NotImplementedError("INI_FORCING: SWFrac3D in pressure coordinates (ini_forcing.F:137-138, :151-155) "
                                  "is not ported")
    for k in range(1, Nr + 2):                                                  # :132-145
        swfac = 0.0                                                             # :133  0. _d 0
        if k == 1:                                                              # :135
            swfac = 1.0                                                         # 1. _d 0
        SWFrac3D = SWFrac3D.at[i, j, k].set(jnp.full_like(grid.maskC[i, j, 1], swfac))   # :139-143
    if fp.selectPenetratingSW > 0:                                              # :146
        rF = jnp.asarray(grid.rF.data)                                          # rF(1..Nr+1)
        SWFracK = rF - rF[0]                                                    # :149-150  rF(k) - rF(1)
        SWFracK = swfrac(Nr + 1, 1.0, SWFracK, 0.0, 0)                          # :160-163  oneRL, zeroRL
        for k in range(1, Nr + 2):                                              # :166-176
            swfac = 1.0                                                         # :167  1. _d 0
            km = min(k, Nr)    # :170  MINMAX-INT: loop integers (no tie or NaN case)
            if k == Nr + 1:                                                     # :171
                swfac = 0.0                                                     # 0. _d 0
            SWFrac3D = SWFrac3D.at[i, j, k].set(SWFracK[k - 1]*swfac
                                                * grid.maskC[i, j, km])         # :178-179
    return SWFrac3D
