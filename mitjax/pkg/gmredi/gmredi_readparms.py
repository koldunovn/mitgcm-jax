"""GMREDI_READPARMS: pkg/gmredi/gmredi_readparms.F @63cdc0b."""

import numpy as np

from mitjax.model.grid import UNSET_RL, zeroRL
from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.gmredi.gmredi_h import Gmredi

_RP = "pkg/gmredi/gmredi_readparms.F"
_F = "data.gmredi"
_G = "GM_PARM01"

# (name, gmredi_readparms.F line of its default): the namelist variables the ported code reads
_DEFAULTS = (("GM_AdvForm", 110), ("GM_AdvSeparate", 111), ("GM_InMomAsStress", 112), ("GM_isopycK", 113),
             ("GM_background_K", 114), ("GM_maxSlope", 115), ("GM_Kmin_horiz", 116), ("GM_Small_Number", 117),
             ("GM_slopeSqCutoff", 118), ("GM_taper_scheme", 119), ("GM_Scrit", 123), ("GM_Sd", 124),
             ("GM_isoFac_calcK", 125), ("GM_iso2dFile", 127), ("GM_iso1dFile", 128), ("GM_bol2dFile", 129),
             ("GM_bol1dFile", 130), ("GM_K3dRediFile", 131), ("GM_K3dGMFile", 132), ("GM_useLeithQG", 133),
             ("GM_UseBVP", 139), ("GM_useSubMeso", 144), ("GM_Visbeck_alpha", 151), ("GM_useGEOM", 160),
             ("GM_useBatesK3d", 172))
# the retired namelist variables (:199-200 set them to ' '; :288-316 stop if the file sets them)
_RETIRED = (("GM_background_K3dFile", "GM_K3dGMFile"), ("GM_isopycK3dFile", "GM_K3dRediFile"))


def _blank(s):
    return str(s).strip() == ""


def _kap_ctrl(cfg, name):
    """GOADK lane: #ifdef ALLOW_KAPREDI_CONTROL / ALLOW_KAPGM_CONTROL as gmredi_readparms.F sees it (it includes
    CTRL_OPTIONS.h only #ifdef ALLOW_CTRL, :2-4)."""
    return bool(cfg.cpp.ALLOW_CTRL) and cfg.cpp.flag(name, "CTRL_OPTIONS.h")


def gmredi_readparms(exp):
    """GMREDI_READPARMS( myThid )   @63cdc0b pkg/gmredi/gmredi_readparms.F:9-326

    C     | SUBROUTINE GMREDI_READPARMS
    C     | o Routine to initialize GM/Redi variables and constants.
    C     | Initialize GM/Redi parameters, read in data.gmredi

    `exp`: mitjax.config.params.Experiment. Returns a `Gmredi` (GMREDI.h) holding the parameters, or None when
    useGMRedi is .FALSE. (:97-105: the routine returns after a warning).

    The namelist values come from mitjax/config (the run's data.gmredi, read as the model reads it); a variable the
    file does not set takes its default of :110-172, read from that line by `fortran_default` (REAL*4 literals as
    binary32: `GM_isopycK = -999.` is exact). Ported here are the settings derived after the READ (:222-277), each
    cited, computed once on the host in float64 (no jit: a setup-time IF on float values gives static logicals;
    `1. _d 0 / GM_maxSlope` is the one IEEE division of numpy, as in the -O0 oracle).

    GM_AdvForm = .TRUE. (GOADK lane, global_ocean.90x40x15/code_ad: GM_skewflx = 0, :242-251) and the
    `IF ( useCTRL ) GM_ExtraDiag = .TRUE.` arms of ALLOW_KAPREDI_CONTROL / ALLOW_KAPGM_CONTROL (:249-251, :263-265,
    as CTRL_OPTIONS.h, which the routine includes, defines them). M3 lane MLAdjust: GM_useLeithQG switched off when
    viscC2LeithQG = 0 (:234-240). Not ported (raise): GM_UseBVP (:274-277). Output
    switches GM_MNC / GM_MDSIO (:280-285) are not carried: they only select output.
    """
    cfg = exp.cfg
    if not cfg.use_flag("useGMRedi"):                              # :97-105
        return None
    rp = RunParams(exp.run)
    v = {}
    for name, line in _DEFAULTS:                                   # :110-172 defaults, :211 READ
        v[name] = rp.get(_F, _G, name, default=fortran_default(f"{_RP}:{line}", name, exp))
    # :155  GM_Visbeck_maxSlope = UNSET_RL (EEPARAMS.h:81 PARAMETER ( UNSET_RL = 1.234567D5 ))
    v["GM_Visbeck_maxSlope"] = (rp.get(_F, _G, "GM_Visbeck_maxSlope")
                                if rp.has(_F, _G, "GM_Visbeck_maxSlope") else UNSET_RL)
    for old, new in _RETIRED:                                      # :288-316
        if rp.has(_F, _G, old) and not _blank(rp.get(_F, _G, old)):
            raise RuntimeError(f'S/R GMREDI_READPARMS: "{old}" has been replaced by "{new}" and is no longer '
                               "allowed in file \"data.gmredi\"; ABNORMAL END: S/R GMREDI_READPARMS")
    f = {k: (np.float64(x) if isinstance(x, float) else x) for k, x in v.items()}

    # :223  IF (GM_isopycK.EQ.-999.) GM_isopycK = GM_background_K   (-999. is a REAL*4 literal, exact)
    if f["GM_isopycK"] == -999.0:
        f["GM_isopycK"] = f["GM_background_K"]
    # :226-227
    if f["GM_Visbeck_maxSlope"] == UNSET_RL:
        f["GM_Visbeck_maxSlope"] = f["GM_maxSlope"]
    # :230-231
    f["GM_rMaxSlope"] = np.float64(0.0)                            # 0. _d 0
    if f["GM_maxSlope"] != zeroRL:
        f["GM_rMaxSlope"] = np.float64(1.0) / f["GM_maxSlope"]    # 1. _d 0 / GM_maxSlope
    # :234-240
    # M3 lane MLAdjust (input.QGLthGM): viscC2LeithQG (PARAMS.h) is the run's PARM01 value, else its default
    # (model/src/set_defaults.F:128 viscC2LeithQG = 0.D0)
    viscC2LeithQG = np.float64(rp.get("data", "PARM01", "viscC2LeithQG",
                                      default=fortran_default("model/src/set_defaults.F:128", "viscC2LeithQG", exp)))
    if f["GM_useLeithQG"] and viscC2LeithQG == zeroRL:
        # :235-238 PRINT_MESSAGE '** WARNING ** ... switch OFF GM_useLeithQG since viscC2LeithQG = 0' (log only)
        f["GM_useLeithQG"] = False                                 # :239
    # :242-267
    if f["GM_AdvForm"]:
        # GOADK lane (M2, global_ocean.90x40x15/code_ad): the advective (bolus) form, :242-251
        f["GM_skewflx"] = np.float64(0.0)                          # :243  0. _d 0
        if not _blank(f["GM_K3dRediFile"]):                        # :244-248
            ExtraDiag = True
        else:
            ExtraDiag = bool(f["GM_isopycK"] != zeroRL)
        if _kap_ctrl(cfg, "ALLOW_KAPREDI_CONTROL") and cfg.use_flag("useCTRL"):   # :249-251
            ExtraDiag = True
    else:
        f["GM_skewflx"] = np.float64(1.0)                          # :253  1. _d 0
        if not _blank(f["GM_K3dRediFile"]) or not _blank(f["GM_K3dGMFile"]):          # :254-256
            ExtraDiag = f["GM_K3dRediFile"].rstrip() != f["GM_K3dGMFile"].rstrip()
        elif (f["GM_iso2dFile"].rstrip() != f["GM_bol2dFile"].rstrip()
              or f["GM_iso1dFile"].rstrip() != f["GM_bol1dFile"].rstrip()):           # :257-259
            ExtraDiag = True
        else:                                                                          # :260-261
            ExtraDiag = bool(f["GM_isopycK"] != f["GM_background_K"])
        if ((_kap_ctrl(cfg, "ALLOW_KAPREDI_CONTROL")                                  # :263-265 (GOADK lane)
             or _kap_ctrl(cfg, "ALLOW_KAPGM_CONTROL")) and cfg.use_flag("useCTRL")):
            ExtraDiag = True
        ExtraDiag = ExtraDiag or f["GM_useGEOM"]                                       # :266
    # :268-272
    if f["GM_isoFac_calcK"] != f["GM_skewflx"] and (f["GM_Visbeck_alpha"] != zeroRL or f["GM_useBatesK3d"]
                                                   or f["GM_useLeithQG"]):
        ExtraDiag = True
    f["GM_ExtraDiag"] = bool(ExtraDiag)
    # :274-277
    if f["GM_UseBVP"]:
        raise NotImplementedError("GMREDI_READPARMS: GM_UseBVP = .TRUE. is not ported")
    f["GM_useBVP"] = f.pop("GM_UseBVP")                            # GMREDI.h:44 spells it GM_useBVP
    return Gmredi(f)
