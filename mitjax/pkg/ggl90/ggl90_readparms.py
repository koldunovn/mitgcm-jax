"""GGL90_READPARMS: pkg/ggl90/ggl90_readparms.F @63cdc0b."""

import numpy as np

from mitjax.config.fortran import real4
from mitjax.model.grid import UNSET_RL
from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.ggl90.ggl90_h import Ggl90

_RP = "pkg/ggl90/ggl90_readparms.F"
_F = "data.ggl90"
UNSET_I = 123456789          # EEPARAMS.h:85  PARAMETER ( UNSET_I = 123456789 )

# (namelist group, name, ggl90_readparms.F line of its literal default): the variables the ported code reads
_DEFAULTS = (("GGL90_PARM01", "GGL90writeState", 107), ("GGL90_PARM01", "GGL90ck", 108),
             ("GGL90_PARM01", "GGL90ceps", 109), ("GGL90_PARM01", "GGL90alpha", 110),
             ("GGL90_PARM01", "GGL90m2", 112), ("GGL90_PARM01", "GGL90TKEmin", 113),
             ("GGL90_PARM01", "GGL90TKEsurfMin", 115), ("GGL90_PARM01", "GGL90viscMax", 117),
             ("GGL90_PARM01", "GGL90diffMax", 118), ("GGL90_PARM01", "GGL90diffTKEh", 119),
             ("GGL90_PARM01", "GGL90mixingLengthMin", 120), ("GGL90_PARM01", "mxlMaxFlag", 121),
             ("GGL90_PARM01", "mxlSurfFlag", 123), ("GGL90_PARM01", "GGL90TKEFile", 124),
             ("GGL90_PARM01", "GGL90_dirichlet", 125), ("GGL90_PARM01", "calcMeanVertShear", 126),
             ("GGL90_PARM01", "useIDEMIX", 127), ("GGL90_PARM01", "useLANGMUIR", 128))
# #ifdef ALLOW_GGL90_IDEMIX (:130-153); IDEMIX_tau_v, IDEMIX_tau_h and IDEMIX_mu0 are expressions (code below)
_IDEMIX = (("IDEMIX_gamma", 138), ("IDEMIX_jstar", 139), ("IDEMIX_mixing_efficiency", 141),
           ("IDEMIX_diff_max", 142), ("IDEMIX_diff_min", 143), ("IDEMIX_frac_F_b", 144),
           ("IDEMIX_frac_F_s", 146), ("IDEMIX_tidal_file", 149), ("IDEMIX_wind_file", 150),
           ("IDEMIX_include_GM", 151), ("IDEMIX_include_GM_bottom", 152))
# #ifdef ALLOW_GGL90_LANGMUIR (:155-162)
_LANGMUIR = (("LC_Gamma", 159), ("LC_num", 160), ("LC_lambda", 161))
# REAL parameters that select code (kept as host values too, Ggl90.static_float)
_HOST = ("GGL90diffTKEh", "IDEMIX_tau_h", "GGL90TKEmin")


def ggl90_readparms(exp):
    """GGL90_READPARMS( myThid )   @63cdc0b pkg/ggl90/ggl90_readparms.F:6-451

    C     | SUBROUTINE GGL90_READPARMS                               |
    C     | o Routine to read in file data.ggl90                     |

    `exp`: mitjax.config.params.Experiment. Returns a `Ggl90` (GGL90.h) holding the parameters, or None when useGGL90
    is .FALSE. (:82-90: the routine returns after a warning).

    The namelist values come from mitjax/config (the run's data.ggl90, read as the model reads it); a variable the
    file does not set takes its default of :104-161, read from that line by `fortran_default`, or, where the default
    is an expression, computed here as the Fortran does: IDEMIX_tau_v = 2.0*86400.0 _d 0 (:136; `2.0` is a REAL*4
    literal, exact), IDEMIX_tau_h = 10.0*86400.0 _d 0 (:137), IDEMIX_mu0 = 1.0 _d 0/3.0 _d 0 (:140, one IEEE
    division), GGL90TKEbottom = UNSET_RL (:116), adMxlMaxFlag = UNSET_I (:122). The PARM02 (IDEMIX) group is read
    only with useIDEMIX (:188), PARM03 (Langmuir) only with useLANGMUIR (:209), each under its CPP option. Ported:
    the settings and checks after the READ (:240-310). Not carried (output only): GGL90dumpFreq (:104, = dumpFreq),
    GGL90mixingMaps (:106) and the printout (:312-441); GGL90taveFreq (:105, retired) raises when the file sets it
    (:292-310).
    """
    cfg = exp.cfg
    if not cfg.use_flag("useGGL90"):                               # :82-90
        return None
    rp = RunParams(exp.run)
    opt = "GGL90_OPTIONS.h"
    idemix_c = cfg.cpp.flag("ALLOW_GGL90_IDEMIX", opt)
    langmuir_c = cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", opt)
    v = {}
    for grp, name, line in _DEFAULTS:                              # :104-128 defaults, :174 READ GGL90_PARM01
        v[name] = rp.get(_F, grp, name, default=fortran_default(f"{_RP}:{line}", name, exp))
    # :116  GGL90TKEbottom = UNSET_RL ; :122  adMxlMaxFlag = UNSET_I
    v["GGL90TKEbottom"] = rp.get(_F, "GGL90_PARM01", "GGL90TKEbottom") if rp.has(_F, "GGL90_PARM01",
                                                                                  "GGL90TKEbottom") else UNSET_RL
    v["adMxlMaxFlag"] = rp.get(_F, "GGL90_PARM01", "adMxlMaxFlag") if rp.has(_F, "GGL90_PARM01",
                                                                              "adMxlMaxFlag") else UNSET_I
    if rp.has(_F, "GGL90_PARM01", "GGL90taveFreq"):                # :292-310 retired parameter
        raise RuntimeError('S/R GGL90_READPARMS: "GGL90taveFreq" is no longer allowed in file "data.ggl90"; '
                           "ABNORMAL END: S/R GGL90_READPARMS")
    if idemix_c:
        # :136-153 defaults; :188-205 READ GGL90_PARM02 only with useIDEMIX
        d = {"IDEMIX_tau_v": real4("2.0")*86400.0,                 # :136  2.0*86400.0 _d 0
             "IDEMIX_tau_h": real4("10.0")*86400.0,                # :137  10.0*86400.0 _d 0
             "IDEMIX_mu0": np.float64(1.0)/np.float64(3.0)}        # :140  1.0 _d 0/3.0 _d 0
        for name, line in _IDEMIX:
            d[name] = fortran_default(f"{_RP}:{line}", name, exp).value
        for name in d:
            if v["useIDEMIX"] and rp.has(_F, "GGL90_PARM02", name):
                d[name] = rp.get(_F, "GGL90_PARM02", name)
        v.update(d)
    elif v["useIDEMIX"]:
        # without ALLOW_GGL90_IDEMIX useIDEMIX only reaches GGL90_INIT_VARIA (:48 GGL90TKE = GGL90eps*maskC)
        raise NotImplementedError("GGL90_READPARMS: useIDEMIX = .TRUE. in a build without ALLOW_GGL90_IDEMIX is "
                                  "not ported")
    if langmuir_c:
        d = {name: fortran_default(f"{_RP}:{line}", name, exp).value for name, line in _LANGMUIR}  # :159-161
        for name in d:                                             # :209-226 READ GGL90_PARM03 with useLANGMUIR
            if v["useLANGMUIR"] and rp.has(_F, "GGL90_PARM03", name):
                d[name] = rp.get(_F, "GGL90_PARM03", name)
        v.update(d)
    elif v["useLANGMUIR"]:
        raise NotImplementedError("GGL90_READPARMS: useLANGMUIR = .TRUE. in a build without ALLOW_GGL90_LANGMUIR "
                                  "is not ported")
    f = {k: (np.float64(x) if isinstance(x, float) else x) for k, x in v.items()}

    sz = cfg.size
    if idemix_c and f["useIDEMIX"] and (sz.OLx < 3 or sz.OLy < 3):  # :241-247
        raise RuntimeError("OLx/OLy must be greater than 2; ABNORMAL END: S/R GGL90_READPARMS")
    if f["GGL90TKEbottom"] == UNSET_RL:                            # :251-253
        f["GGL90TKEbottom"] = f["GGL90TKEmin"]
    if f["adMxlMaxFlag"] == UNSET_I:                               # :254-256
        f["adMxlMaxFlag"] = f["mxlMaxFlag"]
    # :257-266  (`0.` is a REAL*4 zero)
    if (f["GGL90TKEmin"] < 0. if idemix_c else f["GGL90TKEmin"] <= 0.):
        raise RuntimeError("GGL90TKEmin must be greater than zero; ABNORMAL END: S/R GGL90_READPARMS")
    if f["GGL90TKEbottom"] < 0.:                                   # :267-272
        raise RuntimeError("GGL90TKEbottom must not be less than zero; ABNORMAL END: S/R GGL90_READPARMS")
    if f["GGL90mixingLengthMin"] <= 0.:                            # :273-278
        raise RuntimeError("GGL90mixingLengthMin must be greater than zero; ABNORMAL END: S/R GGL90_READPARMS")
    if f["GGL90viscMax"] <= 0.:                                    # :279-283
        raise RuntimeError("GGL90viscMax must be greater than zero; ABNORMAL END: S/R GGL90_READPARMS")
    if f["GGL90diffMax"] <= 0.:                                    # :284-288
        raise RuntimeError("GGL90diffMax must be greater than zero; ABNORMAL END: S/R GGL90_READPARMS")
    for k in ("mxlMaxFlag", "adMxlMaxFlag"):
        f[k] = int(f[k])
    host = {k: float(f[k]) for k in _HOST if k in f}
    return Ggl90(f, host)
