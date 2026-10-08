"""KPP_READPARMS: pkg/kpp/kpp_readparms.F @63cdc0b (and the parameter checks of KPP_CHECK, pkg/kpp/kpp_check.F)."""

import numpy as np

from mitjax.model.grid import UNSET_RL
from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.kpp.kpp_params_h import Kpp

_RP = "pkg/kpp/kpp_readparms.F"
_F = "data.kpp"
_G = "KPP_PARM01"
UNSET_I = 123456789            # EEPARAMS.h:85  PARAMETER ( UNSET_I      = 123456789  )

# (name, kpp_readparms.F line of its literal default): the namelist variables of KPP_PARM01 (:39-55) the port reads
_DEFAULTS = (("KPPwriteState", 83), ("KPPuseDoubleDiff", 84), ("LimitHblStable", 85),
             ("KPP_ghatUseTotalDiffus", 86), ("KPPuseSWfrac3D", 87),
             ("epsln", 95), ("phepsi", 96), ("epsilon", 97), ("vonk", 98), ("dB_dz", 99), ("conc1", 100),
             ("conam", 101), ("concm", 102), ("conc2", 103), ("zetam", 104), ("conas", 105), ("concs", 106),
             ("conc3", 107), ("zetas", 108), ("Ricr", 112), ("cekman", 113), ("cmonob", 114), ("concv", 115),
             ("hbf", 116), ("zmin", 121), ("zmax", 122), ("umin", 123), ("umax", 124), ("num_v_smooth_Ri", 128),
             ("Riinfty", 129), ("BVSQcon", 130), ("difm0", 132), ("difs0", 133), ("dift0", 134), ("difmcon", 136),
             ("difscon", 137), ("diftcon", 138), ("Rrho0", 142), ("dsfmax", 143), ("cstar", 147))
# retired namelist variables (:150-155 defaults; :195-239 an error and STOP when the file sets them)
_RETIRED = ("KPPmixingMaps", "num_v_smooth_BV", "num_z_smooth_sh", "num_m_smooth_sh", "kpp_taveFreq")


def kpp_readparms(exp, tp, io):
    """KPP_READPARMS( myThid )   @63cdc0b pkg/kpp/kpp_readparms.F:8-249

    C     | SUBROUTINE KPP_READPARMS
    C     | o Routine to initialize KPP parameters, read in data.kpp

    `exp`: mitjax.config.params.Experiment; `tp`: its TimeParams (deltaTClock), `io`: its IoParams (dumpFreq), both
    of mitjax/model/src/ini_parms.py. Returns a `Kpp` (KPP_PARAMS.h) with the namelist parameters, or None when useKPP
    is .FALSE. (:59-67: the routine returns after PACKAGES_UNUSED_MSG).

    Values from the run's data.kpp as the model reads it (mitjax/config); a variable the file does not set takes its
    default of :81-147, read from that line by `fortran_default` (`cstar = 10.` is a REAL*4 literal, exact); the
    three non-literal defaults are written here with their lines: `kpp_freq = deltaTClock` (:81), `kpp_dumpFreq =
    dumpFreq` (:82), `minKPPhbl = UNSET_RL` (:88). Retired variables set in the file (:195-239) STOP. Not ported
    (raise): num_v_smooth_Ri > 0 (Z121 in RI_IWMIX; no M3 variant sets it). The output switches KPPwriteState and
    kpp_dumpFreq are carried for KPP_CHECK / KPP_OUTPUT only (output, not ported).
    """
    cfg = exp.cfg
    if not cfg.use_flag("useKPP"):                                             # :59-67
        return None
    rp = RunParams(exp.run)
    v = {}
    for name, line in _DEFAULTS:                                               # :83-147 defaults, :163 READ
        v[name] = rp.get(_F, _G, name, default=fortran_default(f"{_RP}:{line}", name, exp))
    # :81 kpp_freq = deltaTClock ; :82 kpp_dumpFreq = dumpFreq ; :88 minKPPhbl = UNSET_RL
    v["kpp_freq"] = rp.get(_F, _G, "kpp_freq") if rp.has(_F, _G, "kpp_freq") else float(tp.deltaTClock)
    v["kpp_dumpFreq"] = rp.get(_F, _G, "kpp_dumpFreq") if rp.has(_F, _G, "kpp_dumpFreq") else float(io.dumpFreq)
    v["minKPPhbl"] = rp.get(_F, _G, "minKPPhbl") if rp.has(_F, _G, "minKPPhbl") else UNSET_RL
    # :195-239 the tests are on the VALUES the READ left (lane M4COL fix: 1D_ocean_ice_column/input/data.kpp sets
    # KPPmixingMaps = .FALSE., which the Fortran accepts): KPPmixingMaps .TRUE. (:195), num_v_smooth_BV /
    # num_z_smooth_sh / num_m_smooth_sh .NE. UNSET_I (:201, :207, :213; defaults UNSET_I), kpp_taveFreq .NE. UNSET_RL
    # (:219; default UNSET_RL)

    def _retired(n):
        if not rp.has(_F, _G, n):
            return False                                    # the default: never retired
        val = rp.get(_F, _G, n)
        if n == "KPPmixingMaps":
            return bool(val)
        if n == "kpp_taveFreq":
            return float(val) != UNSET_RL
        return int(val) != UNSET_I
    retired = [n for n in _RETIRED if _retired(n)]                             # :195-239
    if retired:
        raise RuntimeError(f"S/R KPP_READPARMS: Error reading file \"data.kpp\": {len(retired):4d} out-of-date "
                           f"parameters were found in the namelist(s) ({', '.join(retired)})\n"
                           "ABNORMAL END: S/R KPP_READPARMS")
    if v["num_v_smooth_Ri"] > 0:
        raise NotImplementedError("KPP_READPARMS: num_v_smooth_Ri > 0 (Z121 smoothing in RI_IWMIX) is not ported")
    f = {k: (np.float64(x) if isinstance(x, float) else x) for k, x in v.items()}
    # host flag for KPP_CALC's `DIFFERENT_MULTIPLE(kpp_freq,myTime,deltaTClock)` (kpp_calc.F:222; see KPP_CALC)
    f["kpp_freq_eq_deltaTClock"] = bool(float(v["kpp_freq"]) == float(tp.deltaTClock))
    return Kpp(f)


def kpp_check(kpp, *, params, exp):
    """KPP_CHECK( myThid )   @63cdc0b pkg/kpp/kpp_check.F:3-187: the parameter checks (:126-145; the WRITE_0D
    print-out :30-123 is output only). `params`: the dyn Params (implicitViscosity, momStepping); cAdjFreq,
    ivdc_kappa, implicitDiffusion are read from the run's `data` (host values: an IF on a REAL namelist value that
    STOPs is a setup-time decision). Raises with the Fortran message where the Fortran STOPs."""
    if kpp is None:
        return
    rp = RunParams(exp.run)
    sd = "model/src/set_defaults.F"

    def nml(key, line):
        return rp.get("data", "PARM01" if key != "cAdjFreq" else "PARM03", key,
                      default=fortran_default(f"{sd}:{line}", key, exp))
    cAdjFreq, ivdc_kappa = nml("cAdjFreq", 327), nml("ivdc_kappa", 221)
    if cAdjFreq != 0.0 or ivdc_kappa != 0.0:                                   # :127-132
        raise RuntimeError("Some form of convection has been enabled\nABNORMAL END: S/R KPP_CHECK")
    if not nml("implicitDiffusion", 209):                                      # :135-139
        raise RuntimeError("KPP needs implicitDiffusion to be enabled\nABNORMAL END: S/R KPP_CHECK")
    if not params.implicitViscosity and params.momStepping:                    # :140-144
        raise RuntimeError("KPP needs implicitViscosity to be enabled\nABNORMAL END: S/R KPP_CHECK")
