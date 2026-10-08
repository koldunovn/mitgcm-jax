"""PTRACERS_READPARMS: pkg/ptracers/ptracers_readparms.F @63cdc0b."""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.model.grid import UNSET_RL
from mitjax.params_io import RunParams
from mitjax.pkg.ptracers.ptracers_params_h import PtracersParams

_F = "data.ptracers"
_G = "PTRACERS_PARM01"
_OPT = "PTRACERS_OPTIONS.h"
# PTRACERS_PARM01 variables read only for output files (names, units, dump frequency, mnc file switches): not kept
_OUTPUT_ONLY = ("PTRACERS_names", "PTRACERS_long_names", "PTRACERS_units", "PTRACERS_dumpFreq",
                "PTRACERS_snapshot_mnc", "PTRACERS_pickup_write_mnc", "PTRACERS_pickup_read_mnc")


def _num(cfg):
    """PTRACERS_SIZE.h:15-16  PARAMETER(PTRACERS_num = ...) of the build (the experiment may override the file)."""
    from mitjax import paths
    from mitjax.config.params import code_path
    for d in (code_path(cfg), paths.UPSTREAM / "pkg" / "ptracers"):
        f = d / "PTRACERS_SIZE.h"
        if f.exists():
            import re
            m = re.search(r"^\s+PARAMETER\s*\(\s*PTRACERS_num\s*=\s*(\d+)\s*\)", f.read_text(), re.M | re.I)
            return int(m.group(1))
    raise FileNotFoundError("PTRACERS_SIZE.h")


def _elems(rp, name):
    v = rp.nml.var(_F, _G, name)
    return dict(v.value) if v is not None else {}


def ptracers_readparms(exp, pp):
    """PTRACERS_READPARMS( myThid )   @63cdc0b pkg/ptracers/ptracers_readparms.F:9-319

    C     Initialize PTRACERS parameters, read in data.ptracers

    `exp`: mitjax.config.params.Experiment. `pp`: the PARAMS.h values the routine reads (a mapping by Fortran name,
    as INI_PARMS left them): baseTime, saltAdvScheme, diffKhS, diffK4S, diffKrNrS (Nr values), useGMRedi,
    useDOWN_SLOPE, useKPP, doAB_onGtGs, dTtracerLev (Nr values), monitorFreq, and under ALLOW_MNC useMNC,
    monitor_mnc and pickup_write_mnc (MNC_PARAMS.h). Returns a `PtracersParams` (PTRACERS_PARAMS.h; the
    PTRACERS_START.h values are added by PTRACERS_INIT_FIXED and PTRACERS_INIT_VARIA), or None when usePTRACERS is
    .FALSE. (:89-97, the routine returns after a warning).

    Defaults :110-161 (cited per value; those that copy a PARAMS.h value take it from `pp`), the namelist READ :176
    (the run's data.ptracers through mitjax/config), then the settings derived after the READ: lambdaTr1ClimRelax
    (:212-223, ALLOW_COST), PTRACERS_numInUse (:229-240), PTRACERS_diffKrNr from PTRACERS_diffKr (:244-250),
    PTRACERS_startAllTrc (:253-268), PTRACERS_calcSurfCor (:272-275). Host-side, float64 (`1./tauTr1ClimRelax`: `1.`
    is an exact REAL*4 literal, one IEEE division). For PTRACERS_MONITOR: PTRACERS_monitorFreq (:109, static: the
    monitor runs on the host), PTRACERS_monitor_mnc (:146-154, :279, :293) and the io labels (PTRACERS_SET_IOLABEL,
    :102-104). Other output-only variables (names, units, dump frequency, file switches) are not kept. Raise: ALLOW_LONGSTEP
    (PTRACERS_dTLev = LS_nIter*dTtracerLev, :158-160), PTRACERS_useRecords, retired PTRACERS_taveFreq /
    PTRACERS_timeave_mnc (:190-208: STOP), tauTr1ClimRelax without ALLOW_COST (:213-217: STOP), PTRACERS_numInUse >
    PTRACERS_num (:234-240: STOP), different start times under ALLOW_AUTODIFF (:258-268: STOP), and
    PTRACERS_startStepFwd / PTRACERS_resetFreq (switching tracers on/off and resetting them is not ported)."""
    cfg = exp.cfg
    if not cfg.use_flag("usePTRACERS"):                                       # :89-97
        return None
    if cfg.cpp.flag("ALLOW_LONGSTEP", _OPT):                                  # :158-160
        raise NotImplementedError("PTRACERS_READPARMS: ALLOW_LONGSTEP (PTRACERS_dTLev = LS_nIter*dTtracerLev) is "
                                  "not ported")
    Nr = cfg.size.Nr
    num = _num(cfg)
    rp = RunParams(exp.run)
    for name in ("PTRACERS_useRecords", "PTRACERS_startStepFwd", "PTRACERS_resetFreq", "PTRACERS_resetPhase"):
        if _elems(rp, name) or rp.has(_F, _G, name):
            raise NotImplementedError(f"PTRACERS_READPARMS: {name} in data.ptracers is not ported")
    for name in ("PTRACERS_taveFreq", "PTRACERS_timeave_mnc"):                # :190-208
        if rp.has(_F, _G, name):
            raise ValueError(f'PTRACERS_READPARMS: "{name}" is no longer allowed in file "data.ptracers"; '
                             "ABNORMAL END: S/R PTRACERS_READPARMS")

    def scalar(name, default):
        return rp.nml.var(_F, _G, name).value if rp.has(_F, _G, name) else default

    def per_tracer(name, default):
        e = _elems(rp, name)
        return [e.get((n,), default) for n in range(1, num + 1)]

    baseTime = float(pp["baseTime"])
    # :110-161 defaults; :176 READ(UNIT=iUnit,NML=PTRACERS_PARM01)
    PTRACERS_Iter0 = int(scalar("PTRACERS_Iter0", 0))                         # :110  = 0
    PTRACERS_numInUse = int(scalar("PTRACERS_numInUse", -1))                  # :111  =-1
    advScheme = [int(x) for x in per_tracer("PTRACERS_advScheme", int(pp["saltAdvScheme"]))]   # :116
    implVertAdv = [bool(x) for x in per_tracer("PTRACERS_ImplVertAdv", False)]                 # :117  .FALSE.
    diffKh = [np.float64(x) for x in per_tracer("PTRACERS_diffKh", pp["diffKhS"])]             # :118
    diffK4 = [np.float64(x) for x in per_tracer("PTRACERS_diffK4", pp["diffK4S"])]             # :119
    diffKr = [np.float64(x) for x in per_tracer("PTRACERS_diffKr", UNSET_RL)]                  # :120  UNSET_RL
    diffKrNrS = np.asarray(pp["diffKrNrS"], np.float64)
    e = _elems(rp, "PTRACERS_diffKrNr")
    diffKrNr = [np.array([e.get((k, n), diffKrNrS[k - 1]) for k in range(1, Nr + 1)], np.float64)
                for n in range(1, num + 1)]                                   # :122  diffKrNrS(k)
    e = _elems(rp, "PTRACERS_ref")
    ref = [np.array([e.get((k, n), 0.0) for k in range(1, Nr + 1)], np.float64)
           for n in range(1, num + 1)]                                        # :123  0. _d 0
    EvPrRn = [np.float64(x) for x in per_tracer("PTRACERS_EvPrRn", UNSET_RL)]                  # :125  UNSET_RL
    useGMRedi = [bool(x) for x in per_tracer("PTRACERS_useGMRedi", bool(pp["useGMRedi"]))]     # :126
    useDWNSLP = [bool(x) for x in per_tracer("PTRACERS_useDWNSLP", bool(pp["useDOWN_SLOPE"]))]  # :127
    useKPP = [bool(x) for x in per_tracer("PTRACERS_useKPP", bool(pp["useKPP"]))]              # :128
    linFSConserve = [bool(x) for x in per_tracer("PTRACERS_linFSConserve", False)]             # :129  .FALSE.
    stay0 = bool(cfg.cpp.flag("GAD_SMOLARKIEWICZ_HACK", _OPT))                # :130-134
    stayPositive = [bool(x) for x in per_tracer("PTRACERS_stayPositive", stay0)]
    initialFile = [str(x) for x in per_tracer("PTRACERS_initialFile", " ")]   # :135  ' '
    monitorFreq = float(scalar("PTRACERS_monitorFreq", float(pp["monitorFreq"])))  # :109  = monitorFreq
    if cfg.cpp.flag("ALLOW_MNC", _OPT):                                       # :146-149 / :151-154
        monitor_mnc = bool(scalar("PTRACERS_monitor_mnc", bool(pp["useMNC"]) and bool(pp["monitor_mnc"])))
        monitor_mnc = bool(pp["useMNC"]) and monitor_mnc                      # :279
        pickup_write_mnc = bool(scalar("PTRACERS_pickup_write_mnc",          # :148, :280
                                       bool(pp["useMNC"]) and bool(pp["pickup_write_mnc"])))
        pickup_write_mnc = bool(pp["useMNC"]) and pickup_write_mnc
    else:
        monitor_mnc = False                                                   # :293
        pickup_write_mnc = False                                              # :294
    doAB_onGpTr = bool(scalar("PTRACERS_doAB_onGpTr", bool(pp["doAB_onGtGs"])))   # :142
    addSrelax2EmP = bool(scalar("PTRACERS_addSrelax2EmP", False))             # :143  .FALSE.
    tauTr1ClimRelax = np.float64(scalar("tauTr1ClimRelax", 0.0))              # :156  0.
    dTLev = np.asarray(pp["dTtracerLev"], np.float64).copy()                  # :157-163  PTRACERS_dTLev(k) = dTtracerLev(k)

    # :212-223  Tracer 1 climatology relaxation time scale
    lambdaTr1ClimRelax = np.float64(0.0)
    if tauTr1ClimRelax != 0.:
        if not cfg.cpp.flag("ALLOW_COST", _OPT):                              # :213-217
            raise ValueError(" PTRACERS_READPARMS: tauTr1ClimRelax has been removed (code is gone)")
        lambdaTr1ClimRelax = np.float64(1.0) / tauTr1ClimRelax                # :219  1./tauTr1ClimRelax
    elif cfg.cpp.flag("ALLOW_COST", _OPT):
        lambdaTr1ClimRelax = np.float64(0.0)                                  # :221  0.
    # :229-231
    if PTRACERS_numInUse < 0:
        PTRACERS_numInUse = num
    if PTRACERS_numInUse > num:                                               # :234-240
        raise ValueError(f" PTRACERS_READPARMS: You requested{PTRACERS_numInUse:4d} tracers at run time when only"
                         f"{num:4d} were specified at compile time. Naughty! ")
    # :244-250  Set vertical diffusion array
    for n in range(PTRACERS_numInUse):
        if diffKr[n] != UNSET_RL:
            diffKrNr[n] = np.full(Nr, diffKr[n], np.float64)
    # :253-268  PTRACERS_startStepFwd(iTracer) = baseTime (:113; the file may not set it: raised above)
    startAllTrc = True
    # :272-275
    calcSurfCor = any(linFSConserve)

    # :102-104 PTRACERS_SET_IOLABEL: the first label set '01'..'99' (ptracers_set_iolabel.F:93-106; '00' skipped)
    if num > 99:
        raise NotImplementedError("PTRACERS_SET_IOLABEL: more than 99 tracers (2nd and 3rd label sets) is not ported")
    ioLabel = tuple(f"{n:02d}" for n in range(1, num + 1))
    static = dict(PTRACERS_num=num, PTRACERS_numInUse=PTRACERS_numInUse, PTRACERS_Iter0=PTRACERS_Iter0,
                  PTRACERS_monitorFreq=monitorFreq, PTRACERS_monitor_mnc=monitor_mnc, PTRACERS_ioLabel=ioLabel,
                  PTRACERS_pickup_write_mnc=pickup_write_mnc,
                  PTRACERS_advScheme=tuple(advScheme), PTRACERS_ImplVertAdv=tuple(implVertAdv),
                  PTRACERS_useGMRedi=tuple(useGMRedi), PTRACERS_useDWNSLP=tuple(useDWNSLP),
                  PTRACERS_useKPP=tuple(useKPP), PTRACERS_linFSConserve=tuple(linFSConserve),
                  PTRACERS_stayPositive=tuple(stayPositive), PTRACERS_initialFile=tuple(initialFile),
                  PTRACERS_doAB_onGpTr=doAB_onGpTr, PTRACERS_addSrelax2EmP=addSrelax2EmP,
                  PTRACERS_startAllTrc=startAllTrc, PTRACERS_calcSurfCor=calcSurfCor,
                  PTRACERS_diffKh_ne_0=tuple(bool(x != 0.) for x in diffKh),
                  PTRACERS_diffK4_ne_0=tuple(bool(x != 0.) for x in diffK4),
                  PTRACERS_EvPrRn_ne_UNSET=tuple(bool(x != UNSET_RL) for x in EvPrRn),
                  _baseTime=baseTime)
    traced = dict(PTRACERS_dTLev=FArray(jnp.asarray(dTLev), "PTRACERS_dTLev", k=(1, Nr), tiled=False),
                  PTRACERS_diffKh=tuple(diffKh), PTRACERS_diffK4=tuple(diffK4), PTRACERS_EvPrRn=tuple(EvPrRn),
                  PTRACERS_diffKrNr=tuple(FArray(jnp.asarray(a), "PTRACERS_diffKrNr", k=(1, Nr), tiled=False) for a in diffKrNr),
                  PTRACERS_ref=tuple(FArray(jnp.asarray(a), "PTRACERS_ref", k=(1, Nr), tiled=False) for a in ref),
                  lambdaTr1ClimRelax=lambdaTr1ClimRelax)
    return PtracersParams(static, traced)
