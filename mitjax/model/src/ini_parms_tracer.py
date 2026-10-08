"""INI_PARMS / SET_PARMS / GAD_INIT_FIXED / INI_MODEL_IO (@63cdc0b), tracer group: the PARAMS.h and GAD.h values the
tracer step reads (THERMODYNAMICS, TEMP_INTEGRATE, GAD_CALC_RHS, GAD_IMPLICIT_R, CALC_3D_DIFFUSIVITY,
DO_OCEANIC_PHYS's convection and density-gradient part), added to the dynamics `Params` of
mitjax/model/src/ini_parms.ini_parms_dyn (core lane) as further static / traced entries.

Same construction as ini_parms.py: namelist values from the run's parameter files, every value the files do not set
from its cited SET_DEFAULTS line (`fortran_default`, checked to be live in this build), the derivation statements
ported as code with their lines. Kept in the tracer lane's own file (ini_parms.py belongs to the core lane), to be
folded into ini_parms.py when the lanes merge.

Diagnostics (decided 2026-10-01 by Nikolay: pkg/diagnostics is not ported; the Fortran %MON
output is unchanged with diagnostics off, job27827363-diagoff): `useDiagnostics` is .FALSE. for every kernel (no
DIAGNOSTICS_FILL output, no botDragU/V accumulators). The one channel through which a diagnostics request changes
model state is DIAGNOSTICS_IS_ON: DO_OCEANIC_PHYS's doDiagsRho (do_oceanic_phys.F:254-266) runs GRAD_SIGMA at k=1
and CALC_OCE_MXLAYER (hMixLayer) when 'MXLDEPTH' is requested. DIAGNOSTICS_IS_ON reads only static lists (the
fields of data.diagnostics), so it is evaluated here on the host: `diagnostics_is_on` is the tuple of names it
returns .TRUE. for in this run (empty when the run's useDiagnostics is .FALSE.).
"""

import numpy as np

from mitjax.model.grid import UNSET_RL
from mitjax.model.src.ini_parms import (IP, SD, UNSET_I, _get, _get_or_unset, _rk, _use, _vector,
                                        gad_init_fixed_ab_flags)
from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.generic_advdiff.gad_h import (ENUM_CENTERED_2ND, ENUM_CENTERED_4TH, ENUM_UPWIND_3RD)

SP = "model/src/set_parms.F"
GIF = "pkg/generic_advdiff/gad_init_fixed.F"


def diagnostics_is_on(exp, rp=None):
    """The names DIAGNOSTICS_IS_ON( diagName ) returns .TRUE. for (pkg/diagnostics/diagnostics_is_on.F:9-64
    @63cdc0b), evaluated on the host: a name listed in fields(:,n) of DIAGNOSTICS_LIST (:57-63) or in
    stat_fields(:,n) of DIAG_STATIS_PARMS (:65-75) of data.diagnostics. idiag(m,n) /= 0 holds for every listed name
    that a compiled package registers (DIAGNOSTICS_SET_POINTERS stops on an unknown name), ndiag(ip) and
    qSdiag(0,0,iSp) are >= 0 during the run (counters). Empty when the run's useDiagnostics is .FALSE. (the package
    is then not initialised and no S/R calls DIAGNOSTICS_IS_ON: every caller tests useDiagnostics first)."""
    cfg = exp.cfg
    if not (cfg.cpp.ALLOW_DIAGNOSTICS and _use(cfg, "useDiagnostics")):
        return ()
    rp = RunParams(exp.run) if rp is None else rp
    names = []
    for group, key in (("DIAGNOSTICS_LIST", "fields"), ("DIAG_STATIS_PARMS", "stat_fields")):
        if rp.nml.var("data.diagnostics", group, key) is None:
            continue
        elems, _ = rp.get_array("data.diagnostics", group, key)
        for v in elems.values():
            if str(v).strip():
                names.append(f"{str(v):<8s}"[:8])
    return tuple(sorted(set(names)))


def ini_parms_tracer(exp, params, tp, ip):
    """`params` (ini_parms_dyn) with the tracer-step values added. `tp`, `ip`: TimeParams, InitParams of the same
    experiment. Branches the M1 variants do not take raise."""
    cfg = exp.cfg
    rp = RunParams(exp.run)
    Nr = cfg.size.Nr
    g = lambda grp, key, line, src=SD: _get(exp, rp, grp, key, f"{src}:{line}")    # noqa: E731

    tempStepping = params.tempStepping
    # ---- PARM01 values read as they are (namelist or SET_DEFAULTS line)
    diffKhT = g("PARM01", "diffKhT", 153)
    diffK4T = g("PARM01", "diffK4T", 155)
    diffKrBL79surf = g("PARM01", "diffKrBL79surf", 159)
    diffKrBL79deep = g("PARM01", "diffKrBL79deep", 160)
    diffKrBL79scl = g("PARM01", "diffKrBL79scl", 161)
    diffKrBL79Ho = g("PARM01", "diffKrBL79Ho", 162)
    temp_stayPositive = g("PARM01", "temp_stayPositive", 197)
    implicitDiffusion = g("PARM01", "implicitDiffusion", 209)
    tempImplVertAdv = g("PARM01", "tempImplVertAdv", 213)
    ivdc_kappa = g("PARM01", "ivdc_kappa", 221)
    tempAdvScheme = g("PARM01", "tempAdvScheme", 226)
    saltAdvScheme = g("PARM01", "saltAdvScheme", 227)
    multiDimAdvection = g("PARM01", "multiDimAdvection", 228)
    tempAdvection_nml = g("PARM01", "tempAdvection", 195)
    hMixCriteria = g("PARM01", "hMixCriteria", 222)
    dRhoSmall = g("PARM01", "dRhoSmall", 223)
    hMixSmooth = g("PARM01", "hMixSmooth", 224)
    forcing_In_AB = g("PARM03", "forcing_In_AB", 969, IP)
    useCubedSphereExchange = _get(exp, rp, "EEPARMS", "useCubedSphereExchange",
                                  "eesupp/src/eeset_parms.F:106", fname="eedata")
    for grp, key in (("PARM01", "diffKpT"), ("PARM01", "BL79LatVary"),
                     ("PARM01", "smag3D_diffCoeff"), ("PARM03", "startFromPickupAB2")):
        if rp.has("data", grp, key):
            raise NotImplementedError(f"INI_PARMS: {key} in data is not ported (ini_parms_tracer)")
    if cfg.cpp.ALLOW_BL79_LAT_VARY:
        # PTRACERS lane: ALLOW_3D_DIFFKR is ported (CALC_3D_DIFFUSIVITY's diffKr arms, INI_MIXING)
        raise NotImplementedError("CALC_3D_DIFFUSIVITY: ALLOW_BL79_LAT_VARY is not ported")

    # ---- INI_PARMS derivations
    tempVertAdvScheme = _get_or_unset(rp, "PARM01", "tempVertAdvScheme", 0)     # ini_parms.F:424  = 0
    if tempVertAdvScheme == 0:                                                  # ini_parms.F:509
        tempVertAdvScheme = tempAdvScheme
    diffKrT = _get_or_unset(rp, "PARM01", "diffKrT", UNSET_RL)                   # ini_parms.F:414  = UNSET_RL
    # PTRACERS lane (tutorial_advection_in_gyre sets diffKzT): ini_parms.F:412 diffKzT = UNSET_RL before the READ,
    # :570 IF ( diffKrT .EQ. UNSET_RL ) diffKrT = diffKzT (diffKpT, :571, stays refused above)
    diffKzT = _get_or_unset(rp, "PARM01", "diffKzT", UNSET_RL)
    if diffKrT == UNSET_RL:
        diffKrT = diffKzT
    diffKrNrT = _vector(rp, "PARM01", "diffKrNrT", Nr, UNSET_RL)                 # set_defaults.F:169
    vertSetCount = int(np.count_nonzero(diffKrNrT != UNSET_RL))                 # ini_parms.F:572-575
    if 0 < vertSetCount < Nr:                                                   # :576-580
        raise ValueError(f"S/R INI_PARMS: Partial setting ({vertSetCount:5d} /{Nr:5d}) of diffKrNrT is not allowed")
    if diffKrT == UNSET_RL:                                                     # :582-583
        diffKrT = fortran_default(f"{SD}:157", "diffKrTDefault", exp).value
    elif vertSetCount > 0:                                                      # :584-588
        raise ValueError("S/R INI_PARMS: Cannot set both diffKrNrT and diffKrT (or Kp,Kz) in param file data")
    if vertSetCount == 0:                                                       # :592-596
        diffKrNrT[:] = diffKrT
    diffKr4T = _vector(rp, "PARM01", "diffKr4T", Nr, 0.)                         # set_defaults.F:171  0. (REAL*4, exact)
    if ivdc_kappa != 0. and not implicitDiffusion:                              # ini_parms.F:738-743
        raise ValueError("S/R INI_PARMS: To use ivdc_kappa you must enable implicit vertical diffusion.")
    tracForcingOutAB = _get_or_unset(rp, "PARM03", "tracForcingOutAB", UNSET_I)  # set_defaults.F:314  = UNSET_I
    if tracForcingOutAB == UNSET_I:                                                  # ini_parms.F:1099-1102
        tracForcingOutAB = 1
        if forcing_In_AB:
            tracForcingOutAB = 0

    # ---- SET_PARMS (set_parms.F:236-248)
    tempVertDiff4 = False                                                       # :236
    for k in range(1, Nr+1):                                                    # :238-241
        tempVertDiff4 = tempVertDiff4 or (diffKr4T[k-1] > 0.)
    tempAdvection = tempStepping and tempAdvection_nml                          # :242
    tempVertDiff4 = tempStepping and tempVertDiff4                              # :243
    tempForcing = tempStepping and g("PARM01", "tempForcing", 196)              # :244
    tempImplVertAdv = tempAdvection and tempImplVertAdv                         # :248
    if tempAdvection != params.tempAdvection:
        raise AssertionError("ini_parms_tracer: tempAdvection disagrees with ini_parms_dyn")

    # ---- GAD_INIT_FIXED (gad_init_fixed.F:126-132): multi-dimensional advection off for the linear schemes
    tempMultiDimAdvec = multiDimAdvection and tempAdvection                     # :126
    if tempAdvScheme in (ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH):    # :128-132
        tempMultiDimAdvec = False
    ab = gad_init_fixed_ab_flags(exp, rp)                                       # :142-162
    SmolarkiewiczMaxFrac = 1.0                                                  # :166  1. _d 0

    # ---- INI_MODEL_IO (ini_model_io.F:125): AB starting level of the tracers
    tempStartAB = tp.nIter0                                                     # :125

    # ---- ADVECT lane (plan Task 14): the salinity group (SALT_INTEGRATE), the mirror of the temperature group above
    saltStepping = params.saltStepping
    diffKhS = g("PARM01", "diffKhS", 154)
    diffK4S = g("PARM01", "diffK4S", 156)
    salt_stayPositive = g("PARM01", "salt_stayPositive", 201)
    saltImplVertAdv = g("PARM01", "saltImplVertAdv", 214)
    saltAdvection_nml = g("PARM01", "saltAdvection", 199)
    if rp.has("data", "PARM01", "diffKpS"):
        raise NotImplementedError("INI_PARMS: diffKpS in data is not ported (ini_parms_tracer)")
    saltVertAdvScheme = _get_or_unset(rp, "PARM01", "saltVertAdvScheme", 0)     # ini_parms.F:425  = 0
    if saltVertAdvScheme == 0:                                                  # ini_parms.F:510
        saltVertAdvScheme = saltAdvScheme
    diffKrS = _get_or_unset(rp, "PARM01", "diffKrS", UNSET_RL)                   # ini_parms.F:417  = UNSET_RL
    # GGL90 lane (vermix sets diffKzS): ini_parms.F:415 diffKzS = UNSET_RL before the READ,
    # :602 IF ( diffKrS .EQ. UNSET_RL ) diffKrS = diffKzS (diffKpS, :603, stays refused above)
    diffKzS = _get_or_unset(rp, "PARM01", "diffKzS", UNSET_RL)
    if diffKrS == UNSET_RL:
        diffKrS = diffKzS
    diffKrNrS = _vector(rp, "PARM01", "diffKrNrS", Nr, UNSET_RL)                 # set_defaults.F:170
    vertSetCountS = int(np.count_nonzero(diffKrNrS != UNSET_RL))                # ini_parms.F:604-607
    if 0 < vertSetCountS < Nr:                                                  # :608-613
        raise ValueError(f"S/R INI_PARMS: Partial setting ({vertSetCountS:5d} /{Nr:5d}) of diffKrNrS is not "
                         "allowed")
    if vertSetCountS == Nr:                                                     # :614-621
        if diffKrS != UNSET_RL:
            raise ValueError("S/R INI_PARMS: Cannot set both diffKrNrS and diffKrS (or Kp,Kz) in param file data")
    elif diffKrS != UNSET_RL:                                                   # :622-626
        diffKrNrS[:] = diffKrS
    else:                                                                       # :627-634
        diffKrS = fortran_default(f"{SD}:158", "diffKrSDefault", exp).value
        diffKrNrS[:] = diffKrNrT                                                # :632  diffKrNrS(k) = diffKrNrT(k)
    diffKr4S = _vector(rp, "PARM01", "diffKr4S", Nr, 0.)                         # set_defaults.F:172  0. (REAL*4, exact)
    saltVertDiff4 = False                                                       # set_parms.F:237
    for k in range(1, Nr+1):                                                    # :238-241
        saltVertDiff4 = saltVertDiff4 or (diffKr4S[k-1] > 0.)
    saltAdvection = saltStepping and saltAdvection_nml                          # :245
    saltVertDiff4 = saltStepping and saltVertDiff4                              # :246
    saltForcing = saltStepping and g("PARM01", "saltForcing", 200)              # :247
    saltImplVertAdv = saltAdvection and saltImplVertAdv                         # :249
    if saltAdvection != params.saltAdvection:
        raise AssertionError("ini_parms_tracer: saltAdvection disagrees with ini_parms_dyn")
    saltMultiDimAdvec = multiDimAdvection and saltAdvection                     # gad_init_fixed.F:127
    if saltAdvScheme in (ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH):    # :133-137
        saltMultiDimAdvec = False
    saltStartAB = tp.nIter0                                                     # ini_model_io.F:126
    # ALLOW_ADAMSBASHFORTH_3 (ADAMS_BASHFORTH3's weights): set_defaults.F:319-320 (5. _d 0 / 12. _d 0, an
    # expression, ported as code); without it alph_AB = beta_AB = UNSET_RL (:323-324, never read)
    if cfg.cpp.ALLOW_ADAMSBASHFORTH_3:
        alph_AB = g("PARM03", "alph_AB", 319)
        beta_AB = rp.get("data", "PARM03", "beta_AB") if rp.has("data", "PARM03", "beta_AB") else 5.0 / 12.0
    else:
        for name in ("alph_AB", "beta_AB"):
            if rp.has("data", "PARM03", name):     # config_check.F:189-197
                raise ValueError("CONFIG_CHECK: #undef ALLOW_ADAMSBASHFORTH_3 but alph_AB,beta_AB are set")
        alph_AB = beta_AB = UNSET_RL
    # ---- end of the ADVECT lane's salinity group

    static = dict(
        saltVertAdvScheme=saltVertAdvScheme, saltMultiDimAdvec=saltMultiDimAdvec, saltImplVertAdv=saltImplVertAdv,
        saltForcing=saltForcing, saltVertDiff4=saltVertDiff4, salt_stayPositive=salt_stayPositive,
        saltStartAB=saltStartAB, diffKhS_ne_0=bool(diffKhS != 0.), diffK4S_ne_0=bool(diffK4S != 0.),   # ADVECT
        tempAdvScheme=tempAdvScheme, tempVertAdvScheme=tempVertAdvScheme, saltAdvScheme=saltAdvScheme,
        multiDimAdvection=multiDimAdvection, tempMultiDimAdvec=tempMultiDimAdvec, tempImplVertAdv=tempImplVertAdv,
        implicitDiffusion=implicitDiffusion, tempForcing=tempForcing, tracForcingOutAB=tracForcingOutAB,
        tempVertDiff4=tempVertDiff4, temp_stayPositive=temp_stayPositive, tempStartAB=tempStartAB,
        useCubedSphereExchange=useCubedSphereExchange,
        # GO lane: PARAMS.h useMultiDimAdvec = .FALSE. (set_defaults.F:229) .OR. temp/saltMultiDimAdvec
        # (gad_init_fixed.F:140-141); PTRACERS_INIT_FIXED ORs its tracers' flags (:82-83, drivers.model)
        useMultiDimAdvec=bool(tempMultiDimAdvec or saltMultiDimAdvec),
        interDiffKr_pCell=g("PARM04", "interDiffKr_pCell", 1215, IP),
        AdamsBashforthGt=ab["AdamsBashforthGt"], AdamsBashforth_T=ab["AdamsBashforth_T"],
        AdamsBashforthGs=ab["AdamsBashforthGs"], AdamsBashforth_S=ab["AdamsBashforth_S"],
        # REAL comparisons that select code (KERNEL_GUIDE §4), decided on the host from the value below
        diffKhT_ne_0=bool(diffKhT != 0.),                                       # gad_calc_rhs.F:328, :457
        diffK4T_ne_0=bool(diffK4T != 0.),                                       # gad_calc_rhs.F:229, :339, :468
        implicDiv2DFlow_ne_0=bool(float(params.implicDiv2DFlow) != 0.),         # integr_continuity.F:211, :317
        # REAL values that select branches in CALC_OCE_MXLAYER (calc_oce_mxlayer.F:98-100, :223), read there through
        # params.static_float (COL lane's interface); the traced copies are below
        _floats=tuple(sorted(dict(params.static_items().get("_floats", ()),
                                  hMixCriteria=float(hMixCriteria), hMixSmooth=float(hMixSmooth)).items())),
        # diagnostics (module docstring)
        useDiagnostics=False, diagnostics_is_on=diagnostics_is_on(exp, rp),
    )
    f64 = np.float64
    traced = dict(
        diffKhT=f64(diffKhT), diffK4T=f64(diffK4T), ivdc_kappa=f64(ivdc_kappa),
        diffKrBL79surf=f64(diffKrBL79surf), diffKrBL79deep=f64(diffKrBL79deep),
        diffKrBL79scl=f64(diffKrBL79scl), diffKrBL79Ho=f64(diffKrBL79Ho),
        SmolarkiewiczMaxFrac=f64(SmolarkiewiczMaxFrac),
        hMixCriteria=f64(hMixCriteria), dRhoSmall=f64(dRhoSmall), hMixSmooth=f64(hMixSmooth),
        diffKrNrT=_rk("diffKrNrT", Nr, diffKrNrT), diffKr4T=_rk("diffKr4T", Nr, diffKr4T),
        dTtracerLev=_rk("dTtracerLev", Nr, tp.dTtracerLev),
        # ADVECT lane: salinity group and the AB3 weights
        diffKhS=f64(diffKhS), diffK4S=f64(diffK4S), diffKrNrS=_rk("diffKrNrS", Nr, diffKrNrS),
        diffKr4S=_rk("diffKr4S", Nr, diffKr4S), alph_AB=f64(alph_AB), beta_AB=f64(beta_AB),
    )
    return params.replace(static=static, traced=traced)
