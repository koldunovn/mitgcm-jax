"""DWNSLP_READPARMS: pkg/down_slope/dwnslp_readparms.F @63cdc0b (lane M4ADLAB session 3, lab_sea/input_ad)."""

from dataclasses import dataclass

import numpy as np

from mitjax.params_io import RunParams

_F = "data.down_slope"
_G = "DWNSLP_PARM01"
debLevA = 1                    # EEPARAMS.h:91  PARAMETER ( debLevA=1 )
debLevD = 4                    # EEPARAMS.h:94  PARAMETER ( debLevD=4 )


@dataclass(frozen=True)
class DwnslpParams:
    """DWNSLP_PARAMS.h (host values: DWNSLP_INIT_FIXED uses the REAL ones on the host; DWNSLP_rec_mu enters the
    traced DWNSLP_CALC_FLOW as a constant) and the host flag of the init log (DWNSLP_ioUnit > 0 at INIT_FIXED)."""
    temp_useDWNSLP: bool
    salt_useDWNSLP: bool
    DWNSLP_slope: float
    DWNSLP_rec_mu: float
    DWNSLP_drFlow: float
    write_init_log: bool       # dwnslp_init_fixed.F:282-285 debugLevel >= debLevA (down_slope.<procId>.log)


def dwnslp_readparms(exp, *, tempStepping, saltStepping, debugLevel):
    """DWNSLP_READPARMS( myThid )   @63cdc0b pkg/down_slope/dwnslp_readparms.F:6-118

    C     | SUBROUTINE DWNSLP_READPARMS
    C     | o Routine to initialize Down-Sloping Parameters

    Returns DwnslpParams, or None when useDOWN_SLOPE is .FALSE. (:42-51, PACKAGES_UNUSED_MSG only prints).
    Defaults :65-69: temp_useDWNSLP = tempStepping, salt_useDWNSLP = saltStepping, DWNSLP_slope = DWNSLP_rec_mu =
    DWNSLP_drFlow = 0. (REAL*4 literal, exact zero); then the namelist DWNSLP_PARM01 (:72). Checks :91-102 STOP as
    the Fortran. The WRITE_0D_RL lines (:107-112) only print (not carried). Not ported (raise): temp_useDWNSLP or
    salt_useDWNSLP different from tempStepping / saltStepping (the onOffFlag of DWNSLP_APPLY is then static per
    tracer: TEMP_INTEGRATE / SALT_INTEGRATE run only with tempStepping / saltStepping), debugLevel >= debLevD (the
    log stays open: the per-step writes of DWNSLP_CALC_FLOW / DWNSLP_APPLY, dwnslp_init_fixed.F:328-331)."""
    cfg = exp.cfg
    if not cfg.use_flag("useDOWN_SLOPE"):                                       # :42-51
        return None
    rp = RunParams(exp.run)
    v = dict(temp_useDWNSLP=bool(tempStepping),                                 # :65
             salt_useDWNSLP=bool(saltStepping),                                 # :66
             DWNSLP_slope=0., DWNSLP_rec_mu=0., DWNSLP_drFlow=0.)               # :67-69
    for name in v:                                                              # :72
        if rp.has(_F, _G, name):
            v[name] = rp.get(_F, _G, name)
    if v["temp_useDWNSLP"] and not tempStepping:                                # :91-96
        raise RuntimeError("need tempStepping=T to apply DWNSLP to Temp (temp_useDWNSLP=T)\n"
                           "ABNORMAL END: S/R DWNSLP_READPARMS")
    if v["salt_useDWNSLP"] and not saltStepping:                                # :97-102
        raise RuntimeError("need saltStepping=T to apply DWNSLP to Salt (salt_useDWNSLP=T)\n"
                           "ABNORMAL END: S/R DWNSLP_READPARMS")
    if bool(v["temp_useDWNSLP"]) != bool(tempStepping) or bool(v["salt_useDWNSLP"]) != bool(saltStepping):
        raise NotImplementedError("DWNSLP: temp_useDWNSLP / salt_useDWNSLP = .FALSE. with tempStepping / "
                                  "saltStepping (dwnslp_apply.F:142-145 onOffFlag) is not ported")
    if debugLevel >= debLevD:                                                   # dwnslp_init_fixed.F:328-331
        raise NotImplementedError("DWNSLP: debugLevel >= debLevD (the per-step down_slope log of DWNSLP_CALC_FLOW "
                                  "and DWNSLP_APPLY) is not ported")
    return DwnslpParams(temp_useDWNSLP=bool(v["temp_useDWNSLP"]), salt_useDWNSLP=bool(v["salt_useDWNSLP"]),
                        DWNSLP_slope=float(np.float64(v["DWNSLP_slope"])),
                        DWNSLP_rec_mu=float(np.float64(v["DWNSLP_rec_mu"])),
                        DWNSLP_drFlow=float(np.float64(v["DWNSLP_drFlow"])),
                        write_init_log=bool(debugLevel >= debLevA))           # dwnslp_init_fixed.F:282-285
