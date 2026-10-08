"""KPP_DO_EXCH: pkg/kpp/kpp_do_exch.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_3D_RL, EXCH_XYZ_RL


def kpp_do_exch(kppf, *, cfg, ex):
    """KPP_DO_EXCH( myThid )   @63cdc0b pkg/kpp/kpp_do_exch.F:6-42

    C     | SUBROUTINE KPP_DO_EXCH
    C     | o Apply exchanges to KPP variables

    `kppf`: the KPP.h fields {name: FArray}; returns the dict with KPPviscAz exchanged, EXCH_3D_RL( KPPviscAz, Nr )
    (:34, #ifndef ALLOW_AUTODIFF) or _EXCH_XYZ_RL( KPPviscAz ) (:36, ALLOW_AUTODIFF; lane M4ADCOL). The other KPP.h
    fields are not exchanged."""
    kppf = dict(kppf)
    if cfg.cpp.flag("ALLOW_AUTODIFF", "KPP_OPTIONS.h"):
        kppf["KPPviscAz"] = EXCH_XYZ_RL(kppf["KPPviscAz"], ex=ex)              # :36
    else:
        kppf["KPPviscAz"] = EXCH_3D_RL(kppf["KPPviscAz"], cfg.size.Nr, ex=ex)   # :34
    return kppf
