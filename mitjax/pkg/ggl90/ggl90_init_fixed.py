"""GGL90_INIT_FIXED: pkg/ggl90/ggl90_init_fixed.F @63cdc0b."""


def ggl90_init_fixed(*, cfg):
    """GGL90_INIT_FIXED( myThid )   @63cdc0b pkg/ggl90/ggl90_init_fixed.F:7-67

    C     | S/R GGL90_INIT_FIXED
    C     | Initialize GGL90 variables that are kept fixed during the run.

    Without ALLOW_GGL90_SMOOTH the routine only calls GGL90_DIAGNOSTICS_INIT (:226-230, pkg/diagnostics: not ported,
    output only), so it sets no model variable. Raise: ALLOW_GGL90_SMOOTH (mskCor, :232-250)."""
    if cfg.cpp.flag("ALLOW_GGL90_SMOOTH", "GGL90_OPTIONS.h"):
        raise NotImplementedError("GGL90_INIT_FIXED: ALLOW_GGL90_SMOOTH (mskCor) is not ported")
    return None
