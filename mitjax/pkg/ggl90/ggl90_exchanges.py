"""GGL90_EXCHANGES: pkg/ggl90/ggl90_exchanges.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_XYZ_RL


def ggl90_exchanges(*, cfg, ggl, ex):
    """GGL90_EXCHANGES( myThid )   @63cdc0b pkg/ggl90/ggl90_exchanges.F:7-48

    C     | S/R GGL90_EXCHANGES
    C     | Exchange data to update overlaps for GGL90TKE

    Returns `ggl` with GGL90TKE (and IDEMIX_E) exchanged (_EXCH_XYZ_RL) where the Fortran does: under
    ALLOW_GGL90_IDEMIX, both with useIDEMIX .AND. IDEMIX_tau_h > 0 (:32-34), else GGL90TKE with GGL90diffTKEh > 0
    (:35-39); the REAL thresholds are decided on the host from the namelist values (Ggl90.static_float)."""
    zeroRL = 0.0                                                        # EEPARAMS.h:72
    if cfg.cpp.flag("ALLOW_GGL90_IDEMIX", "GGL90_OPTIONS.h") and ggl.useIDEMIX \
            and ggl.static_float("IDEMIX_tau_h") > zeroRL:              # :32
        return ggl.replace(GGL90TKE=EXCH_XYZ_RL(ggl.GGL90TKE, ex=ex),   # :33
                           IDEMIX_E=EXCH_XYZ_RL(ggl.IDEMIX_E, ex=ex))   # :34
    if ggl.static_float("GGL90diffTKEh") > zeroRL:                      # :35 / :37
        return ggl.replace(GGL90TKE=EXCH_XYZ_RL(ggl.GGL90TKE, ex=ex))   # :39
    return ggl
