"""ECCO_WRITE_PICKUP   @63cdc0b pkg/ecco/ecco_write_pickup.F:6-66 (lane M4ADCOL session 3)."""


def ecco_write_pickup(permPickup, suff, myTime, myIter, *, cfg):
    """ECCO_WRITE_PICKUP( permPickup, suff, myTime, myIter, myThid ), called by PACKAGES_WRITE_PICKUP under useECCO
    (packages_write_pickup.F:236-240): its whole body is #ifdef ALLOW_PSBAR_STERIC (:37-63, pickup_ecco.<suff> with
    VOLsumGlob_0, RHOsumGlob_0), which ECCO_OPTIONS.h leaves undefined (:47): it writes nothing. Returns the files
    written (none); a build with ALLOW_PSBAR_STERIC raises."""
    if cfg.cpp.flag("ALLOW_PSBAR_STERIC", "ECCO_OPTIONS.h"):
        raise NotImplementedError("ECCO_WRITE_PICKUP: ALLOW_PSBAR_STERIC (pickup_ecco) is not ported")
    return []
