"""CD_CODE_INIT_FIXED: pkg/cd_code/cd_code_init_fixed.F @63cdc0b."""


def cd_code_init_fixed(*, cfg, ip):
    """CD_CODE_INIT_FIXED( myThid )   @63cdc0b pkg/cd_code/cd_code_init_fixed.F:3-73

    Called by PACKAGES_INIT_FIXED when useCDscheme (packages_init_fixed.F:208-215). The routine's only statements are
    the MNC variable definitions under `#ifdef ALLOW_MNC` / `IF (useMNC)` (:21-69): without MNC it does nothing.
    Raise: useMNC (pkg/mnc is not ported). `ip`: InitParams (useMNC)."""
    if cfg.cpp.ALLOW_CD_CODE and cfg.cpp.ALLOW_MNC and ip.useMNC:               # :21-23
        raise NotImplementedError("CD_CODE_INIT_FIXED: the MNC definitions (useMNC) are not ported")
    return None
