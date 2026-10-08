"""EXF_RADIATION: pkg/exf/exf_radiation.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.exf.exf_param_h import stefanBoltzmann


def exf_radiation(exf_Tsf, myTime, myIter, f, *, cfg, exf):
    """EXF_RADIATION( exf_Tsf, myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_radiation.F:7-168

    C     | o Set radiative fluxes at the surface.

    `f` EXF_FIELDS.h (dict); returns the new dict. Ported: ALLOW_DOWNWARD_RADIATION with ALLOW_ATM_TEMP: lwflux from
    lwdown (:60-86, EXF_LWDOWN_WITH_EMISSIVITY :71-79 as compiled), swflux from swdown with exf_albedo (:107-150,
    without ALLOW_ZENITHANGLE). `exf_Tsf**4` is an integer power (lax.integer_pow, libgcc __powidf2's sequence).
    `1.0` is a REAL*4 literal (exact)."""
    if not cfg.cpp.flag("ALLOW_DOWNWARD_RADIATION", "EXF_OPTIONS.h"):          # :40
        return f
    if cfg.cpp.flag("ALLOW_ZENITHANGLE", "EXF_OPTIONS.h") and (exf.useExfZenAlbedo or exf.useExfZenIncoming):
        # :108-109 EXF_ZENITHANGLE and the zen_albedo arm :125-131 (lane M4CS32ICE: compiled, unused: the plain
        # exf_albedo arm :133-138 runs; ALLOW_AUTODIFF's zen_* reset :110-121 is not compiled with it)
        raise NotImplementedError("EXF_RADIATION: useExfZenAlbedo / useExfZenIncoming (ALLOW_ZENITHANGLE) is not "
                                  "ported")
    if cfg.cpp.flag("ALLOW_ZENITHANGLE", "EXF_OPTIONS.h") and cfg.cpp.flag("ALLOW_AUTODIFF"):
        raise NotImplementedError("EXF_RADIATION: ALLOW_ZENITHANGLE with ALLOW_AUTODIFF (:110-121) is not ported")
    sz = cfg.size
    f = dict(f)
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    if cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h"):                        # :52-104
        if exf.lwfluxfile.strip() == "" and exf.lwdownfile.strip() != "":      # :60
            if cfg.cpp.flag("EXF_LWDOWN_WITH_EMISSIVITY", "EXF_OPTIONS.h"):   # :67-79
                f["lwflux"] = f["lwflux"].at[i, j].set(
                    exf.ocean_emissivity*stefanBoltzmann*exf_Tsf[i, j]**4
                    - f["lwdown"][i, j]*exf.ocean_emissivity)
            else:
                f["lwflux"] = f["lwflux"].at[i, j].set(
                    exf.ocean_emissivity*stefanBoltzmann*exf_Tsf[i, j]**4
                    - f["lwdown"][i, j])
    if cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h") or cfg.cpp.flag("SHORTWAVE_HEATING"):   # :106-163
        if exf.swfluxfile.strip() == "" and exf.swdownfile.strip() != "":      # :107
            f["swflux"] = f["swflux"].at[i, j].set(-f["swdown"][i, j]          # :139-144
                                                   * (1.0-exf.exf_albedo))
    return f
