"""EXF_INIT_VARIA: pkg/exf/exf_init_varia.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.exf.exf_init_fld import exf_init_fld

# EXF_INIT_FLD calls of :146-414 after the wind block, in order, with the option compiling each
_INIT_CALLS = (("wspeed", None), ("hflux", None), ("sflux", None), ("atemp", "ALLOW_ATM_TEMP"),
               ("aqh", "ALLOW_ATM_TEMP"), ("lwflux", "ALLOW_ATM_TEMP"), ("precip", "ALLOW_ATM_TEMP"),
               ("snowprecip", "ALLOW_ATM_TEMP"), ("swflux", "ALLOW_ATM_TEMP|SHORTWAVE_HEATING"),
               ("swdown", "ALLOW_DOWNWARD_RADIATION"), ("lwdown", "ALLOW_DOWNWARD_RADIATION"),
               ("apressure", "ATMOSPHERIC_LOADING"), ("tidePot", "EXF_ALLOW_TIDES"),
               ("areamask", "EXF_SEAICE_FRACTION"), ("runoff", "ALLOW_RUNOFF"),
               ("runoftemp", "ALLOW_RUNOFF&ALLOW_RUNOFTEMP"), ("saltflx", "ALLOW_SALTFLX"),
               ("climsst", "ALLOW_CLIMSST_RELAXATION"), ("climsss", "ALLOW_CLIMSSS_RELAXATION"))


def _compiled(cfg, opt):
    if opt is None:
        return True
    if "&" in opt:
        return all(cfg.cpp.flag(o, "EXF_OPTIONS.h") for o in opt.split("&"))
    return any(cfg.cpp.flag(o, "EXF_OPTIONS.h") if o != "SHORTWAVE_HEATING" else cfg.cpp.flag(o)
               for o in opt.split("|"))


def exf_init_varia(f, *, cfg, exf, grid, params, rw):
    """EXF_INIT_VARIA( myThid )   @63cdc0b pkg/exf/exf_init_varia.F:3-493

    C     | o Initialise the EXF fields (constant values; time-constant fields read once).

    `f` EXF_FIELDS.h (exf_fields_h). Returns the new dict. Raises for options not ported: ALLOW_READ_TURBFLUXES,
    EXF_READ_EVAP (evap stays the never-written zero otherwise, as here),
    ALLOW_CLIMSTRESS_RELAXATION, ALLOW_BULK_OFFLINE. Lane M4ADLAB session 3: EXF_SEAICE_FRACTION (areamask,
    :344-355; areamaskfile is refused in EXF_READPARMS, so the field is areamaskconst). Lane M4CS32ICE (global_ocean.cs32x15/input.seaice): useAtmWind
    = .FALSE. (ustress/vstress by EXF_INIT_FLD, :64-90; uwind/vwind = 0, :91-103 and :132-143), EXF_ALLOW_TIDES
    (tidePot, :331-342; tidePotFile is refused in EXF_READPARMS, so the field is tidePotconst) and ALLOW_RUNOFTEMP
    (runoftemp with runoff's mask and period, :368-379)."""
    for opt in ("ALLOW_READ_TURBFLUXES", "EXF_READ_EVAP",
                "ALLOW_CLIMSTRESS_RELAXATION", "ALLOW_BULK_OFFLINE"):
        if cfg.cpp.flag(opt, "EXF_OPTIONS.h"):
            raise NotImplementedError(f"EXF_INIT_VARIA: {opt} is not ported")
    sz = cfg.size
    f = dict(f)
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    for n in ("wStress", "cw", "sw", "sh"):                                    # :51-54
        f[n] = f[n].at[i, j].set(0.)
    if cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h"):                        # :55-58
        f["hs"] = f["hs"].at[i, j].set(0.)
        f["hl"] = f["hl"].at[i, j].set(0.)
    def init(n, mask=None, period=None, const=None):
        f[n], f[n + "0"], f[n + "1"] = exf_init_fld(
            n, getattr(exf, n + "file"), getattr(exf, (mask or n) + "mask"), getattr(exf, (period or n) + "period"),
            getattr(exf, "exf_inscal_" + n), getattr(exf, (const or n) + "const"), f[n], f[n + "0"], f[n + "1"],
            cfg=cfg, exf=exf, grid=grid, params=params, rw=rw)
    if not exf.useAtmWind:                                                     # :64-90
        init("ustress")
        init("vstress")
    else:                                                                      # :91-103
        f["uwind"] = f["uwind"].at[i, j].set(0.)
        f["vwind"] = f["vwind"].at[i, j].set(0.)
    if exf.useAtmWind:                                                         # :105-131
        init("uwind")
        init("vwind")
    else:                                                                      # :132-143
        f["uwind"] = f["uwind"].at[i, j].set(0.)
        f["vwind"] = f["vwind"].at[i, j].set(0.)
    for n, opt in _INIT_CALLS:                                                 # :146-414
        if not _compiled(cfg, opt):
            continue
        if n == "runoftemp":                                                   # :368-379
            init(n, mask="runoff", period="runoff")
        else:
            init(n)
    return f
