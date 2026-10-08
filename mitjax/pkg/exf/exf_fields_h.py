"""EXF_FIELDS.h of pkg/exf @63cdc0b: the EXF fields as a dict {Fortran name: FArray}.

Every field is `_RL fld(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)`; each field read from a file has its two record
arrays `<fld>0`, `<fld>1` (EXF_FIELDS.h, same declaration). `exf_fields` returns them all zero: the value of a
common block no routine has written (EXF_INIT_VARIA then sets what it sets). Only the fields of the options this
port carries are present: a field the build compiles but this list lacks raises (KeyError) at its first use.
"""

import jax.numpy as jnp

from mitjax.farray import FArray

# EXF_FIELDS.h fields (each under the option that declares it, :13-260): the derived fields and the fields with
# records (`True`)
FIELDS = (("ustress", True), ("vstress", True), ("wStress", False), ("sflux", True), ("hflux", True),
          ("uwind", True), ("vwind", True), ("wspeed", True), ("sh", False), ("cw", False), ("sw", False),
          ("atemp", True), ("aqh", True), ("hs", False), ("hl", False), ("lwflux", True), ("evap", False),
          ("precip", True), ("snowprecip", True), ("swflux", True), ("swdown", True), ("lwdown", True),
          ("apressure", True), ("runoff", True), ("saltflx", True), ("climsst", True), ("climsss", True),
          ("tidePot", True), ("runoftemp", True))   # EXF_FIELDS.h:295-297 (EXF_ALLOW_TIDES), :281-284 (ALLOW_RUNOFTEMP)
# EXF_FIELDS.h:301-308 (#ifdef EXF_SEAICE_FRACTION; lane M4ADLAB session 3): areamask with its records and the _RS
# exf_iceFraction (float64: REAL4_IS_SLOW); present only when the build compiles the option
FIELDS_SEAICE_FRACTION = (("areamask", True), ("exf_iceFraction", False))


def xy(name, data, sz):
    """FArray of a (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, nSx, nSy) field from [tile, j, i] data."""
    return FArray(data, name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))


def exf_fields(sz, cfg=None):
    """{name: FArray} of EXF_FIELDS.h, all zero (see the module docstring); `cfg` (the build) adds the fields of
    EXF_SEAICE_FRACTION when it compiles the option."""
    shape = (sz.nSx*sz.nSy, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx)
    f = {}
    extra = (FIELDS_SEAICE_FRACTION if cfg is not None and cfg.cpp.flag("EXF_SEAICE_FRACTION", "EXF_OPTIONS.h")
             else ())
    for n, recs in FIELDS + extra:
        f[n] = xy(n, jnp.zeros(shape), sz)
        if recs:
            f[n + "0"] = xy(n + "0", jnp.zeros(shape), sz)
            f[n + "1"] = xy(n + "1", jnp.zeros(shape), sz)
    return f
