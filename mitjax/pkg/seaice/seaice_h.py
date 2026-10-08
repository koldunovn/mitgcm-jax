"""SEAICE.h and SEAICE_GRID.h of pkg/seaice @63cdc0b: the sea-ice common blocks as a dict {Fortran name: FArray}.

Every field is `_RL fld(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)` (or `_RS`, float64 in this build) except TICES
`(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nITD,nSx,nSy)` (SEAICE.h:245; the nITD axis is the FArray's k axis) and the INTEGER
KGEO (SEAICE.h:210, int32), and (lane M4LAB session 4, ALLOW_SITRACER) the SEAICE_TRACER.h fields with their last index
as the k axis (SITR_FIELDS). `seaice_fields` returns them all zero: the value of a common block no routine has written
(SEAICE_INIT_FIXED / SEAICE_INIT_VARIA then set what they set; SIMaskU/SIMaskV are written by no routine of the
B-grid build and stay zero). Only the fields of the options this port carries are present: a field the build
compiles but this list lacks raises (KeyError) at its first use.
"""

import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.pkg.seaice.seaice_params_h import SItrMaxNum, nITD

# SEAICE.h (:34-245) and SEAICE_GRID.h (:16-42) fields by the option that declares them (lane M4OFF: the C-grid build
# of offline_exf_seaice; the column's B-grid build has the same list as before)
FIELDS_ALL = ("AREA", "HEFF", "HSNOW", "UICE", "VICE", "DWATN", "uIceNm1", "vIceNm1",          # :34-45
              "d_HEFFbyNEG", "d_HSNWbyNEG", "saltWtrIce", "frWtrIce",                         # :213-215, :236-238
              "HEFFM", "SIMaskU", "SIMaskV")                                                 # SEAICE_GRID.h:16-19
FIELDS_CB = ("ETA", "etaZ", "ZETA", "zetaZ", "PRESS", "tensileStrFac", "e11", "e22", "e12", "deltaC",
             "FORCEX", "FORCEY", "PRESS0", "FORCEX0", "FORCEY0", "SEAICE_zMax", "SEAICE_zMin",   # :87-114
             "k1AtC", "k2AtC", "k1AtU", "k1AtV", "k2AtU", "k2AtV")                           # SEAICE_GRID.h:32-39
FIELDS_C = ("stressDivergenceX", "stressDivergenceY", "seaiceMassC", "seaiceMassU", "seaiceMassV",   # :67-71, :130-133
            "seaiceMaskU", "seaiceMaskV", "k1AtZ", "k2AtZ")                                  # SEAICE_GRID.h:21-25, :41-44
FIELDS_EVP = ("seaice_sigma1", "seaice_sigma2", "seaice_sigma12")                            # :73-84 (C-grid)
FIELDS_FD = ("uice_fd", "vice_fd")                                                           # :137-142 (C-grid)
FIELDS_BD = ("CbotC",)                                                                       # :145-149 (C-grid)
FIELDS_B = ("AMASS", "DAIRN", "uIceB", "vIceB", "WINDX", "WINDY", "GWATX", "GWATY", "UVM")   # :185-211, GRID.h:27-30
FIELDS_SAL = ("HSALT", "saltFluxAdjust")                                                     # :227-234
FIELDS_RLX = ("d_AREAbyRLX", "d_HEFFbyRLX")       # :219-225 #ifdef EXF_SEAICE_FRACTION (lane M4ADLAB session 3)
# SEAICE_TRACER.h:23-28 (#ifdef ALLOW_SITRACER, lane M4LAB session 4): _RL SItracer, SItrBucket
# (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,SItrMaxNum), SItrHEFF (...,5), SItrAREA (...,3); the last index is the FArray's
# k axis (SItracer(i,j,bi,bj,iTr) is SItracer[i, j, iTr])
SITR_FIELDS = (("SItracer", SItrMaxNum), ("SItrBucket", SItrMaxNum), ("SItrHEFF", 5), ("SItrAREA", 3))
# the B-grid list of the column build (kept: its gates read FIELDS)
FIELDS = FIELDS_ALL[:8] + FIELDS_CB[:17] + FIELDS_B[:8] + FIELDS_ALL[8:10] + FIELDS_SAL + FIELDS_ALL[10:] + (
    "UVM",) + FIELDS_CB[17:]


def field_names(cfg):
    """The SEAICE.h / SEAICE_GRID.h names this build declares (by its SEAICE_OPTIONS.h)."""
    f = lambda o: cfg.cpp.flag(o, "SEAICE_OPTIONS.h")                           # noqa: E731
    names = list(FIELDS_ALL)
    if f("SEAICE_CGRID") or f("SEAICE_BGRID_DYNAMICS"):
        names += FIELDS_CB
    if f("SEAICE_CGRID"):
        names += FIELDS_C
        if f("SEAICE_ALLOW_EVP"):
            names += FIELDS_EVP
        if f("SEAICE_ALLOW_FREEDRIFT"):
            names += FIELDS_FD
        if f("SEAICE_ALLOW_BOTTOMDRAG"):                                       # lane M4LAB session 3
            names += FIELDS_BD
    if f("SEAICE_BGRID_DYNAMICS"):
        names += FIELDS_B
    if f("SEAICE_VARIABLE_SALINITY"):
        names += FIELDS_SAL
    if cfg.cpp.flag("ALLOW_EXF") and cfg.cpp.flag("EXF_SEAICE_FRACTION", "EXF_OPTIONS.h"):
        names += FIELDS_RLX
    return tuple(names)


def xy(name, data, sz):
    """FArray of a (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, nSx, nSy) field from [tile, j, i] data."""
    return FArray(data, name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))


def xyn(name, data, sz, n):
    """FArray of a (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, n, nSx, nSy) field from [tile, n, j, i] data."""
    return FArray(data, name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy), k=(1, n))


def level(arr, n):
    """The 2-D FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy) of level n of a field with a k axis: what a Fortran call
    passing `fld(1-OLx,1-OLy,1,1,n)` (e.g. SItracer's iTr slab) hands to a 2-D dummy (lane M4LAB session 4)."""
    (_, ilo, ihi), (_, jlo, jhi) = arr.dims[:2]
    return FArray(arr.data[:, n - 1], arr.name, i=(ilo, ihi), j=(jlo, jhi))


def set_level(arr, n, lev):
    """`arr` with level n replaced by the 2-D FArray `lev` (every point, halos included)."""
    return FArray(arr.data.at[:, n - 1].set(lev.data), arr.name, tiled=arr.tiled, _dims=arr.dims)


def seaice_fields(cfg):
    """{name: FArray} of SEAICE.h / SEAICE_GRID.h, all zero (see the module docstring). Lane M4OFF: the field list
    follows the build's options (field_names); the C-grid build's JFNK/Krylov solver state (:165-181, compiled, used
    only with SEAICEuseJFNK/Krylov) is not carried."""
    if cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE.h: SEAICE_ITD (its fields) is not ported")
    sz = cfg.size
    T = sz.nSx*sz.nSy
    shape = (T, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx)
    f = {n: xy(n, jnp.zeros(shape), sz) for n in field_names(cfg)}
    n = nITD(cfg)
    f["TICES"] = xyn("TICES", jnp.zeros((T, n) + shape[1:]), sz, n)
    if cfg.cpp.flag("SEAICE_BGRID_DYNAMICS", "SEAICE_OPTIONS.h"):
        f["KGEO"] = xy("KGEO", jnp.zeros(shape, dtype=jnp.int32), sz)          # :210 INTEGER
    if cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h"):                    # SEAICE_TRACER.h:23-28 (session 4)
        for name, n in SITR_FIELDS:
            f[name] = xyn(name, jnp.zeros((T, n) + shape[1:]), sz, n)
    return f
