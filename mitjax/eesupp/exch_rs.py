"""Exchanges called by the model routines on Fortran-index arrays (FArray, mitjax/farray.py): the `_RS` exchanges and
FArray front-ends of the `_RL` exchanges (plan Tasks 10-11).

`_RS` exchanges: in every M1 build CPP_EEOPTIONS.h defines REAL4_IS_SLOW, so `_RS` is `Real*8`
(eesupp/inc/CPP_EEMACROS.h:122-124 @63cdc0b, `#define _RS Real*8`) and EXCH_*_RS are the RL instances of the same
templates (eesupp/src/exch_*_rx.template): each wrapper calls the probed RL exchange of mitjax/eesupp (Task 7b,
mitjax/eesupp/exchange.py) on the FArray's storage and returns an FArray with the same declaration. A build without
REAL4_IS_SLOW (`_RS` = Real*4) raises (`check_rs_is_real8`).

`_XYZ` exchanges: exch_xyz_rx.template is exch_xy_rx.template with `Nr` levels instead of 1 (the only differences: the
declaration of the level dimension and `EXCH2_3D_RX( phi, Nr, ...)` vs `( phi, 1, ...)`, exch_xyz_rx.template vs
exch_xy_rx.template), and exch_uv_xyz_rx.template is exch_uv_xy_rx.template with `myNz = Nr`; the probed maps act on
each level of a [tile, k, j, i] array alike, so the XYZ exchanges apply the probed XY map to every level.

Owner: M1 core lane (Task 10 follow-up; thin wrappers over lane B's exchanges, no exchange logic of their own).
"""

from mitjax.farray import FArray


def _rewrap(data, like):
    return FArray(data, like.name, tiled=like.tiled, _dims=like.dims)


def check_rs_is_real8(cfg):
    """`_RS` is Real*8 only if CPP_EEOPTIONS.h defines REAL4_IS_SLOW (CPP_EEMACROS.h:122-128)."""
    if not cfg.cpp.flag("REAL4_IS_SLOW", "CPP_EEOPTIONS.h"):
        raise NotImplementedError("_RS = Real*4 (REAL4_IS_SLOW undefined): RS exchanges are not ported")


# ---------------------------------------------------------------------------------------------------------------- RS

def EXCH_XY_RS(phi, *, ex):
    """EXCH_XY_RS( phi, myThid ): eesupp/src/exch_xy_rx.template with _RX = _RS = Real*8 = the RL exchange."""
    return _rewrap(ex.EXCH_XY_RL(phi.data), phi)


def EXCH_XYZ_RS(phi, *, ex):
    """EXCH_XYZ_RS( phi, myThid ): eesupp/src/exch_xyz_rx.template with _RX = _RS = Real*8, the XY exchange on each
    of the Nr levels (as EXCH_XYZ_RL below; M3 Task 30: RBCS_INIT_FIXED, RBCS_FIELDS_LOAD)."""
    if phi.data.ndim != 4:
        raise ValueError(f"EXCH_XYZ_RS: {phi.name} is not a 3-D field")
    return _rewrap(ex.EXCH_XY_RL(phi.data), phi)


def EXCH_UV_XY_RS(uPhi, vPhi, withSigns, *, ex):
    """EXCH_UV_XY_RS( uPhi, vPhi, withSigns, myThid ): eesupp/src/exch_uv_xy_rx.template, _RX = _RS = Real*8."""
    u, v = ex.EXCH_UV_XY_RL(uPhi.data, vPhi.data, withSigns)
    return _rewrap(u, uPhi), _rewrap(v, vPhi)


def EXCH_UV_XYZ_RS(uPhi, vPhi, withSigns, *, ex):
    """EXCH_UV_XYZ_RS( uPhi, vPhi, withSigns, myThid ): eesupp/src/exch_uv_xyz_rx.template, _RX = _RS = Real*8.

    The template is exch_uv_xy_rx.template with myNz = Nr instead of 1 (the two differ only there: the declaration
    of the Nr dimension, `withSigns, Nr` vs `withSigns, 1` at :55-61, `myNz = Nr` at :70), and EXCH1_RX /
    EXCH2_UV_3D_RX repeat the same per-level operation for k = 1..myNz (pkg/exch2/exch2_uv_3d_rx.template:88-108),
    so the probed 2-D map EXCH_UV_XY_RL is applied to every level (mitjax/eesupp's maps act per level on [tile, k, j,
    i] arrays)."""
    u, v = ex.EXCH_UV_XY_RL(uPhi.data, vPhi.data, withSigns)
    return _rewrap(u, uPhi), _rewrap(v, vPhi)


def EXCH_UV_AGRID_3D_RS(uPhi, vPhi, withSigns, myNz, *, ex):
    """EXCH_UV_AGRID_3D_RS( uPhi, vPhi, withSigns, myNz, myThid ): eesupp/src/exch_uv_agrid_3d_rx.template with
    _RX = _RS = Real*8 (INI_CURVILINEAR_GRID: dxF/dyF, angleSinC/angleCosC; lane B, Task 22). A 2-D field is
    myNz = 1."""
    _check_nz("EXCH_UV_AGRID_3D_RS", (uPhi, vPhi), myNz)
    u, v = ex.EXCH_UV_AGRID_3D_RL(uPhi.data, vPhi.data, withSigns)
    return _rewrap(u, uPhi), _rewrap(v, vPhi)


def EXCH_UV_BGRID_3D_RS(uPhi, vPhi, withSigns, myNz, *, ex):
    """EXCH_UV_BGRID_3D_RS( uPhi, vPhi, withSigns, myNz, myThid ): eesupp/src/exch_uv_bgrid_3d_rx.template, _RX = _RS
    = Real*8 (INI_CURVILINEAR_GRID: dxV/dyU)."""
    _check_nz("EXCH_UV_BGRID_3D_RS", (uPhi, vPhi), myNz)
    u, v = ex.EXCH_UV_BGRID_3D_RL(uPhi.data, vPhi.data, withSigns)
    return _rewrap(u, uPhi), _rewrap(v, vPhi)


def EXCH_Z_3D_RS(phi, myNz, *, ex):
    """EXCH_Z_3D_RS( phi, myNz, myThid ): eesupp/src/exch_z_3d_rx.template, _RX = _RS = Real*8 (INI_CURVILINEAR_GRID:
    xG, yG, rAz)."""
    _check_nz("EXCH_Z_3D_RS", (phi,), myNz)
    return _rewrap(ex.EXCH_Z_3D_RL(phi.data), phi)


def _check_nz(name, arrays, myNz):
    for a in arrays:
        nz = a.data.shape[1] if a.data.ndim == 4 else 1
        if nz != myNz:
            raise ValueError(f"{name}: myNz = {myNz} does not match {a.name} ({nz} levels)")


# ---------------------------------------------------------------------------------------------------------------- RL

def EXCH_UV_XY_RL(uPhi, vPhi, withSigns, *, ex):
    """EXCH_UV_XY_RL( uPhi, vPhi, withSigns, myThid ) (eesupp/src/exch_uv_xy_rl.F from exch_uv_xy_rx.template; lane
    M4OFF: the sea-ice C-grid fields)."""
    u, v = ex.EXCH_UV_XY_RL(uPhi.data, vPhi.data, withSigns)
    return _rewrap(u, uPhi), _rewrap(v, vPhi)


def EXCH_XY_RL(phi, *, ex):
    """EXCH_XY_RL( phi, myThid ) (eesupp/src/exch_xy_rl.F from exch_xy_rx.template)."""
    return _rewrap(ex.EXCH_XY_RL(phi.data), phi)


def EXCH_XYZ_RL(phi, *, ex):
    """EXCH_XYZ_RL( phi, myThid ) = _EXCH_XYZ_RL (exch_xyz_rx.template: the XY exchange on each of the Nr levels)."""
    if phi.data.ndim != 4:
        raise ValueError(f"EXCH_XYZ_RL: {phi.name} is not a 3-D field")
    return _rewrap(ex.EXCH_XY_RL(phi.data), phi)


def EXCH_UV_XYZ_RL(uPhi, vPhi, withSigns, *, ex):
    """EXCH_UV_XYZ_RL( uPhi, vPhi, withSigns, myThid ) (exch_uv_xyz_rx.template: the UV XY exchange per level)."""
    u, v = ex.EXCH_UV_XY_RL(uPhi.data, vPhi.data, withSigns)
    return _rewrap(u, uPhi), _rewrap(v, vPhi)


def EXCH_3D_RL(phi, myNz, *, ex):
    """EXCH_3D_RL( phi, myNz, myThid ) (eesupp/src/exch_3d_rx.template; myNz checked against the array)."""
    if phi.data.shape[1] != myNz:
        raise ValueError(f"EXCH_3D_RL: {phi.name} has {phi.data.shape[1]} levels, myNz = {myNz}")
    return _rewrap(ex.EXCH_3D_RL(phi.data), phi)


def EXCH_UV_3D_RL(uPhi, vPhi, withSigns, myNz, *, ex):
    """EXCH_UV_3D_RL( uPhi, vPhi, withSigns, myNz, myThid ) (eesupp/src/exch_uv_3d_rx.template)."""
    if uPhi.data.shape[1] != myNz or vPhi.data.shape[1] != myNz:
        raise ValueError(f"EXCH_UV_3D_RL: myNz = {myNz} does not match {uPhi.name}, {vPhi.name}")
    u, v = ex.EXCH_UV_3D_RL(uPhi.data, vPhi.data, withSigns)
    return _rewrap(u, uPhi), _rewrap(v, vPhi)
