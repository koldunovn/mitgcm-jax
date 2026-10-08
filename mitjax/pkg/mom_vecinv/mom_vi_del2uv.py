"""pkg/mom_vecinv/mom_vi_del2uv.F: del^2 of (u,v) from hDiv and vort3 (MOM_VI_DEL2UV)."""

import jax.numpy as jnp

from mitjax.eesupp.fill_cs_corner_tr_rl import fill_cs_corner_tr_rl
from mitjax.farray import FArray, loop_i, loop_j

_OPT = "MOM_VECINV_OPTIONS.h"       # mom_vi_del2uv.F:1


def _fill(fill4dir, trFld, cfg, params, w2):
    """CALL FILL_CS_CORNER_TR_RL( fill4dir, .FALSE., trFld, bi,bj, myThid ) on every tile (mitjax/eesupp, lane B):
    the corner flags of each tile from the W2 tile view, as fill_cs_corner_tr_rl.F:75-82 derives them."""
    if not cfg.cpp.flag("ALLOW_EXCH2", _OPT):
        raise NotImplementedError("MOM_VI_DEL2UV: FILL_CS_CORNER_TR_RL on the cube without ALLOW_EXCH2 is not ported")
    isW, isE = jnp.asarray(w2.exch2_isWedge) == 1, jnp.asarray(w2.exch2_isEedge) == 1
    isS, isN = jnp.asarray(w2.exch2_isSedge) == 1, jnp.asarray(w2.exch2_isNedge) == 1
    corners = jnp.stack([isW & isS, isE & isS, isW & isN, isE & isN], axis=1)       # SW, SE, NW, NE
    sz = cfg.size
    new = fill_cs_corner_tr_rl(fill4dir, False, trFld.data, corners, params.useCubedSphereExchange,
                               sNx=sz.sNx, sNy=sz.sNy, OLx=sz.OLx, OLy=sz.OLy)
    return FArray(new, trFld.name, tiled=trFld.tiled, _dims=trFld.dims)


def mom_vi_del2uv(k, hDiv, vort3, hFacZ, del2u, del2v, *, cfg, grid, params, w2=None):
    """MOM_VI_DEL2UV(bi,bj,k, hDiv,vort3,hFacZ, del2u,del2v, myThid)   @63cdc0b pkg/mom_vecinv/mom_vi_del2uv.F:3-127

    C     Calculate del^2 of (u,v) in terms of hDiv and vort3

    Returns (del2u, del2v, hDiv): hDiv is returned because FILL_CS_CORNER_TR_RL overwrites its corner halos on the
    cube (:79-83, :104-108; KERNEL_GUIDE: arguments the Fortran overwrites by reference are returned). The
    commented-out alternatives (:33-74) are not ported. #ifdef ALLOW_OBCS (:95-97, :120-122) is not ported (raises).
    On the cube (`params.useCubedSphereExchange`, static) the two FILL_CS_CORNER_TR_RL calls are lane B's eesupp
    port; `w2` is the W2 tile view of mom_calc_relvort3 (the one addition to the Fortran signature), from which the
    corner flags of each tile follow. The point loops run on the whole (i,j) range.
    """
    if cfg.cpp.flag("ALLOW_OBCS", _OPT):
        raise NotImplementedError("MOM_VI_DEL2UV: ALLOW_OBCS (maskInW/S) is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    recip_dxC, recip_dyC, recip_dxG, recip_dyG = grid.recip_dxC, grid.recip_dyC, grid.recip_dxG, grid.recip_dyG
    recip_hFacW, recip_hFacS, maskW, maskS = grid.recip_hFacW, grid.recip_hFacS, grid.maskW, grid.maskS
    recip_deepFacC = grid.recip_deepFacC

#     - bi-harmonic viscosity :
    if params.useCubedSphereExchange:                               # :79-83
        hDiv = _fill(1, hDiv, cfg, params, w2)
    j = loop_j(2-OLy, sNy+OLy-1)                                    # :87-99
    i = loop_i(2-OLx, sNx+OLx-1)
    del2u = del2u.at[i, j].set(
        ((hDiv[i, j] - hDiv[i-1, j])*recip_dxC[i, j]
         -recip_hFacW[i, j, k]*
         (hFacZ[i, j+1]*vort3[i, j+1] - hFacZ[i, j]*vort3[i, j])
         *recip_dyG[i, j]
         )*maskW[i, j, k]*recip_deepFacC[k])

    if params.useCubedSphereExchange:                               # :104-108
        hDiv = _fill(2, hDiv, cfg, params, w2)
    del2v = del2v.at[i, j].set(                                     # :112-124
        ((hDiv[i, j] - hDiv[i, j-1])*recip_dyC[i, j]
         +recip_hFacS[i, j, k]*
         (hFacZ[i+1, j]*vort3[i+1, j] - hFacZ[i, j]*vort3[i, j])
         *recip_dxG[i, j]
         )*maskS[i, j, k]*recip_deepFacC[k])

    return del2u, del2v, hDiv
