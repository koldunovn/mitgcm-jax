"""CALC_SURF_DR: model/src/calc_surf_dr.F @63cdc0b (lane B, plan Task 25: adjustment.cs-32x32x1/input.nlfs)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_UV_XY_RS, EXCH_XY_RS
from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MIN


def _sl(lo, hi, OL):
    """Storage slice of the Fortran index range lo..hi of a dimension declared 1-OL:..."""
    return slice(lo - 1 + OL, hi + OL)


def _at_ks(a3, ks, Nr):
    """a3(i,j,ks(i,j)) of a [tile, k, j, i] array cut to the same (j, i) range as ks; ks clamped into 1..Nr (the value
    is used only where ks <= Nr)."""
    kk = jnp.clip(ks.astype(jnp.int32) - 1, 0, Nr - 1)       # MINMAX-RAW: integer index clamp, used where ks <= Nr
    return jnp.take_along_axis(a3, kk[:, None], axis=1)[:, 0], kk


def calc_surf_dr(etaFld, myTime, myIter, *, cfg, grid, params, state, ex):
    """CALC_SURF_DR( etaFld, myTime, myIter, myThid )   @63cdc0b model/src/calc_surf_dr.F:7-233 (NONLIN_FRSURF)

    C     | SUBROUTINE CALC_SURF_DR
    C     | o Calculate the new surface level thickness according to
    C     |   the surface r-position (Non-Linear Free-Surf)
    C     | o take decision if needed to switch k-index of surface level

    Returns (state, adjust): the State with hFac_surfNm1C/W/S = hFac_surfC/W/S (:99-105) and the new hFac_surfC
    (j = 0..sNy+1, i = 0..sNx+1, :107-151), hFac_surfW (j = 1..sNy, i = 1..sNx+1, :153-163), hFac_surfS
    (j = 1..sNy+1, i = 1..sNx, :165-175); points with kSurf > Nr keep their values. rSurftmp = Ro_surf + etaFld,
    replaced by Rmin_surf where below it (:109-140). `adjust` = (int32 [nTiles] number of interior points so
    clipped per tile, [nTiles, sNy+2, sNx+2] their rA*(Rmin_surf - rSurftmp), 0. elsewhere), both gathered over every
    real tile in tile order (ex.all_tiles: identity on one device, the same on every device under shard_map), for the
    host's SURF_ADJUSTMENT line (:200-212: _GLOBAL_SUM_RL of the counts, STDOUT when >= 1 point); the numbWrite-limited WARNING lines (errorMessageUnit, :115-133, :143-150) are not written.
    Then _EXCH_XY_RS( hFac_surfC ) and EXCH_UV_XY_RS( hFac_surfW, hFac_surfS, .FALSE. ) (:214-215). The C statements of a
    point read only that point; the W / S statements read rSurftmp at the point and its i-1 / j-1 neighbour, which
    the C loop has finished: vectorised. MIN(hhm,hhp) at :160 / :180 with the oracle's winner (p="b";
    $MJX_REFERENCE/minmax_sites). OBCS_APPLY_SURF_DR (useOBCS) raises."""
    if cfg.cpp.ALLOW_OBCS and cfg.use_flag("useOBCS"):
        raise NotImplementedError("CALC_SURF_DR: OBCS_APPLY_SURF_DR is not ported")
    sz = cfg.size
    Nr, sNx, sNy, OLx, OLy = sz.Nr, sz.sNx, sz.sNy, sz.OLx, sz.OLy
    g = grid
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    out = {"hFac_surfNm1C": state.hFac_surfNm1C.at[i, j].set(state.hFac_surfC[i, j]),       # :101
           "hFac_surfNm1S": state.hFac_surfNm1S.at[i, j].set(state.hFac_surfS[i, j]),       # :102
           "hFac_surfNm1W": state.hFac_surfNm1W.at[i, j].set(state.hFac_surfW[i, j])}       # :103
    rdrF = jnp.asarray(g.recip_drF.data)

    # :107-151  C points, j = 0..sNy+1, i = 0..sNx+1 (rSurftmp(0:sNx+1,0:sNy+1) is local)
    jc, ic = _sl(0, sNy+1, OLy), _sl(0, sNx+1, OLx)
    eta = jnp.asarray(etaFld.data)[:, jc, ic]
    Ro = jnp.asarray(g.Ro_surf.data)[:, jc, ic]
    rSurftmp = Ro + eta                                                         # :109
    ks = jnp.asarray(g.kSurfC.data)[:, jc, ic]                                  # :110
    wet = ks <= Nr                                                              # :111
    Rmin = jnp.asarray(state.Rmin_surf.data)[:, jc, ic]
    below = wet & (rSurftmp < Rmin)                                             # :112
    interior = jnp.zeros(rSurftmp.shape, bool).at[:, 1:sNy+1, 1:sNx+1].set(True)   # i = 1..sNx, j = 1..sNy
    rA = jnp.asarray(g.rA.data)[:, jc, ic]
    # :134-138 per tile; the _GLOBAL_SUM_RLs of :200-203 on the host (drivers/run._surf_adjust_lines) from every
    # real tile in tile order (ex.all_tiles, the fixed-order gather of mitjax/eesupp [E§7], as CALC_R_STAR's counters)
    adjust = (ex.all_tiles(jnp.sum(below & interior, axis=(1, 2), dtype=jnp.int32)),
              ex.all_tiles(jnp.where(below & interior, rA*(Rmin - rSurftmp), 0.)))
    rSurftmp = jnp.where(below, Rmin, rSurftmp)                                 # :139
    h0, kk = _at_ks(jnp.asarray(g.h0FacC.data)[:, :, jc, ic], ks, Nr)
    mC, _ = _at_ks(jnp.asarray(g.maskC.data)[:, :, jc, ic], ks, Nr)
    hC = h0 + (rSurftmp - Ro)*rdrF[kk]*mC                                       # :141-143
    old = jnp.asarray(state.hFac_surfC.data)[:, jc, ic]
    out["hFac_surfC"] = state.hFac_surfC.at[loop_i(0, sNx+1), loop_j(0, sNy+1)].set(jnp.where(wet, hC, old))

    # :153-163  W points, j = 1..sNy, i = 1..sNx+1: hhm = rSurftmp(i-1,j), hhp = rSurftmp(i,j)
    jw, iw = _sl(1, sNy, OLy), _sl(1, sNx+1, OLx)
    hhm, hhp = rSurftmp[:, 1:sNy+1, 0:sNx+1], rSurftmp[:, 1:sNy+1, 1:sNx+2]
    ksW = jnp.asarray(g.kSurfW.data)[:, jw, iw]
    h0W, kw = _at_ks(jnp.asarray(g.h0FacW.data)[:, :, jw, iw], ksW, Nr)
    mW, _ = _at_ks(jnp.asarray(g.maskW.data)[:, :, jw, iw], ksW, Nr)
    hW = h0W + (MIN(hhm, hhp, p="b") - jnp.asarray(g.rSurfW.data)[:, jw, iw])*rdrF[kw]*mW   # :160
    old = jnp.asarray(state.hFac_surfW.data)[:, jw, iw]
    out["hFac_surfW"] = state.hFac_surfW.at[loop_i(1, sNx+1), loop_j(1, sNy)].set(jnp.where(ksW <= Nr, hW, old))

    # :165-175  S points, j = 1..sNy+1, i = 1..sNx: hhm = rSurftmp(i,j-1), hhp = rSurftmp(i,j)
    js, is_ = _sl(1, sNy+1, OLy), _sl(1, sNx, OLx)
    hhm, hhp = rSurftmp[:, 0:sNy+1, 1:sNx+1], rSurftmp[:, 1:sNy+2, 1:sNx+1]
    ksS = jnp.asarray(g.kSurfS.data)[:, js, is_]
    h0S, kS = _at_ks(jnp.asarray(g.h0FacS.data)[:, :, js, is_], ksS, Nr)
    mS, _ = _at_ks(jnp.asarray(g.maskS.data)[:, :, js, is_], ksS, Nr)
    hS = h0S + (MIN(hhm, hhp, p="b") - jnp.asarray(g.rSurfS.data)[:, js, is_])*rdrF[kS]*mS   # :180
    old = jnp.asarray(state.hFac_surfS.data)[:, js, is_]
    out["hFac_surfS"] = state.hFac_surfS.at[loop_i(1, sNx), loop_j(1, sNy+1)].set(jnp.where(ksS <= Nr, hS, old))
    out["hFac_surfC"] = EXCH_XY_RS(out["hFac_surfC"], ex=ex)                    # :214
    out["hFac_surfW"], out["hFac_surfS"] = EXCH_UV_XY_RS(out["hFac_surfW"], out["hFac_surfS"], False, ex=ex)   # :215
    return state.replace(**out), adjust
