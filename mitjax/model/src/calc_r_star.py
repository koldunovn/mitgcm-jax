"""CALC_R_STAR: model/src/calc_r_star.F @63cdc0b (traced part) and its host-side WRITE / STOP part."""

import jax.numpy as jnp
import numpy as np
from mitjax.ops.fortran_minmax_host import MAX as MAX_host

from mitjax.eesupp.exch_rs import EXCH_XY_RL
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.safe import div, safe_div


def calc_r_star(etaFld, myTime, myIter, *, cfg, grid, params, state, ex):
    """CALC_R_STAR( etaFld, myTime, myIter, myThid )   @63cdc0b model/src/calc_r_star.F:10-343

    C     | SUBROUTINE CALC_R_STAR
    C     | o Calculate new column thickness & scaling factor for r*
    C     |   according to the surface r-position (Non-Lin Free-Surf)
    C     etaFld    :: current eta field used to update the hFactor
    C     myTime    :: current time in simulation
    C     myIter    :: current iteration number in simulation

    Returns (state, counters). `state`: the SURFACE.h /RSTAR_CHANGE/ fields the routine writes (rStarFacNm1C/W/S,
    rStarExpC/W/S, rStarFacC/W/S, rStarDhCDt/WDt/SDt). Reads (GRID.h) kSurfC/W/S, Ro_surf, R_low, recip_Rcol,
    rSurfW/S, rLowW/S, rA, recip_rAw, recip_rAs from `grid`; hFacInf, hFacSup, deltaTFreeSurf (traced),
    vectorInvariantMomentum, selectKEscheme, fluidIsAir (static) from `params` (ini_parms_dyn + ini_parms_rstar).

    `counters` (the part of :177-253 that cannot run inside jit [E§11]): per tile, in tile order on every device
    (`ex.all_tiles`, the fixed-order gather of mitjax/eesupp [E§7]): `icntc1`, `icntw`, `icnts`, `icntc2` (int32 [T],
    the counts of :183-199) and `maxhFacC` (float64 [T], the largest rStarFacC > hFacSup of the tile, 0. if none).
    The host decides the STOP of :201-243 and the warnings of :246-253 from them: `calc_r_star_host`. GO lane:
    `nzeroC`, `nzeroW`, `nzeroS` (int32 [T]) count the zero denominators of the guard at :306-311 (below),
    reported on the host by `calc_r_star_zero_report`.

    Ported (global_ocean.90x40x15, advect_xz/input.nlfs: select_rStar = 2): :79-97 copies; :100-110 rStarFacC;
    :111-140 area-weighted rStarFacW/S, or :141-166 the simple average when vectorInvariantMomentum and selectKEscheme
    = 1 or 3 (rStarAreaWeight, :64-68: a static switch; no M1 variant runs it, gated by the RSTAR replay harness);
    :183-199 counters; :259-260 exchanges; :298-313 rStarDh*Dt and rStarExp*. Raise: ALLOW_OBCS (:168-175
    OBCS_APPLY_R_STAR), W2_FILL_NULL_REGIONS (:268-296), fluidIsAir (:315-320, pStarFacK = rStarFacC**atm_kappa).
    PTRACERS lane: the ALLOW_AUTODIFF arm :321-328 (pStarFacK = 1. _d 0, tutorial_tracer_adjsens/code_ad).
    DEBUG_ENTER/LEAVE (:73, :336, debugMode): output only, not ported.

    Vectorisation: every loop computes each point from inputs of the routine or from arrays completed by an earlier
    loop (the counters read rStarFacW/S after :113-140, the second tile loop reads the exchanged rStarFac*), so each
    DO j / DO i nest is one vectorised assignment, in the Fortran's statement order.

    Guards (the two divisions without a Fortran IF of their own): :117-121, :131-135 divide by tmpfldW/S only where
    the Fortran's `kSurfW.LE.Nr` holds (safe_div on that mask, the ELSE value 1. elsewhere); :306-311 divide by the
    previous rStarFac (rStarExp*), which the Fortran never sees as zero on the M1 runs (0 zero denominators on every
    point incl. halos of every dumped step, mitjax/tests/test_rstar.py), so `div` gets the denominator itself where it
    is non-zero and 1. where it is zero (padding tiles only): the forward value of every lane the Fortran computes
    is unchanged and no 0/0 reaches a backward pass [E§6]."""
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    if cfg.cpp.ALLOW_OBCS:
        raise NotImplementedError("CALC_R_STAR: ALLOW_OBCS (OBCS_APPLY_R_STAR, calc_r_star.F:168-175) is not ported")
    if cfg.cpp.ALLOW_EXCH2 and cfg.cpp.flag("W2_FILL_NULL_REGIONS", "W2_OPTIONS.h"):
        raise NotImplementedError("CALC_R_STAR: W2_FILL_NULL_REGIONS (calc_r_star.F:268-296) is not ported")
    if params.fluidIsAir:
        raise NotImplementedError("CALC_R_STAR: fluidIsAir (pStarFacK, calc_r_star.F:315-320) is not ported")
    g = grid
    hFacInf, hFacSup, deltaTFreeSurf = params.hFacInf, params.hFacSup, params.deltaTFreeSurf

    # :64-68  rStarAreaWeight
    rStarAreaWeight = True
    if params.vectorInvariantMomentum and (params.selectKEscheme == 1 or params.selectKEscheme == 3):
        rStarAreaWeight = False

    rStarFacC, rStarFacW, rStarFacS = state.rStarFacC, state.rStarFacW, state.rStarFacS
    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    # :79-86  before updating rStarFacC/S/W save current fields
    rStarFacNm1C = state.rStarFacNm1C.at[iA, jA].set(rStarFacC[iA, jA])                    # :82
    rStarFacNm1S = state.rStarFacNm1S.at[iA, jA].set(rStarFacS[iA, jA])                    # :83
    rStarFacNm1W = state.rStarFacNm1W.at[iA, jA].set(rStarFacW[iA, jA])                    # :84
    # :90-97  copy rStarFacX -> rStarExpX
    rStarExpC = state.rStarExpC.at[iA, jA].set(rStarFacC[iA, jA])                          # :93
    rStarExpW = state.rStarExpW.at[iA, jA].set(rStarFacW[iA, jA])                          # :94
    rStarExpS = state.rStarExpS.at[iA, jA].set(rStarFacS[iA, jA])                          # :95

    # :99-110  Compute the new column thikness
    j = loop_j(0, sNy+1)
    i = loop_i(0, sNx+1)
    rStarFacC = rStarFacC.at[i, j].set(jnp.where(
        g.kSurfC[i, j] <= Nr,                                                               # :102
        (etaFld[i, j]+g.Ro_surf[i, j]-g.R_low[i, j])
        * g.recip_Rcol[i, j],                                                               # :103-105
        1.))                                                                                # :107
    if rStarAreaWeight:
        # :112-126  Area weighted average
        j = loop_j(1, sNy)
        i = loop_i(1, sNx+1)
        wet = g.kSurfW[i, j] <= Nr                                                          # :115
        tmpfldW = g.rSurfW[i, j] - g.rLowW[i, j]                                            # :116
        rStarFacW = rStarFacW.at[i, j].set(jnp.where(wet, safe_div(                         # :117-121
            (0.5*(etaFld[i-1, j]*g.rA[i-1, j]
                  + etaFld[i, j]*g.rA[i, j]
                  )*g.recip_rAw[i, j]
             + tmpfldW), tmpfldW, wet), 1.))                                                # :123
        # :127-140
        j = loop_j(1, sNy+1)
        i = loop_i(1, sNx)
        wet = g.kSurfS[i, j] <= Nr                                                          # :129
        tmpfldS = g.rSurfS[i, j] - g.rLowS[i, j]                                            # :130
        rStarFacS = rStarFacS.at[i, j].set(jnp.where(wet, safe_div(                         # :131-135
            (0.5*(etaFld[i, j-1]*g.rA[i, j-1]
                  + etaFld[i, j]*g.rA[i, j]
                  )*g.recip_rAs[i, j]
             + tmpfldS), tmpfldS, wet), 1.))                                                # :137
    else:
        # :142-166  Simple average
        j = loop_j(1, sNy)
        i = loop_i(1, sNx+1)
        wet = g.kSurfW[i, j] <= Nr                                                          # :145
        tmpfldW = g.rSurfW[i, j] - g.rLowW[i, j]                                            # :146
        rStarFacW = rStarFacW.at[i, j].set(jnp.where(wet, safe_div(                         # :147-149
            (0.5*(etaFld[i-1, j] + etaFld[i, j])
             + tmpfldW), tmpfldW, wet), 1.))                                                # :151
        j = loop_j(1, sNy+1)
        i = loop_i(1, sNx)
        wet = g.kSurfS[i, j] <= Nr                                                          # :157
        tmpfldS = g.rSurfS[i, j] - g.rLowS[i, j]                                            # :158
        rStarFacS = rStarFacS.at[i, j].set(jnp.where(wet, safe_div(                         # :159-161
            (0.5*(etaFld[i, j-1] + etaFld[i, j])
             + tmpfldS), tmpfldS, wet), 1.))                                                # :163
    # :177-199  Needs to do something when r* ratio is too small: count (the STOP is on the host)
    j = loop_j(1, sNy+1)
    i = loop_i(1, sNx+1)
    cC, cW, cS = rStarFacC[i, j], rStarFacW[i, j], rStarFacS[i, j]
    icntc1 = jnp.sum(cC < hFacInf, axis=(1, 2), dtype=jnp.int32)                            # :185-187
    icntw = jnp.sum(cW < hFacInf, axis=(1, 2), dtype=jnp.int32)                             # :188-190
    icnts = jnp.sum(cS < hFacInf, axis=(1, 2), dtype=jnp.int32)                             # :191-193
    big = cC > hFacSup                                                                      # :194
    icntc2 = jnp.sum(big, axis=(1, 2), dtype=jnp.int32)                                     # :195
    # :196 maxhFacC = max(rStarFacC(i,j,bi,bj),maxhFacC), starting from 0. _d 0 (:62): every value that reaches the
    # MAX is > hFacSup (2.0 in both r* variants) and finite (a NaN fails .GT.), so ties of +0/-0 and NaN cannot
    # occur and the running MAX of the tile is the plain maximum of those values (site :196, p = "a" in the minmax
    # table: no effect on such values).
    # MINMAX-RAW: positive finite values only (above); the per-tile partial, the running max over tiles is on the host
    maxhFacC = jnp.max(jnp.where(big, cC, 0.), axis=(1, 2))
    counters = dict(icntc1=ex.all_tiles(icntc1), icntw=ex.all_tiles(icntw), icnts=ex.all_tiles(icnts),
                    icntc2=ex.all_tiles(icntc2), maxhFacC=ex.all_tiles(maxhFacC))

    rStarFacC = EXCH_XY_RL(rStarFacC, ex=ex)                                                # :259
    rStarFacW, rStarFacS = EXCH_UV_XY_RL(rStarFacW, rStarFacS, False, ex=ex)                # :260

    # :298-313  2nd bi,bj loop
    rStarDhCDt = state.rStarDhCDt.at[iA, jA].set((rStarFacC[iA, jA]
                                                  - rStarExpC[iA, jA])/deltaTFreeSurf)      # :300-301
    rStarDhWDt = state.rStarDhWDt.at[iA, jA].set((rStarFacW[iA, jA]
                                                  - rStarExpW[iA, jA])/deltaTFreeSurf)      # :302-303
    rStarDhSDt = state.rStarDhSDt.at[iA, jA].set((rStarFacS[iA, jA]
                                                  - rStarExpS[iA, jA])/deltaTFreeSurf)      # :304-305
    # GO lane (Nikolay 2026-10-01): the guard of :306-311 is accepted WITH a host-side count of the zero denominators
    # per step (counters nzeroC/W/S per tile, every point of the loop :298-313 incl. halos; padding tiles are dropped
    # by ex.all_tiles), reported by calc_r_star_zero_report so that a real occurrence is visible
    nzeroC = jnp.sum(rStarExpC[iA, jA] == 0., axis=(1, 2), dtype=jnp.int32)
    nzeroW = jnp.sum(rStarExpW[iA, jA] == 0., axis=(1, 2), dtype=jnp.int32)
    nzeroS = jnp.sum(rStarExpS[iA, jA] == 0., axis=(1, 2), dtype=jnp.int32)
    counters.update(nzeroC=ex.all_tiles(nzeroC), nzeroW=ex.all_tiles(nzeroW), nzeroS=ex.all_tiles(nzeroS))
    rStarExpC = rStarExpC.at[iA, jA].set(div(rStarFacC[iA, jA],
                                             _nonzero(rStarExpC[iA, jA])))                  # :306-307
    rStarExpW = rStarExpW.at[iA, jA].set(div(rStarFacW[iA, jA],
                                             _nonzero(rStarExpW[iA, jA])))                  # :308-309
    rStarExpS = rStarExpS.at[iA, jA].set(div(rStarFacS[iA, jA],
                                             _nonzero(rStarExpS[iA, jA])))                  # :310-311

    state = state.replace(rStarFacNm1C=rStarFacNm1C, rStarFacNm1W=rStarFacNm1W, rStarFacNm1S=rStarFacNm1S,
                          rStarExpC=rStarExpC, rStarExpW=rStarExpW, rStarExpS=rStarExpS,
                          rStarFacC=rStarFacC, rStarFacW=rStarFacW, rStarFacS=rStarFacS,
                          rStarDhCDt=rStarDhCDt, rStarDhWDt=rStarDhWDt, rStarDhSDt=rStarDhSDt)
    if cfg.cpp.ALLOW_AUTODIFF:                         # :321-328 (PTRACERS lane, tutorial_tracer_adjsens/code_ad)
        # ELSE (not fluidIsAir, refused above): pStarFacK = 1. _d 0 on every point
        state = state.replace(pStarFacK=state.pStarFacK.at[iA, jA].set(1.0))
    return state, counters


def EXCH_UV_XY_RL(uPhi, vPhi, withSigns, *, ex):
    """EXCH_UV_XY_RL( uPhi, vPhi, withSigns, myThid ) (eesupp/src/exch_uv_xy_rx.template) on FArrays: the probed
    exchange of mitjax/eesupp. A local front-end until mitjax/eesupp/exch_rs.py (core lane) has one."""
    u, v = ex.EXCH_UV_XY_RL(uPhi.data, vPhi.data, withSigns)
    return (FArray(u, uPhi.name, tiled=uPhi.tiled, _dims=uPhi.dims),
            FArray(v, vPhi.name, tiled=vPhi.tiled, _dims=vPhi.dims))


def _nonzero(den):
    """The denominator itself where it is non-zero, 1. where it is exactly zero (the guard before `div`, docstring)."""
    return jnp.where(den != 0., den, 1.)


def calc_r_star_host(counters, myIter, numbWrite, *, cfg, myThid=1):
    """The WRITE / STOP part of CALC_R_STAR (calc_r_star.F:59-60, :201-253), on the host from `counters`.

    Loops over the tiles in the Fortran order (bj outer, bi inner = tile order, one thread). For each tile: if
    icntc1+icnts+icntw > 0 (:201), the run STOPs ('ABNORMAL END: S/R CALC_R_STAR', :242) after the tile's warning
    lines: raises RuntimeError with the lines the Fortran writes to errorMessageUnit for that tile (:228-241; the
    per-point ' fail at i,j=' lines :203-227 need the values before the exchange and are not reproduced). Else, if
    icntc2 > 0 and numbWrite <= numbWrMax (:246; numbWrMax = Nx*Ny, :60; numbWrite a SAVEd counter, DATA 0, :59):
    numbWrite + 1 and the two warning lines (:248-252), with maxhFacC the running MAX over the tiles so far (:62,
    :196: initialised once per call, not per tile). Returns (lines, numbWrite): the caller keeps numbWrite across
    calls and writes the lines to errorMessageUnit (STDERR)."""
    sz = cfg.size
    nSx = sz.nSx
    numbWrMax = sz.sNx*sz.nSx*sz.nPx * sz.sNy*sz.nSy*sz.nPy                     # :60  Nx*Ny (SIZE.h)
    c = {k: np.asarray(v) for k, v in counters.items()}
    lines = []
    maxhFacC = 0.                                                               # :62
    for t in range(c["icntc1"].shape[0]):
        bi, bj = t % nSx + 1, t // nSx + 1
        if c["icntc2"][t] > 0:
            maxhFacC = MAX_host(float(c["maxhFacC"][t]), maxhFacC, p="a")      # :196 (winner from the oracle site table)
        n1, nw, ns = int(c["icntc1"][t]), int(c["icntw"][t]), int(c["icnts"][t])
        if n1 + ns + nw > 0:                                                    # :201
            if n1 > 0:
                lines.append(f"WARNING: r*FacC < hFacInf at{n1:8d} pts : bi,bj,Thid,Iter="
                             f"{bi:4d}{bj:4d}{myThid:4d}{myIter:10d}")                       # :228-231
            if nw > 0:
                lines.append(f"WARNING: r*FacW < hFacInf at{nw:8d} pts : bi,bj,Thid,Iter="
                             f"{bi:4d}{bj:4d}{myThid:4d}{myIter:10d}")                       # :232-235
            if ns > 0:
                lines.append(f"WARNING: r*FacS < hFacInf at{ns:8d} pts : bi,bj,Thid,Iter="
                             f"{bi:4d}{bj:4d}{myThid:4d}{myIter:10d}")                       # :236-239
            lines.append("STOP in CALC_R_STAR : too SMALL rStarFac[C,W,S] !")                # :240-241
            raise RuntimeError("ABNORMAL END: S/R CALC_R_STAR\n" + "\n".join(lines))         # :242
        n2 = int(c["icntc2"][t])
        if n2 > 0 and numbWrite <= numbWrMax:                                   # :246
            numbWrite += 1                                                      # :247
            lines.append(f"WARNING: r*FacC > hFacSup at{n2:8d} pts : bi,bj,Thid,Iter="
                         f"{bi:4d}{bj:4d}{myThid:4d}{myIter:10d}")                           # :248-250
            lines.append(f"WARNING: max(hFacC) is {_e14_6(maxhFacC)}")                      # :251-252
    return lines, numbWrite


def calc_r_star_zero_report(counters, myIter):
    """The host-side report of the accepted guard at calc_r_star.F:306-311 (GO lane, Nikolay 2026-10-01): the number
    of zero denominators rStarExpC/W/S of this call, summed over the real tiles in tile order, from `counters`
    (nzeroC, nzeroW, nzeroS). Returns (n, lines): n = (nC, nW, nS); `lines` is empty when every count is 0 (the
    Fortran writes nothing there either), else one line naming the guard, the counts and the iteration (the Fortran
    would have divided by zero at those points; the port divides by 1. instead). Not Fortran output: the driver
    writes the lines to STDERR, so a run whose STDERR matches the oracle's had no such point."""
    n = tuple(int(np.sum(np.asarray(counters[k]))) for k in ("nzeroC", "nzeroW", "nzeroS"))
    if sum(n) == 0:
        return n, []
    return n, [f"MITJAX GUARD calc_r_star.F:306-311: zero denominator rStarExp[C,W,S] at{n[0]:8d}{n[1]:8d}{n[2]:8d}"
               f" pts : Iter={myIter:10d}"]


def _e14_6(x):
    """Fortran E14.6 (gfortran): 0.ddddddE+xx right-justified in 14 columns."""
    if x == 0.:
        return f"{'0.000000E+00':>14s}"
    m, e = f"{x:.5e}".split("e")
    exp = int(e) + 1
    digits = m.replace(".", "").replace("-", "")
    s = ("-" if x < 0 else "") + "0." + digits[:6].ljust(6, "0") + f"E{exp:+03d}"
    return f"{s:>14s}"
