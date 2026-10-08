"""MON_KE   @63cdc0b pkg/monitor/mon_ke.F:8-339"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.ops.fortran_minmax_host import max_chain
from mitjax.pkg.monitor.mon_out import mon_out_rl
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.monitor_h import (global_max_rl, global_sum_rl_scalar, global_sum_tile_rl, mon_foot_max,
                                          mon_foot_mean, mon_foot_vol, mon_string_none, tile_sums)


def mon_ke(myIter, *, cfg, grid, state, mon, ex=None, params=None):
    """MON_KE( myIter, myThid )

    C     Calculates stats for Kinetic Energy, (barotropic) Potential Energy
    C                      and total Angular Momentum

    Inputs: state.uVel, state.vVel (DYNVARS.h), state.etaN, state.phi0surf (FFIELDS.h); grid.rA, recip_rA, dxC, dyC,
    dxG, dyG, hFacC, hFacW, hFacS, recip_hFacC, maskInC, Bo_surf (SURFACE.h), drF, deepFac2C, deepFac2F, rhoFacC.

    Per tile (:61-152): tileVol, tileVlAv and tileMean are chains over k, j, i (the k loop is vectorised for the
    per-point expressions, which are independent, and the sums keep the k, j, i order through mitjax/eesupp);
    tilePEav is a chain over j, i. theMax (from `theMax=0.`, :56) and numPnts run on across tiles: a MAX chain in
    which the new value wins ties and NaN ($MJX_REFERENCE/minmax_sites) and an exact count. Not ported (raise): the non-hydrostatic term (#ifdef
    ALLOW_NONHYDROSTATIC with nonHydrostatic, :107-120). Lane B (Task 25, solid-body.cs-32x32x1): the total angular
    momentum (mon_output_AM = fluidIsAir .AND. useCoriolis, :166-322; `_angular_momentum`, with `params`)."""
    sNx, sNy, Nr = cfg.sNx, cfg.sNy, cfg.Nr
    g, s = grid, state
    if cfg.ALLOW_NONHYDROSTATIC and cfg.nonHydrostatic:
        raise NotImplementedError("MON_KE: nonHydrostatic KE term (mon_ke.F:107-120) is not ported")
    numPnts = np.float64(0.0)                                       # :54-59
    theVol = np.float64(0.0)
    theMax = np.float64(0.0)
    theMean = np.float64(0.0)
    theVolMean = np.float64(0.0)
    potEnMean = np.float64(0.0)

    k, j, i = loops_kji((1, Nr), (1, sNy), (1, sNx))                # :67-130
    vol = (g.rA[i, j]*g.deepFac2C[k]                                # :77-80
           * g.rhoFacC[k]*g.drF[k]*g.hFacC[i, j, k]
           * g.maskInC[i, j])
    tileVol = tile_sums(vol)
    tmpVal = 0.25*(                                                 # :93-102
        s.uVel[i, j, k]*s.uVel[i, j, k]
        * g.dyG[i, j]*g.dxC[i, j]*g.hFacW[i, j, k]
        + s.uVel[i+1, j, k]*s.uVel[i+1, j, k]
        * g.dyG[i+1, j]*g.dxC[i+1, j]*g.hFacW[i+1, j, k]
        + s.vVel[i, j, k]*s.vVel[i, j, k]
        * g.dxG[i, j]*g.dyC[i, j]*g.hFacS[i, j, k]
        + s.vVel[i, j+1, k]*s.vVel[i, j+1, k]
        * g.dxG[i, j+1]*g.dyC[i, j+1]*g.hFacS[i, j+1, k]
    )*g.maskInC[i, j]
    tileVlAv = tile_sums(tmpVal*g.deepFac2C[k]*g.rhoFacC[k]*g.drF[k])     # :103-104
    tmpVal = tmpVal*g.recip_hFacC[i, j, k]*g.recip_rA[i, j]                # :105
    theMax = max_chain(theMax, tmpVal, p="b")                             # :122
    nonzero = tmpVal != 0.0                                                # :123-126
    tileMean = tile_sums(jnp.where(nonzero, tmpVal, jnp.zeros((), tmpVal.dtype)))
    numPnts = numPnts + np.float64(np.count_nonzero(np.asarray(nonzero)))  # +1. per point: exact count

    j2, i2 = loop_j(1, sNy), loop_i(1, sNx)                                # :132-149
    tmpPE = 0.5*g.Bo_surf[i2, j2]*s.etaN[i2, j2]*s.etaN[i2, j2]          # :134-135 (0.5 _d 0)
    tmpPE = tmpPE + s.phi0surf[i2, j2]*s.etaN[i2, j2]                     # :138-139
    tilePEav = tile_sums(tmpPE*g.rA[i2, j2]*g.deepFac2F[1]                # :140-142
                         * g.maskInC[i2, j2])

    numPnts = global_sum_rl_scalar(numPnts, ex)                            # :153
    theMax = global_max_rl(theMax, ex)                                     # :154
    theMean = global_sum_tile_rl(tileMean, ex)                             # :155
    theVol = global_sum_tile_rl(tileVol, ex)                               # :156
    theVolMean = global_sum_tile_rl(tileVlAv, ex)                          # :157
    potEnMean = global_sum_tile_rl(tilePEav, ex)                           # :158
    if numPnts != 0.0:                                                     # :159
        theMean = theMean/numPnts
    if theVol != 0.0:                                                      # :160-163
        theVolMean = theVolMean/theVol
        potEnMean = potEnMean/theVol

    if mon.mon_output_AM:                                                  # :166-322 (lane B)
        _angular_momentum(myIter, cfg=cfg, grid=g, state=s, mon=mon, ex=ex, params=params)

    mon_set_pref("pe_b", mon=mon)                                          # :325-327
    mon_out_rl(mon_string_none, potEnMean, mon_foot_mean, cfg=cfg, mon=mon)

    mon_set_pref("ke", mon=mon)                                            # :330-336
    mon_out_rl(mon_string_none, theMax, mon_foot_max, cfg=cfg, mon=mon)
    mon_out_rl(mon_string_none, theVolMean, mon_foot_mean, cfg=cfg, mon=mon)
    mon_out_rl(mon_string_none, theVol, mon_foot_vol, cfg=cfg, mon=mon)


def _angular_momentum(myIter, *, cfg, grid, state, mon, ex, params):
    """mon_ke.F:166-322 (mon_output_AM; lane B, Task 25): the total angular momentum per unit area, printed as am_eta_mean,
    am_uZo_mean, am_tot_mean. Per tile (non-AB3 build): abFac1 = -(0.5 _d 0 + abEps) for myIter >= 1, else 0. _d 0
    (:170-178); tmpFld(i,j) = SUM_k R_drK*(abFac1*guNm1*deltaTMom + uVel)*hFacW (k order, from 0. _d 0, :180-202), then
    *u2zonDir*cosLat*rAw*maskInW with cosLat = COS(deg2rad*(yG(i,j)+yG(i,j+1))*halfRL) (:204-212); the same for vVel
    with hFacS, yG(i+1,j), v2zonDir, rAs, maskInS (:219-251); tileAMu chains the u points then the v points in j, i
    order (:213-218, :252-257). The mass term (:258-301): etaHnm1 (exactConserv) or etaN, *omega*rSphere*rSphere*(four
    corner cos^2 of yG)*0.25 _d 0, *maskInC*deepFac2F(ks)*rA*deepFac2F(ks)*rhoFacF(ks) with ks = kSurfC. Then
    GLOBAL_SUM_TILE_RL, *rUnit2mass, /globalArea (GRID.h: INI_GLOBAL_DOMAIN) and the three MON_OUT_RL (:305-320).
    COS is jnp.cos (glibc's bit for bit: mitjax/ops/libm.MEASURED). Every expression left to right as written.
    `params`: abEps, deltaTMom, rSphere, omega, rUnit2mass, freeSurfFac, globalArea (floats), exactConserv."""
    if params is None:
        raise ValueError("MON_KE: the angular momentum needs `params` (abEps, deltaTMom, rSphere, omega, ...)")
    from mitjax.model.grid import deg2rad
    from mitjax.ops.libm import M1_LIBM
    cos = M1_LIBM.cos
    sNx, sNy, Nr, OLx, OLy = cfg.sNx, cfg.sNy, cfg.Nr, cfg.OLx, cfg.OLy
    g, s, p = grid, state, params
    halfRL = 0.5                                                            # EEPARAMS.h:73
    abFac1 = 0.0                                                            # :170  0. _d 0
    if myIter >= 1:                                                         # :178
        abFac1 = -(0.5 + p.abEps)                                           # 0.5 _d 0
    j, i = loop_j(1, sNy), loop_i(1, sNx)
    parts = []
    for vel, gNm1, hFac, zon, rAx, mskIn, dj, di in (
            (s.uVel, s.guNm1, g.hFacW, g.u2zonDir, g.rAw, g.maskInW, 1, 0),        # :180-218
            (s.vVel, s.gvNm1, g.hFacS, g.v2zonDir, g.rAs, g.maskInS, 0, 1)):       # :219-257
        tmpFld = jnp.zeros_like(vel[i, j, 1])                               # :181-185  0. _d 0
        for k in range(1, Nr + 1):                                          # :186
            R_drK = p.rSphere*g.deepFacC[k]*g.deepFac2C[k]*g.rhoFacC[k]*g.drF[k]   # :187-188
            tmpVal = abFac1*gNm1[i, j, k]                                   # :195 / :234
            tmpVal = tmpVal*p.deltaTMom + vel[i, j, k]                      # :197 / :236
            tmpFld = tmpFld + R_drK*tmpVal*hFac[i, j, k]                    # :198-199 / :237-238
        cosLat = cos(deg2rad*(g.yG[i, j] + g.yG[i+di, j+dj])*halfRL)       # :206-207 / :245-246
        tmpFld = tmpFld*zon[i, j]*cosLat*rAx[i, j]*mskIn[i, j]             # :208-210 / :247-249
        parts.append(tmpFld)
    tileAMu = tile_sums(jnp.stack(parts, axis=1))                           # :213-218, :252-257: u then v
    tmpFld = s.etaHnm1[i, j] if p.exactConserv else s.etaN[i, j]           # :258-270
    jA, iA = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)
    cosLat = cos(deg2rad*g.yG[iA, jA])                                      # :275
    c2 = g.yG.local("cos2LatG").at[iA, jA].set(cosLat*cosLat)              # :276
    tmpFld = (tmpFld*p.omega*p.rSphere*p.rSphere                            # :281-285
              * ((c2[i, j] + c2[i+1, j+1]) + (c2[i+1, j] + c2[i, j+1]))
              * 0.25)                                                       # 0.25 _d 0
    ks = jnp.asarray(g.kSurfC[i, j]).astype(jnp.int32)                     # :290
    d2F = jnp.asarray(g.deepFac2F.data)[ks - 1]                             # deepFac2F(ks): gather (1..Nr+1)
    rFF = jnp.asarray(g.rhoFacF.data)[ks - 1]                               # rhoFacF(ks)
    tmpFld = tmpFld*g.maskInC[i, j]*d2F*g.rA[i, j]*d2F*rFF                 # :291-293
    tileAMs = tile_sums(tmpFld)                                             # :296-301
    totAMu = global_sum_tile_rl(tileAMu, ex)                                # :305
    totAMs = global_sum_tile_rl(tileAMs, ex)                                # :306
    mon_set_pref("am", mon=mon)                                             # :309
    totAMu = totAMu*p.rUnit2mass                                            # :310
    totAMs = totAMs*p.rUnit2mass                                            # :311
    if p.globalArea > 0.0:                                                  # :312
        totAMu = totAMu/p.globalArea
    if p.globalArea > 0.0:                                                  # :313
        totAMs = totAMs/p.globalArea
    mon_out_rl(mon_string_none, totAMs, "_eta_mean", cfg=cfg, mon=mon)     # :314-315
    mon_out_rl(mon_string_none, totAMu, "_uZo_mean", cfg=cfg, mon=mon)     # :316-317
    totAMu = totAMu + p.freeSurfFac*totAMs                                  # :318
    mon_out_rl(mon_string_none, totAMu, "_tot_mean", cfg=cfg, mon=mon)     # :319-320
