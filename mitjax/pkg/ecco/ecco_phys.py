"""ECCO_PHYS   @63cdc0b pkg/ecco/ecco_phys.F:17-636 (lane M4ADCOL session 3)

Called from ECCO_INIT_VARIA with (startTime, -1) (ecco_init_varia.F:50) and at the end of every FORWARD_STEP under
useECCO (forward_step.F:1169, after COST_TILE). It writes ECCO.h fields only: m_eta, m_bp (:205-255), m_UE, m_VN
(:311-325), trVol / trHeat / trSalt (:334-336, ECCO_ZERO) and gencost_storefld (:349-631). In
1D_ocean_ice_column/input_ad no executed line reads any of them (COST_GENCOST_CUSTOMIZE's m_eta / m_bp / m_UE /
m_VN / storefld arms, cost_gencost_customize.F:122-147, are unexecuted on the coverage page of job27856010), so no
oracle output can show them: the port is literal and computed, but blind (stated in the lane handoff).

This build: ALLOW_PSBAR_STERIC, ATMOSPHERIC_LOADING / ALLOW_IB_CORR, ECCO_VARIABLE_AREAVOLGLOB undefined
(ECCO_OPTIONS.h, CPP_OPTIONS.h of code_ad); ALLOW_DIAGNOSTICS compiled with useDiagnostics off (:257-273 skipped).
The gencost_storefld loop (:349-631) for this build's gencost: no m_freeboard / m_boxmean / m_horflux / m_tr* barfile
and no density mask (those arms raise), and gencost_mskCsurf / mskWsurf / mskSsurf hold ECCO_READPARMS' zeros (no
mask file is loaded without flags -3..-5, which raise there): every accumulation adds tmpmsk*tmpfld*... = 0*0*(...)
= +0 to the ECCO_ZERO'd field, so gencost_storefld is the zero field (areavolTile is read only by the m_boxmean
ECCO_DIV, :617-629). The carried part: {"m_eta", "m_bp", "m_UE", "m_VN"}.
"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.pkg.ecco.ecco_readparms import _eq, _sub


def check_ecco_phys(ep):
    """Set-up check of the gencost arms ECCO_PHYS' storefld loop would take (module docstring)."""
    for g in ep.gencost:
        bf = g.gencost_barfile
        if (g.gencost_useDensityMask or g.gencost_msk_is3d
                or any(_eq(_sub(bf, n), p) for n, p in ((11, "m_freeboard"), (9, "m_boxmean"), (9, "m_horflux"),
                                                         (7, "m_trVol"), (8, "m_trHeat"), (8, "m_trSalt")))):
            raise NotImplementedError(f"ECCO_PHYS: gencost {g.k} ({bf.rstrip()}) storefld arm not ported")


def ecco_phys(myIter, *, cfg, grid, params, eos, state, ff, fp):
    """ECCO_PHYS( myTime, myIter, myThid ) -> {"m_eta", "m_bp" (FArrays xy), "m_UE", "m_VN" (xyz)}.
    `myIter` a Python int only for the initial call (-1); a traced counter otherwise takes the rhoInSitu arm
    (:115-133: myIter .EQ. -1 never holds inside the time loop, myIter >= nIter0 >= 0)."""
    from mitjax.model.src.find_rho import find_rho_2d
    from mitjax.model.src.rotate_uv2en import rotate_uv2en_rl
    sz = cfg.size
    recip_rhoConst = params.recip_rhoConst
    # :108 area_reg_sq (read only by the m_freeboard arm, which raises in check_ecco_phys); :110 tmpfac (read only
    # under ALLOW_IB_CORR)
    sIceLoadFacLoc = 0.0                                                     # :111 zeroRL
    if fp.useRealFreshWaterFlux:                                             # :112 (PARAMS.h; the port's fp)
        sIceLoadFacLoc = recip_rhoConst
    rhoLoc = state.rhoInSitu.local("rhoLoc")
    if isinstance(myIter, int) and myIter == -1:                             # :116-124
        from mitjax.model.src.do_oceanic_phys import _level, _set_level
        from mitjax.ops.scan_k import scan_levels

        def rho_k(k, rl):                                                    # DO k = 1,Nr: CALL FIND_RHO_2D(..k..)
            lev = find_rho_2d(1 - sz.OLx, sz.sNx + sz.OLx, 1 - sz.OLy, sz.sNy + sz.OLy, k, _level(state.theta, k),
                              _level(state.salt, k), _level(rl, k), k, cfg=cfg, grid=grid, params=params, eos=eos,
                              state=state)
            return _set_level(rl, k, lev)
        rhoLoc = scan_levels(rho_k, rhoLoc, 1, sz.Nr)                        # KERNEL_GUIDE §4: a per-level caller
    else:                                                                    # :125-132
        k, j, i = loops_kji((1, sz.Nr), (1 - sz.OLy, sz.sNy + sz.OLy), (1 - sz.OLx, sz.sNx + sz.OLx))
        rhoLoc = rhoLoc.at[i, j, k].set(state.rhoInSitu[i, j, k])
    j, i = loop_j(1 - sz.OLy, sz.sNy + sz.OLy), loop_i(1 - sz.OLx, sz.sNx + sz.OLx)
    m_eta = state.etaN.local("m_eta")
    m_eta = m_eta.at[i, j].set((state.etaN[i, j] + ff.sIceLoad[i, j]*sIceLoadFacLoc) * grid.maskC[i, j, 1])  # :211-219
    m_bp = state.etaN.local("m_bp")
    m_bp = m_bp.at[i, j].set((state.etaN[i, j] - grid.R_low[i, j]) * params.gravity
                             + ff.sIceLoad[i, j] * params.gravity * sIceLoadFacLoc
                             + ff.pLoad[i, j] * recip_rhoConst)                                            # :225-241
    from mitjax.ops.scan_k import level, scan_k

    def iteration(mbp, x):                                                   # :243-250 one level k (a recursion)
        mbp = mbp.at[i, j].set(mbp[i, j] + x["rhoLoc"][i, j]*x["drF"]*x["hFacC"][i, j]
                               * params.gravity * recip_rhoConst)
        return mbp, None
    m_bp, _ = scan_k(iteration, m_bp, range(1, sz.Nr + 1),
                     lambda kk: {"rhoLoc": level(rhoLoc, kk), "drF": grid.drF[kk], "hFacC": level(grid.hFacC, kk)})
    m_bp = m_bp.at[i, j].set(m_bp[i, j] * grid.maskC[i, j, 1])                                             # :251-255
    k, j3, i3 = loops_kji((1, sz.Nr), (1 - sz.OLy, sz.sNy + sz.OLy), (1 - sz.OLx, sz.sNx + sz.OLx))
    m_UE = state.uVel.local("m_UE").at[i3, j3, k].set(0.0)                                                 # :311-321
    m_VN = state.vVel.local("m_VN").at[i3, j3, k].set(0.0)
    _, _, m_UE, m_VN = rotate_uv2en_rl(state.uVel, state.vVel, m_UE, m_VN, True, True, False, sz.Nr, cfg=cfg,
                                       grid=grid, fp=fp)                                                   # :323-325
    return {"m_eta": m_eta, "m_bp": m_bp, "m_UE": m_UE, "m_VN": m_VN}
