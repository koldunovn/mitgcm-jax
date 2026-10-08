"""DYNAMICS: model/src/dynamics.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.src.calc_grad_phi_surf import calc_grad_phi_surf
from mitjax.model.src.calc_phi_hyd import calc_phi_hyd
from mitjax.model.src.calc_viscosity import calc_viscosity
from mitjax.model.src.timestep import timestep
from mitjax.pkg.mom_fluxform.mom_fluxform import mom_fluxform


def dynamics(myTime, myIter, *, cfg, grid, params, state, ff, fp, phi0surf, probe=None, visc=None, ctrlf=None,
             mix=None):
    """DYNAMICS( myTime, myIter, myThid )   @63cdc0b model/src/dynamics.F:21-739

    C     | SUBROUTINE DYNAMICS
    C     | o Controlling routine for the explicit part of the model dynamics.
    C     | This routine evaluates the "dynamics" terms for each vertical layer in turn. Vertical layers are
    C     | traversed from the bottom to the top (sic: k = 1..Nr, top to bottom at 63cdc0b) ...

    Returns the State (gU, gV, guNm1, gvNm1, totPhiHyd, phiHydLow written). The level loop DO k=1,Nr (:422-567) is an
    unrolled Python loop in Fortran order (KERNEL_GUIDE §4, decided for M1): the level fields phiHydF (carried from
    k to k+1) and fVerU/fVerV(.,.,kUp/kDown) (the two-slot rotation of :428-431) are carried exactly as the Fortran
    carries them. Locals fVerU, fVerV (..,2), phiHydF, phiHydC, phiSurfX/Y, guDissip, gvDissip zeroed as :308-335;
    dPhiHydX/Y are not initialised by DYNAMICS (written by CALC_PHI_HYD -> CALC_GRAD_PHI_HYD).
    GOADK lane: vectorInvariantMomentum calls MOM_VECINV (:525-533) with `visc` (MOM_VISC.h of MOM_INIT_FIXED) and
    `ctrlf` (CTRL_FIELDS.h, for the bottom-drag control); its per-level probe is D00c_mom_vecinv.
    M3 lane MLAdjust: MOM_U/V_IMPLICIT_R after the level loop (:569-578, INCLUDE_IMPLVERTADV_CODE with
    ALLOW_MOM_COMMON and without ALLOW_AUTODIFF); lane M4ADCOL: else IMPLDIFF of gU, gV (:580-600, implicitViscosity). Lane M4LAB: the CD scheme's IMPLDIFF
    of vVelD/uVelD (:614-632, lab_sea); useDiagnostics is output only (the
    convention of brainstorm §3: DIAGNOSTICS_FILL and the botDragU/V accumulators, read by nothing else),
    ALLOW_NONHYDROSTATIC / ALLOW_SMAG_3D / ALLOW_LEITH_QG blocks (not compiled in M1), the debug block (:707).
    `probe((stage, k), values)`: optional, called with the per-level values of the dump stages D00a_phi_hyd and
    D00b_mom_fluxform (gates only). R5 arm (plan Task 16, ALLOW_AUTODIFF builds): gU = gV = 0 on every point
    (:297-307), phiHydLow = 0 (:325), kappaRU/kappaRV = 0 whatever momViscosity (:370-381), guDissip = gvDissip = 0
    before each level's MOM_FLUXFORM (:491-498)."""
    if params.vectorInvariantMomentum and visc is None:              # GOADK lane: MOM_VECINV needs MOM_VISC.h
        raise ValueError("DYNAMICS: vectorInvariantMomentum needs `visc` (MOM_VISC.h of MOM_INIT_FIXED)")
    ggl = (mix or {}).get("ggl")
    if ggl is not None and cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", "GGL90_OPTIONS.h") and ggl.useLANGMUIR \
            and not params.vectorInvariantMomentum:                           # mom_fluxform.F:1083-1088
        raise NotImplementedError("MOM_FLUXFORM: useLANGMUIR (GGL90_ADD_STOKESDRIFT) is not wired")
    implvert = (cfg.cpp.flag("INCLUDE_IMPLVERTADV_CODE") and cfg.cpp.flag("ALLOW_MOM_COMMON")
                and not cfg.cpp.flag("ALLOW_AUTODIFF"))                         # :569-570
    # lane M4ADCOL: without the MOM_U/V_IMPLICIT_R arm (:569-570; e.g. ALLOW_AUTODIFF, 1D_ocean_ice_column/code_ad)
    # the #else IF ( implicitViscosity ) arm calls IMPLDIFF for gU and gV (:580-600); momImplVertAdv or
    # selectImplicitDrag >= 1 without implicitViscosity has no code there (CONFIG_CHECK's business): raise
    impl_diff = (not implvert) and params.implicitViscosity
    if (params.momImplVertAdv or params.selectImplicitDrag >= 1) and not implvert:
        raise NotImplementedError("DYNAMICS: momImplVertAdv / selectImplicitDrag without MOM_U/V_IMPLICIT_R")
    if params.implicitViscosity and params.useCDscheme and not cfg.cpp.ALLOW_CD_CODE:
        raise NotImplementedError("DYNAMICS: useCDscheme without ALLOW_CD_CODE")
    if params.useDiagnostics:
        # vermix lane (M3 Task 30): with useDiagnostics, DYNAMICS and its callees only add DIAGNOSTICS_FILL calls and
        # the bottom-stress accumulators botDragU/V (FFIELDS.h:269-274, "for diagnostics"; reset :336-343, added to
        # in MOM_U/V_BOTTOMDRAG, MOM_U/V_IMPLICIT_R), which no model routine reads: output only (pkg/diagnostics is
        # not ported), so the kernels see useDiagnostics = .FALSE. (as THERMODYNAMICS' DO_STATEVARS_DIAGS)
        params = params.replace(static={"useDiagnostics": False})
    if params.debugLevel >= 4:                                                  # :707 debLevD
        raise NotImplementedError("DYNAMICS: the debugLevel >= debLevD statistics block is not ported")
    sz = cfg.size
    OLx, OLy, sNx, sNy, Nr = sz.OLx, sz.OLy, sz.sNx, sz.sNy, sz.Nr
    iMin, iMax = 0, sNx+1                                                       # :191
    jMin, jMax = 0, sNy+1                                                       # :192
    b2 = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    z2 = state.etaN.local("z2")
    if cfg.cpp.ALLOW_AUTODIFF:                                                  # :297-307 (R5 arm)
        from mitjax.farray import loops_kji
        k3, j3, i3 = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
        state = state.replace(gU=state.gU.at[i3, j3, k3].set(0.),              # :302  0. _d 0
                              gV=state.gV.at[i3, j3, k3].set(0.))              # :303
    jA = loop_j(1-OLy, sNy+OLy)                                                 # :308
    iA = loop_i(1-OLx, sNx+OLx)                                                 # :309
    fVerU = [z2.at[iA, jA].set(0.), z2.at[iA, jA].set(0.)]                     # :310-311  fVerU(i,j,1:2)
    fVerV = [z2.at[iA, jA].set(0.), z2.at[iA, jA].set(0.)]                     # :312-313
    phiHydF = z2.at[iA, jA].set(0.)                                             # :314
    phiHydC = z2.at[iA, jA].set(0.)                                             # :315
    phiSurfX = z2.at[iA, jA].set(0.)                                            # :320
    phiSurfY = z2.at[iA, jA].set(0.)                                            # :321
    guDissip = z2.at[iA, jA].set(0.)                                            # :322
    gvDissip = z2.at[iA, jA].set(0.)                                            # :323
    if cfg.cpp.ALLOW_AUTODIFF:                                                  # :324-333 (R5 arm)
        state = state.replace(phiHydLow=state.phiHydLow.at[iA, jA].set(0.))    # :325  0. _d 0
        if cfg.cpp.NONLIN_FRSURF and cfg.cpp.ALLOW_MOM_FLUXFORM and not cfg.cpp.DISABLE_RSTAR_CODE:
            # :326-331 (PTRACERS lane, tutorial_tracer_adjsens/code_ad): MOM_FLUXFORM.h dWtransC/U/V = 0. _d 0
            state = state.replace(dWtransC=state.dWtransC.at[iA, jA].set(0.),
                                  dWtransU=state.dWtransU.at[iA, jA].set(0.),
                                  dWtransV=state.dWtransV.at[iA, jA].set(0.))
    dPhiHydX = z2.local("dPhiHydX")
    dPhiHydY = z2.local("dPhiHydY")

    # implicSurfPress is a traced REAL: `IF (implicSurfPress.NE.1.)` (:353) cannot select code. CALC_GRAD_PHI_SURF is
    # called when the static copy says so; implicSurfPress = 1 is checked on the host (Cg2dParams) for every M1 run.
    if params.implicSurfPress_ne_1:                                             # :353-359
        phiSurfX, phiSurfY = calc_grad_phi_surf(iMin, iMax, jMin, jMax, state.etaN, phiSurfX, phiSurfY,
                                                cfg=cfg, grid=grid)

    # P=4 fix (tracer lane): the locals take the tile count of the arrays this program holds (the device's block
    # under shard_map), not SIZE.h's nSx*nSy (grid.local), so the same code runs at P=1 and P=N
    nT = state.uVel.data.shape[0]
    # full_like of the tiled uVel: tile-varying under shard_map (CALC_VISCOSITY's level scan writes them per tile)
    kappaRU = FArray(jnp.full_like(state.uVel.data, jnp.nan, shape=(nT, Nr+1) + state.uVel.data.shape[2:]),
                     "kappaRU", k=(1, Nr+1), **b2)                              # :154  (..,Nr+1)
    kappaRV = FArray(jnp.full_like(state.uVel.data, jnp.nan, shape=(nT, Nr+1) + state.uVel.data.shape[2:]),
                     "kappaRV", k=(1, Nr+1), **b2)                              # :155
    if cfg.cpp.ALLOW_AUTODIFF or not params.momViscosity:                       # :370-381 (#ifndef ALLOW_AUTODIFF IF)
        kA = (1, Nr+1)
        for k in range(kA[0], kA[1]+1):
            kappaRU = kappaRU.at[iA, jA, k].set(0.)
            kappaRV = kappaRV.at[iA, jA, k].set(0.)
    if params.momViscosity:                                                     # :385-390
        kappaRU, kappaRV = calc_viscosity(iMin, iMax, jMin, jMax, kappaRU, kappaRV, cfg=cfg, params=params,
                                          grid=grid, mix=mix)            # vermix lane: KPP_CALC_VISC

    # DO k=1,Nr (:422): the level body below, k = 1 and k = Nr as static calls (CALC_PHI_HYD and MOM_FLUXFORM branch
    # on them) and k = 2 .. Nr-1 in a level scan (KERNEL_GUIDE §4: mitjax/ops/scan_k.scan_levels; k is then a traced
    # level index). `c` carries what the Fortran carries from level to level; `y` the per-level gate probes.
    w2 = _w2_tile_view(grid, params)                                            # w2: lane B (cube)

    def level_k(k, c):
        state, phiHydF, phiHydC, dPhiHydX, dPhiHydY, fVerU, fVerV, guDissip, gvDissip = c
        fVerU, fVerV = list(fVerU), list(fVerV)
        y = {}
        kUp = 1+(k+1) % 2                                                       # :430  1+MOD(k+1,2)
        kDown = 1+k % 2                                                         # :431  1+MOD(k,2)
        state, phiHydF, phiHydC, dPhiHydX, dPhiHydY = calc_phi_hyd(            # :482-486
            iMin, iMax, jMin, jMax, k, phiHydF, phiHydC, dPhiHydX, dPhiHydY, myTime, myIter,
            cfg=cfg, grid=grid, params=params, state=state, phi0surf=phi0surf)
        if probe is not None:                                                   # gates: D00a (per level k)
            y["D00a_phi_hyd"] = dict(dPhiHydX=dPhiHydX, dPhiHydY=dPhiHydY, phiHydC=phiHydC, phiHydF=phiHydF)
        if params.momStepping:                                                  # :490
            if cfg.cpp.ALLOW_AUTODIFF:                                          # :491-498 (R5 arm)
                guDissip = guDissip.at[iA, jA].set(0.)                          # :494  0. _d 0
                gvDissip = gvDissip.at[iA, jA].set(0.)                          # :495
            if not params.vectorInvariantMomentum:                              # :516
                state, fVerU[kUp-1], fVerV[kUp-1], fVerU[kDown-1], fVerV[kDown-1], guDissip, gvDissip = \
                    mom_fluxform(k, iMin, iMax, jMin, jMax, kappaRU, kappaRV,   # :517-523
                                 fVerU[kUp-1], fVerV[kUp-1], fVerU[kDown-1], fVerV[kDown-1], guDissip, gvDissip,
                                 myTime, myIter, cfg=cfg, grid=grid, params=params, state=state,
                                 visc=visc)                                     # visc: M3 lane MLAdjust
                if probe is not None:                                           # gates: D00b (per level k)
                    y["D00b_mom_fluxform"] = dict(gU=state.gU, gV=state.gV, guDissip=guDissip, gvDissip=gvDissip)
            else:                                                               # :525-533 (GOADK lane)
                from mitjax.pkg.mom_vecinv.mom_vecinv import mom_vecinv
                fVerU[kDown-1], fVerV[kDown-1], guDissip, gvDissip, state = mom_vecinv(
                    k, iMin, iMax, jMin, jMax, kappaRU, kappaRV,
                    fVerU[kUp-1], fVerV[kUp-1], fVerU[kDown-1], fVerV[kDown-1], guDissip, gvDissip,
                    myTime, myIter, cfg=cfg, grid=grid, params=params, state=state, visc=visc, ctrlf=ctrlf,
                    w2=w2)
                if probe is not None:                                           # gates: D00c (per level k)
                    y["D00c_mom_vecinv"] = dict(gU=state.gU, gV=state.gV, guDissip=guDissip, gvDissip=gvDissip)
            state = timestep(iMin, iMax, jMin, jMax, k, dPhiHydX, dPhiHydY, phiSurfX, phiSurfY,   # :558-562
                             guDissip, gvDissip, myTime, myIter, cfg=cfg, grid=grid, params=params, state=state,
                             ff=ff, fp=fp)
        return (state, phiHydF, phiHydC, dPhiHydX, dPhiHydY, tuple(fVerU), tuple(fVerV), guDissip, gvDissip), y

    from mitjax.ops.scan_k import scan_levels
    c = (state, phiHydF, phiHydC, dPhiHydX, dPhiHydY, tuple(fVerU), tuple(fVerV), guDissip, gvDissip)
    c, y1 = level_k(1, c)
    outs = [(1, y1)]
    if Nr > 1:
        c, mid = scan_levels(level_k, c, 2, Nr-1, with_out=True)
        c, yN = level_k(Nr, c)
        outs += mid + [(Nr, yN)]
    state = c[0]
    if probe is not None:                                                       # the gates' per-level stages
        for k, y in outs:
            for stage, vals in y.items():
                probe((stage, k), vals)
    # vermix lane (M3 Task 30): the dump stages around the implicit vertical viscosity (:569-600)
    if probe is not None:                                                       # gates: D01 (before :569)
        probe("D01_before_impl_visc", dict(gU=state.gU, gV=state.gV, kappaRU=kappaRU, kappaRV=kappaRV))
    if implvert and (params.momImplVertAdv or params.implicitViscosity
                     or params.selectImplicitDrag >= 1):                        # :572-578 (M3 lane MLAdjust)
        from mitjax.pkg.mom_common.mom_implicit_r import mom_u_implicit_r, mom_v_implicit_r
        state = mom_u_implicit_r(kappaRU, myTime, myIter, cfg=cfg, grid=grid, params=params, state=state)
        state = mom_v_implicit_r(kappaRV, myTime, myIter, cfg=cfg, grid=grid, params=params, state=state)
    if impl_diff:                                                               # :580-600 (lane M4ADCOL)
        from mitjax.model.src.impldiff import impldiff
        gU = impldiff(iMin, iMax, jMin, jMax, -1, kappaRU, grid.recip_hFacW, state.gU,
                      cfg=cfg, grid=grid, params=params)                        # :587-591
        gV = impldiff(iMin, iMax, jMin, jMax, -2, kappaRV, grid.recip_hFacS, state.gV,
                      cfg=cfg, grid=grid, params=params)                        # :595-599
        state = state.replace(gU=gU, gV=gV)
    if probe is not None:                                                       # gates: D02 (after :600)
        probe("D02_after_impl_visc", dict(gU=state.gU, gV=state.gV))
    if cfg.cpp.ALLOW_CD_CODE and params.implicitViscosity and params.useCDscheme:   # :614-615 (lane M4LAB)
        # :619-623 IMPLDIFF( bi,bj,iMin,iMax,jMin,jMax, 0, kappaRU, recip_hFacW, vVelD ) and :627-631 the same with
        # kappaRV, recip_hFacS, uVelD: the D-grid velocities (vVelD lives at u points, uVelD at v points); tracerId
        # 0 -> deltaTMom (impldiff.F:100-103). No ALLOW_OBCS here (:603-612 compiled only with it).
        from mitjax.model.src.impldiff import impldiff
        vVelD = impldiff(iMin, iMax, jMin, jMax, 0, kappaRU, grid.recip_hFacW, state.vVelD,
                         cfg=cfg, grid=grid, params=params)                      # :619-623
        uVelD = impldiff(iMin, iMax, jMin, jMax, 0, kappaRV, grid.recip_hFacS, state.uVelD,
                         cfg=cfg, grid=grid, params=params)                      # :627-631
        state = state.replace(uVelD=uVelD, vVelD=vVelD)
    return state


def _w2_tile_view(grid, params):
    """Lane B (Task 25, solid-body.cs-32x32x1): the per-tile W2_EXCH2_TOPOLOGY.h view MOM_VECINV's cube passes read
    (mom_calc_relvort3, mom_calc_visc, mom_vi_del2uv: exch2_myFace, exch2_is[WSEN]edge of myTile = W2_myTileList(bi,bj),
    as [tile] arrays) from the grid's W2 tile view (mitjax/model/grid.w2_tile_view, attached by drivers.Model for a
    cubed-sphere run). None without useCubedSphereExchange (the cube passes are not run)."""
    if not params.useCubedSphereExchange:
        return None
    from types import SimpleNamespace
    from mitjax.model.grid import W2_TILE_VIEW
    return SimpleNamespace(**{n: getattr(grid, n).data[:, 0, 0] for n in W2_TILE_VIEW})
