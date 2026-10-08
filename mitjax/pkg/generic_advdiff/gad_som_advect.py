"""GAD_SOM_ADVECT: tendency of a tracer from the 2nd-Order Moment advection scheme (Prather, 1986), multi-dimensional.

GAD.h constants come from `gad_h.py` (each cited there). The tracer's moments `smTr` (the pickup state som_T / som_S of GAD_SOM_VARS.h) are a
tuple of nSOM FArrays: smTr[n-1] holds the Fortran smTr(:,:,:,bi,bj,n), n = 1..9 = x, y, z, xx, yy, zz, xy, xz, yz
(GAD_SOM_VARS.h:219-220).
"""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.scan_k import scan_levels
from mitjax.pkg.generic_advdiff.gad_h import ENUM_SOM_LIMITER, ENUM_SOM_PRATHER, nSOM
from mitjax.pkg.generic_advdiff.gad_som_adv_r import gad_som_adv_r
from mitjax.pkg.generic_advdiff.gad_som_adv_x import gad_som_adv_x
from mitjax.pkg.generic_advdiff.gad_som_adv_y import gad_som_adv_y
from mitjax.pkg.generic_advdiff.gad_som_lim_r import gad_som_lim_r


# the 24 flux work arrays of the vertical step, in GAD_SOM_ADV_R's argument order (gad_som_adv_r.F:18-21)
_FLUX_R = ("alp", "aln", "fp_v", "fn_v", "fp_o", "fn_o", "fp_x", "fn_x", "fp_y", "fn_y", "fp_z", "fn_z",
           "fp_xx", "fn_xx", "fp_yy", "fn_yy", "fp_zz", "fn_zz", "fp_xy", "fn_xy", "fp_xz", "fn_xz", "fp_yz", "fn_yz")


def _level(A, k, cfg):
    """A(1-OLx,1-OLy,k) as the Fortran passes it to a routine with a 2-D dummy argument: the level-k section, a 2-D
    FArray (1-OLx:sNx+OLx, 1-OLy:sNy+OLy) of the same tiles."""
    jA, iA = loop_j(1-cfg.OLy, cfg.sNy+cfg.OLy), loop_i(1-cfg.OLx, cfg.sNx+cfg.OLx)
    return FArray(A[iA, jA, k], A.name, i=(1-cfg.OLx, cfg.sNx+cfg.OLx), j=(1-cfg.OLy, cfg.sNy+cfg.OLy))


def _set_level(A, k, a2, cfg):
    """The level-k section of A after the routine updated its 2-D dummy argument `a2`."""
    jA, iA = loop_j(1-cfg.OLy, cfg.sNy+cfg.OLy), loop_i(1-cfg.OLx, cfg.sNx+cfg.OLx)
    return A.at[iA, jA, k].set(a2[iA, jA])


def _local_kupdw(like, name, cfg):
    """A local `_RL name(1-OLx:sNx+OLx,1-OLy:sNy+OLy,2)` (the kUp/kDown slots), every point NaN, tiles as `like`."""
    shape = (like.ntiles, 2, cfg.sNy+2*cfg.OLy, cfg.sNx+2*cfg.OLx)
    return FArray(jnp.full(shape, jnp.nan, like.dtype), name, i=(1-cfg.OLx, cfg.sNx+cfg.OLx),
                  j=(1-cfg.OLy, cfg.sNy+cfg.OLy), k=(1, 2))


def _som_cs_masks(ovl, intr, N_edge, S_edge, E_edge, W_edge, dirX, cfg):
    """[T,j,i] write masks of one cube sweep of GAD_SOM_ADV_X (dirX) or _Y: (moments, flux) from the strip ranges of
    gad_som_adv_x.F:137-159 / gad_som_adv_y.F:137-159 (part 2-3 write the moments on [iMin..iMax] x strip rows, part 1
    the flux on [iMin..iMax+1]; Y transposed)."""
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    jj = jnp.arange(1-OLy, sNy+OLy+1)[None, :, None]
    ii = jnp.arange(1-OLx, sNx+OLx+1)[None, None, :]
    if dirX:
        rows_ovl = (S_edge & (jj <= 0)) | (N_edge & (jj >= sNy+1))             # :146-154 (strips 1, 2)
        jMin = jnp.where(intr & S_edge, 1, 1-OLy)                              # :140, :157
        jMax = jnp.where(intr & N_edge, sNy, sNy+OLy)                          # :141, :158
        rows = jnp.where(ovl, rows_ovl, (jj >= jMin) & (jj <= jMax))
        iMin = jnp.where(ovl & W_edge, 1, 1-OLx+1)                             # :138, :144
        iMax = jnp.where(ovl & E_edge, sNx, sNx+OLx-1)                         # :139, :145
        return rows & (ii >= iMin) & (ii <= iMax), rows & (ii >= iMin) & (ii <= iMax+1)
    cols_ovl = (W_edge & (ii <= 0)) | (E_edge & (ii >= sNx+1))                 # :146-154 (strips 1, 2)
    iMin = jnp.where(intr & W_edge, 1, 1-OLx)                                  # :138, :157
    iMax = jnp.where(intr & E_edge, sNx, sNx+OLx)                              # :139, :158
    cols = jnp.where(ovl, cols_ovl, (ii >= iMin) & (ii <= iMax))
    jMin = jnp.where(ovl & S_edge, 1, 1-OLy+1)                                 # :140, :144
    jMax = jnp.where(ovl & N_edge, sNy, sNy+OLy-1)                             # :141, :145
    return cols & (jj >= jMin) & (jj <= jMax), cols & (jj >= jMin) & (jj <= jMax+1)


def _som_cs_passes(cs, k, limiter, advectionScheme, deltaT, uTrans, vTrans, maskInC, smVol, smTr0, smTr, afx, afy,
                   smCorners, cfg):
    """The three cube passes of one level (gad_som_advect.F:309-439), every tile at once. Per tile the flags of
    gad_advection's cube passes (the same statements, :316-332); GAD_SOM_PREP_CS_CORNER where `calc_fluxes .AND.
    (.NOT.overlapOnly .OR. edges) .AND. .NOT.interiorOnly` (:351-362, :398-409); the sweep GAD_SOM_ADV_X / _Y is
    evaluated on its full-tile range (overlapOnly = interiorOnly = .FALSE., the ported branch) and written only where
    the Fortran's strips write (_som_cs_masks): its statements are pointwise per row (X) or column (Y), each strip row
    reads only its own row (part 1: the moments at i-1 and i; parts 2-3: the fluxes at i and i+1 and its own
    moments), so the full-range values at the written points equal the strip loop's. Limiter 0 only (the caller
    raises for 81)."""
    from mitjax.pkg.generic_advdiff.gad_advection import _cs_pass_flags
    from mitjax.pkg.generic_advdiff.gad_cs_passes import NPASS_CS
    from mitjax.pkg.generic_advdiff.gad_som_prep_cs_corner import gad_som_prep_cs_corner
    kw = dict(sNx=cfg.sNx, sNy=cfg.sNy, OLx=cfg.OLx, OLy=cfg.OLy)
    N_edge, S_edge, E_edge, W_edge = cs["N_edge"], cs["S_edge"], cs["E_edge"], cs["W_edge"]
    for ipass in range(1, NPASS_CS+1):                                          # :309
        overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y = _cs_pass_flags(cs["nCFace"], ipass)  # :316-332
        for dirX in (True, False):
            if dirX:                                                            # :351-390
                doIt = calc_fluxes_X & (~overlapOnly | N_edge | S_edge)
            else:                                                               # :398-437
                doIt = calc_fluxes_Y & (~overlapOnly | E_edge | W_edge)
            lv = [_level(a, k, cfg) for a in [smVol, smTr0] + list(smTr)]
            v, o, tr, smCorners = gad_som_prep_cs_corner(                       # :356-362 / :403-409
                lv[0].data, lv[1].data, [a.data for a in lv[2:]], smCorners, dirX, overlapOnly, interiorOnly,
                N_edge, S_edge, E_edge, W_edge, ipass, active=doIt & ~interiorOnly, **kw)
            lv = [FArray(d, a.name, tiled=a.tiled, _dims=a.dims) for d, a in zip([v, o] + tr, lv)]
            if advectionScheme not in (ENUM_SOM_PRATHER, ENUM_SOM_LIMITER):
                raise ValueError("GAD_SOM_ADVECT: adv. scheme incompatibale with SOM")   # :386 / :433  STOP
            sweep = gad_som_adv_x if dirX else gad_som_adv_y                    # :367-384 / :414-431
            out = sweep(k, limiter, False, False, False, False, False, False, deltaT, uTrans if dirX else vTrans,
                        maskInC, *lv, afx if dirX else afy, cfg=cfg)
            mMom, mFlux = _som_cs_masks(overlapOnly, interiorOnly, N_edge, S_edge, E_edge, W_edge, dirX, cfg)
            new = [FArray(jnp.where(doIt & mMom, o_.data, a.data), a.name, tiled=a.tiled, _dims=a.dims)
                   for o_, a in zip(out[:11], lv)]
            smVol = _set_level(smVol, k, new[0], cfg)
            smTr0 = _set_level(smTr0, k, new[1], cfg)
            smTr = [_set_level(smTr[n-1], k, new[1+n], cfg) for n in range(1, nSOM+1)]
            fl = afx if dirX else afy
            fl = FArray(jnp.where(doIt & mFlux, out[11].data, fl.data), fl.name, tiled=fl.tiled, _dims=fl.dims)
            if dirX:
                afx = fl
            else:
                afy = fl
    return smVol, smTr0, smTr, afx, afy, smCorners


def gad_som_advect(implicitAdvection, advectionScheme, vertAdvecScheme, tracerIdentity, deltaTLev,
                   uFld, vFld, wFld, tracer, smTr, gTracer, myTime, myIter, *, cfg, grid, params):
    """GAD_SOM_ADVECT(implicitAdvection, advectionScheme, vertAdvecScheme, tracerIdentity, deltaTLev,
                      uFld, vFld, wFld, tracer, smTr, gTracer, bi,bj, myTime,myIter,myThid)
    @63cdc0b pkg/generic_advdiff/gad_som_advect.F:14-679

    C !DESCRIPTION:
    C Calculates the tendency of a tracer due to advection.
    C It uses the 2nd-Order moment advection scheme with multi-dimensional method
    C  see Prather, 1986, JGR, v.91, D-6, pp.6671-6681.
    C
    C The tendency (output) is over-written by this routine.
    C !INPUT PARAMETERS:
    C  implicitAdvection :: implicit vertical advection (later on)
    C  advectionScheme   :: advection scheme to use (Horizontal plane)
    C  vertAdvecScheme   :: advection scheme to use (vertical direction)
    C  tracerIdentity    :: tracer identifier (required only for OBCS)
    C  uFld              :: Advection velocity field, zonal component
    C  vFld              :: Advection velocity field, meridional component
    C  wFld              :: Advection velocity field, vertical component
    C  tracer            :: tracer field
    C  myTime            :: current time
    C  myIter            :: iteration number
    C !OUTPUT PARAMETERS:
    C  smTr              :: tracer 1rst & 2nd Order moments
    C  gTracer           :: tendency array

    Returns (smTr, gTracer). Arrays: deltaTLev (1:Nr, not tiled), uFld, vFld, wFld, tracer, gTracer and each smTr[n-1]
    FArrays (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, 1:Nr); `implicitAdvection`, the schemes and `tracerIdentity` are static;
    myTime and myIter are only used by the diagnostics and debug branches (not ported). `grid` holds the GRID.h fields
    read here (dxG, dyG, rA, recip_rA, maskInC, hFacC, hFacW, hFacS, recip_hFacC, maskC, drF, recip_drF, deepFacC,
    deepFac2C, recip_deepFac2C, deepFac2F), `params` the PARAMS.h arrays (rhoFacC, recip_rhoFacC, rhoFacF).

    Ported branches (advect_xy: T scheme 80; advect_xz: S scheme 81): explicit advection, not a cubed sphere (npass = 2,
    X then Y; GAD_SOM_PREP_CS_CORNER not called), schemes 80 and 81 (limiter = MOD(scheme,10)), a linear free surface
    (:595-605, rTrans not masked); PTRACERS lane (tutorial_tracer_adjsens/input_ad.som81): noFlowAcrossSurf
    (rigidLid, nonlinFreeSurf >= 1 or select_rStar /= 0: :575-593). Not ported (raise): implicitAdvection (:215-220, a Fortran STOP), vertAdvecScheme .NE.
    advectionScheme (:221-226, STOP), useCubedSphereExchange (:229-244, :316-333, :356-362, :403-409, :477-486),
    diagnostics with useDiagnostics
    (:170-184, :460-475, :658-670; pkg/diagnostics is not ported), a build without GAD_ALLOW_TS_SOM_ADV and
    PTRACERS_ALLOW_DYN_STATE (:82, an empty routine). The CADJ lines are TAF directives (no forward effect); smCorners
    (:205-213) is set and used only on a cubed sphere.

    The horizontal k loop (:268-489) is a level scan (KERNEL_GUIDE §4: GAD_SOM_ADV_X/_Y take 2-D sections; each
    level reads and writes only its own level of smVol, smTr0 and smTr; afx/afy, and smCorners on the cube, are
    carried). The vertical loop (:517-673, k = Nr..1) is a recursion through the kUp/kDown flux slots and the moments
    at k-1: a level scan in the Fortran order, k = Nr and k = 1 static. The point loops run on whole (i,j) ranges.
    `0.` and `1.` are exact REAL*4 literals.
    """
    if not (cfg.GAD_ALLOW_TS_SOM_ADV or cfg.PTRACERS_ALLOW_DYN_STATE):
        raise NotImplementedError("GAD_SOM_ADVECT: compiled without GAD_ALLOW_TS_SOM_ADV and PTRACERS_ALLOW_DYN_STATE "
                                  "(an empty routine, gad_som_advect.F:82)")
    if cfg.ALLOW_DIAGNOSTICS and cfg.useDiagnostics:                # :170-184
        raise NotImplementedError("GAD_SOM_ADVECT: diagnostics (pkg/diagnostics is not ported)")
    sNx, sNy, OLx, OLy, Nr = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy, cfg.Nr
    dxG, dyG, rA, recip_rA, maskInC = grid.dxG, grid.dyG, grid.rA, grid.recip_rA, grid.maskInC
    hFacC, hFacW, hFacS, recip_hFacC, maskC = grid.hFacC, grid.hFacW, grid.hFacS, grid.recip_hFacC, grid.maskC
    drF, recip_drF = grid.drF, grid.recip_drF
    deepFacC, deepFac2C, recip_deepFac2C, deepFac2F = (grid.deepFacC, grid.deepFac2C, grid.recip_deepFac2C,
                                                       grid.deepFac2F)
    rhoFacC, recip_rhoFacC, rhoFacF = params.rhoFacC, params.recip_rhoFacC, params.rhoFacF

    jA, iA = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)

#--   Set up work arrays with valid (i.e. not NaN) values            # :186-213
    afx = rA.local("afx")
    afy = rA.local("afy")
    afx = afx.at[iA, jA].set(0.)
    afy = afy.at[iA, jA].set(0.)

    if implicitAdvection:                                           # :215-220
        raise ValueError("S/R GAD_SOM_ADVECT: not coded for implicit-vertical Advection "
                         "(STOP 'ABNORMAL END: S/R GAD_SOM_ADVECT')")
    if vertAdvecScheme != advectionScheme:                          # :221-226
        raise ValueError("S/R GAD_SOM_ADVECT: not coded for different vertAdvecScheme "
                         "(STOP 'ABNORMAL END: S/R GAD_SOM_ADVECT')")

#--   Set tile-specific parameters for horizontal fluxes
    cube = bool(cfg.useCubedSphereExchange)
    if cube:                                                        # :229-244 (ADVECT lane, M2: advect_cs)
        if not cfg.ALLOW_EXCH2:
            raise NotImplementedError("GAD_SOM_ADVECT: the cube without ALLOW_EXCH2 (nCFace = bi) is not ported")
        if advectionScheme % 10 != 0:
            raise NotImplementedError("GAD_SOM_ADVECT: the cube passes with the SOM limiter (scheme 81: the limiter "
                                      "loop's own strip range, gad_som_adv_x.F:164-185) are not ported")
        from mitjax.pkg.generic_advdiff.gad_advection import _cs_tile_flags
        cs = _cs_tile_flags(grid)                                   # :232-237
        smCorners = {c: jnp.zeros((tracer.ntiles, 2 + nSOM, OLy, OLx), tracer.dtype)   # :205-213  0.
                     for c in ("SW", "SE", "NE", "NW")}
    npass = 2                                                       # :245-252
    N_edge = False
    S_edge = False
    E_edge = False
    W_edge = False

    limiter = advectionScheme % 10                                  # :254

    smVol = tracer.local("smVol")                                   # :116-117 local 3-D arrays
    smTr0 = tracer.local("smTr0")
    smTr = list(smTr)
#--   Start of k loop for horizontal fluxes
    # :268-489 as a level scan (KERNEL_GUIDE §4; no level branches); `c` carries what the levels share
    def level_h(k, c):
        smVol, smTr0, smTr, afx, afy, smCorners = c
        smTr = list(smTr)

#--   Get temporary terms used by tendency routines
        xA = dyG[iA, jA]*deepFacC[k]*drF[k]*hFacW[iA, jA, k]         # :271-278
        yA = dxG[iA, jA]*deepFacC[k]*drF[k]*hFacS[iA, jA, k]
#--   Calculate "volume transports" through tracer cell faces.
#     anelastic: scaled by rhoFacC (~ mass transport)
        uTrans = FArray(uFld[iA, jA, k]*xA*rhoFacC[k], "uTrans",     # :281-286
                        i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
        vTrans = FArray(vFld[iA, jA, k]*yA*rhoFacC[k], "vTrans",
                        i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))

#--   grid-box volume and tracer content (zero order moment)        # :288-299
        smVol = smVol.at[iA, jA, k].set(rA[iA, jA]*deepFac2C[k]
                                        * drF[k]*hFacC[iA, jA, k]
                                        * rhoFacC[k])
        smTr0 = smTr0.at[iA, jA, k].set(tracer[iA, jA, k]*smVol[iA, jA, k])
#-    fill empty grid-box:
        smVol = smVol.at[iA, jA, k].set(smVol[iA, jA, k]
                                        + (1.0 - maskC[iA, jA, k]))

        if cube:                                                    # :309-439 (ADVECT lane, M2)
            smVol, smTr0, smTr, afx, afy, smCorners = _som_cs_passes(
                cs, k, limiter, advectionScheme, deltaTLev[k], uTrans, vTrans, maskInC, smVol, smTr0, smTr,
                afx, afy, smCorners, cfg)
#--   Multiple passes for different directions on different tiles
        for ipass in (range(1, npass+1) if not cube else ()):      # :309-439
            interiorOnly = False
            overlapOnly = False
#-    not CubedSphere
            calc_fluxes_X = ipass % 2 == 1
            calc_fluxes_Y = not calc_fluxes_X

#--   X direction
            if calc_fluxes_X and (not overlapOnly or N_edge or S_edge):     # :351-390
                if advectionScheme in (ENUM_SOM_PRATHER, ENUM_SOM_LIMITER):
                    out = gad_som_adv_x(k, limiter, overlapOnly, interiorOnly, N_edge, S_edge, E_edge, W_edge,
                                        deltaTLev[k], uTrans, maskInC,
                                        _level(smVol, k, cfg), _level(smTr0, k, cfg),
                                        *(_level(smTr[n-1], k, cfg) for n in range(1, nSOM+1)),
                                        afx, cfg=cfg)
                    smVol = _set_level(smVol, k, out[0], cfg)
                    smTr0 = _set_level(smTr0, k, out[1], cfg)
                    for n in range(1, nSOM+1):
                        smTr[n-1] = _set_level(smTr[n-1], k, out[1+n], cfg)
                    afx = out[11]
                else:
                    raise ValueError("GAD_SOM_ADVECT: adv. scheme incompatibale with SOM")

#--   Y direction
            if calc_fluxes_Y and (not overlapOnly or E_edge or W_edge):     # :398-437
                if advectionScheme in (ENUM_SOM_PRATHER, ENUM_SOM_LIMITER):
                    out = gad_som_adv_y(k, limiter, overlapOnly, interiorOnly, N_edge, S_edge, E_edge, W_edge,
                                        deltaTLev[k], vTrans, maskInC,
                                        _level(smVol, k, cfg), _level(smTr0, k, cfg),
                                        *(_level(smTr[n-1], k, cfg) for n in range(1, nSOM+1)),
                                        afy, cfg=cfg)
                    smVol = _set_level(smVol, k, out[0], cfg)
                    smTr0 = _set_level(smTr0, k, out[1], cfg)
                    for n in range(1, nSOM+1):
                        smTr[n-1] = _set_level(smTr[n-1], k, out[1+n], cfg)
                    afy = out[11]
                else:
                    raise ValueError("GAD_SOM_ADVECT: adv. scheme incompatibale with SOM")
#--   End of ipass loop
#--   End of K loop for horizontal fluxes
        return smVol, smTr0, tuple(smTr), afx, afy, smCorners

    if not cube:
        smCorners = None
    smVol, smTr0, smTr, afx, afy, smCorners = scan_levels(level_h, (smVol, smTr0, tuple(smTr), afx, afy, smCorners),
                                                          1, Nr)
    smTr = list(smTr)

    noFlowAcrossSurf = (cfg.rigidLid or cfg.nonlinFreeSurf >= 1    # :493-494
                        or cfg.select_rStar != 0)

#--   Apply limiter (if any):                                       # :496-514
    out = gad_som_lim_r(limiter, smVol, smTr0, *smTr, cfg=cfg)
    smVol, smTr0, smTr = out[0], out[1], list(out[2:])

    fluxR = [_local_kupdw(tracer, name, cfg) for name in _FLUX_R]  # :118-141 local (i,j,2) arrays
    rTrans = rA.local("rTrans")
    maskUp = rA.local("maskUp")
    afr = rA.local("afr")
#--   Start of k loop for vertical flux
    # :517-673 DO k=Nr,1,-1: k = Nr and k = 1 static (their branches), k = Nr-1..2 a level scan (KERNEL_GUIDE §4)
    def level_r(k, c):
        smVol, smTr0, smTr, fluxR, afr, rTrans, maskUp, gTracer = c
        smTr, fluxR = list(smTr), list(fluxR)
#--   kUp    Cycles through 1,2 to point to w-layer above
#--   kDown  Cycles through 2,1 to point to w-layer below
        kUp = 1+(Nr-k) % 2
        kDown = 1+(Nr-k+1) % 2
        if k == Nr:                                                 # :534-564
#--   Set advective fluxes at the very bottom:
            fluxR = [a.at[iA, jA, kDown].set(0.0) for a in fluxR]

#-- Compute Vertical transport
        if noFlowAcrossSurf and k == 1:                             # :575-582 (PTRACERS lane: som81)
#- Surface interface :
            rTrans = rTrans.at[iA, jA].set(0.)                      # 0. (REAL*4, exact)
            maskUp = maskUp.at[iA, jA].set(0.)
        elif noFlowAcrossSurf:                                      # :584-593 (PTRACERS lane: som81)
#- Interior interface :
            rTrans = rTrans.at[iA, jA].set(wFld[iA, jA, k]*rA[iA, jA]
                                           * deepFac2F[k]*rhoFacF[k]
                                           * maskC[iA, jA, k-1])
            maskUp = maskUp.at[iA, jA].set(1.)                      # 1. (REAL*4, exact)
        else:                                                       # :595-607
#- Linear Free-Surface: do not mask rTrans :
            km1 = max(k-1, 1)  # MINMAX-INT: integer (no tie or NaN case)
            rTrans = rTrans.at[iA, jA].set(wFld[iA, jA, k]*rA[iA, jA]
                                           * deepFac2F[k]*rhoFacF[k])
            maskUp = maskUp.at[iA, jA].set(maskC[iA, jA, km1]*maskC[iA, jA, k])

#-    Compute vertical advective flux in the interior:
        if vertAdvecScheme in (ENUM_SOM_PRATHER, ENUM_SOM_LIMITER):  # :610-634
            out = gad_som_adv_r(k, kUp, kDown, deltaTLev[k], rTrans, maskUp, maskInC,
                                smVol, smTr0, *smTr, *fluxR, afr, cfg=cfg)
            smVol, smTr0, smTr = out[0], out[1], list(out[2:11])
            fluxR, afr = list(out[11:35]), out[35]
        else:
            raise ValueError("GAD_SOM_ADVECT: adv. scheme incompatibale with SOM")

#--   Compute new tracer value and store tracer tendency           # :636-656
#--  Non-Lin Free-Surf: consistent with rescaling of tendencies
#     (in FREESURF_RESCALE_G) and RealFreshFlux/addMass.
        gTracer = gTracer.at[iA, jA, k].set(
            (smTr0[iA, jA, k] - tracer[iA, jA, k]*smVol[iA, jA, k])
            * recip_rA[iA, jA]*recip_deepFac2C[k]
            * recip_drF[k]*recip_hFacC[iA, jA, k]
            * recip_rhoFacC[k]
            / deltaTLev[k])
#--   End of k loop for vertical flux
        return smVol, smTr0, tuple(smTr), tuple(fluxR), afr, rTrans, maskUp, gTracer

    c = scan_levels(level_r, (smVol, smTr0, tuple(smTr), tuple(fluxR), afr, rTrans, maskUp, gTracer), 1, Nr,
                    down=True, peel=(1, 1))
    smVol, smTr0, smTr, fluxR, afr, rTrans, maskUp, gTracer = c

    return tuple(smTr), gTracer
