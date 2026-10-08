"""GAD_ADVECTION: pkg/generic_advdiff/gad_advection.F @63cdc0b, the multi-dimensional advection of one tracer
(direction splitting X, Y on the horizontal levels, then the vertical fluxes)."""

from mitjax.eesupp.fill_cs_corner_tr_rl import fill_cs_corner_tr_rl
from mitjax.eesupp.fill_cs_corner_uv_rs import fill_cs_corner_uv_rs
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.grid import oneRS
from mitjax.ops.scan_k import scan_levels
from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_kernel_cfg
from mitjax.pkg.generic_advdiff.gad_cs_passes import NPASS_CS
from mitjax.pkg.generic_advdiff.gad_dst3_adv_x import gad_dst3_adv_x
from mitjax.pkg.generic_advdiff.gad_dst3_adv_y import gad_dst3_adv_y
from mitjax.pkg.generic_advdiff.gad_dst3fl_adv_x import gad_dst3fl_adv_x
from mitjax.pkg.generic_advdiff.gad_dst3fl_adv_y import gad_dst3fl_adv_y
from mitjax.pkg.generic_advdiff.gad_fluxlimit_adv_x import gad_fluxlimit_adv_x
from mitjax.pkg.generic_advdiff.gad_fluxlimit_adv_y import gad_fluxlimit_adv_y
from mitjax.pkg.generic_advdiff.gad_h import (ENUM_DST2, ENUM_DST3, ENUM_DST3_FLUX_LIMIT, ENUM_FLUX_LIMIT, ENUM_OS7MP,
                                              ENUM_PPM_MONO_LIMIT, ENUM_PPM_NULL_LIMIT, ENUM_PPM_WENO_LIMIT,
                                              ENUM_PQM_MONO_LIMIT, ENUM_PQM_NULL_LIMIT, ENUM_PQM_WENO_LIMIT,
                                              ENUM_UPWIND_1RST)
from mitjax.pkg.generic_advdiff.gad_ppm_adv_r import gad_ppm_adv_r
from mitjax.pkg.generic_advdiff.gad_ppm_adv_x import gad_ppm_adv_x
from mitjax.pkg.generic_advdiff.gad_ppm_adv_y import gad_ppm_adv_y
from mitjax.pkg.generic_advdiff.gad_pqm_adv_r import gad_pqm_adv_r
from mitjax.pkg.generic_advdiff.gad_pqm_adv_x import gad_pqm_adv_x
from mitjax.pkg.generic_advdiff.gad_pqm_adv_y import gad_pqm_adv_y

_OPT = "GAD_OPTIONS.h"      # gad_advection.F:1  #include "GAD_OPTIONS.h"

_PPM = (ENUM_PPM_NULL_LIMIT, ENUM_PPM_MONO_LIMIT, ENUM_PPM_WENO_LIMIT)
_PQM = (ENUM_PQM_NULL_LIMIT, ENUM_PQM_MONO_LIMIT, ENUM_PQM_WENO_LIMIT)


def _level(A, k):
    """A(1-OLx,1-OLy,k) passed to a 2-D dummy argument (sequence association, KERNEL_GUIDE §4): level k."""
    (_, ilo, ihi), (_, jlo, jhi), (_, klo, _) = A.dims
    kk = k - klo                                                    # k: a Python int or a traced level (KIdx)
    return FArray(A.data[:, getattr(kk, "value", kk)], A.name, i=(ilo, ihi), j=(jlo, jhi), tiled=A.tiled)


def _local2(like, name):
    """A local `name(1-OLx:sNx+OLx,1-OLy:sNy+OLy)` of the tiles of `like` (3-D), every point NaN."""
    return _level(like, like.dims[2][1]).local(name)


def _local_kupdw(like, name):
    """A local `_RL name(1-OLx:sNx+OLx,1-OLy:sNy+OLy,2)` (the kUp/kDown slots), every point NaN."""
    import jax.numpy as jnp
    (_, ilo, ihi), (_, jlo, jhi), _ = like.dims
    d = jnp.full((like.data.shape[0], 2) + tuple(like.data.shape[2:]), jnp.nan, like.data.dtype)
    return FArray(d, name, i=(ilo, ihi), j=(jlo, jhi), k=(1, 2), tiled=like.tiled)


def _scheme_x(advectionScheme, k, deltaT, uTrans, uFld_k, maskLocW, localTij, af, kc, grid):
    """The X-flux call of :393-427 for one scheme (static dispatch on advectionScheme)."""
    if advectionScheme in (ENUM_UPWIND_1RST, ENUM_DST2):                        # :406-410
        raise NotImplementedError("GAD_ADVECTION: GAD_DST2U1_ADV_X is not ported")
    if advectionScheme == ENUM_FLUX_LIMIT:                                      # :411-415
        return gad_fluxlimit_adv_x(k, True, deltaT, uTrans, uFld_k, maskLocW, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_DST3:                                            # :419-422 (GOADK lane)
        return gad_dst3_adv_x(k, True, deltaT, uTrans, uFld_k, maskLocW, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_DST3_FLUX_LIMIT:                                 # :424-427
        return gad_dst3fl_adv_x(k, True, deltaT, uTrans, uFld_k, maskLocW, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_OS7MP:                                           # :428-431 (M3 Task 30)
        from mitjax.pkg.generic_advdiff.gad_os7mp_adv_x import gad_os7mp_adv_x
        return gad_os7mp_adv_x(k, True, deltaT, uTrans, uFld_k, maskLocW, localTij, af, cfg=kc, grid=grid)
    if advectionScheme in _PPM:                                                 # :432-437
        return gad_ppm_adv_x(advectionScheme, k, True, deltaT, uFld_k, uTrans, localTij, af, cfg=kc, grid=grid)
    if advectionScheme in _PQM:                                                 # :438-443
        return gad_pqm_adv_x(advectionScheme, k, True, deltaT, uFld_k, uTrans, localTij, af, cfg=kc, grid=grid)
    raise ValueError("GAD_ADVECTION: adv. scheme incompatible with multi-dim")  # :445-446  STOP


def _scheme_y(advectionScheme, k, deltaT, vTrans, vFld_k, maskLocS, localTij, af, kc, grid):
    """The Y-flux call of :598-632 for one scheme (static dispatch on advectionScheme)."""
    if advectionScheme in (ENUM_UPWIND_1RST, ENUM_DST2):                        # :627-631
        raise NotImplementedError("GAD_ADVECTION: GAD_DST2U1_ADV_Y is not ported")
    if advectionScheme == ENUM_FLUX_LIMIT:                                      # :632-635
        return gad_fluxlimit_adv_y(k, True, deltaT, vTrans, vFld_k, maskLocS, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_DST3:                                            # :640-643 (GOADK lane)
        return gad_dst3_adv_y(k, True, deltaT, vTrans, vFld_k, maskLocS, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_DST3_FLUX_LIMIT:                                 # :644-647
        return gad_dst3fl_adv_y(k, True, deltaT, vTrans, vFld_k, maskLocS, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_OS7MP:                                           # :648-651 (M3 Task 30)
        from mitjax.pkg.generic_advdiff.gad_os7mp_adv_y import gad_os7mp_adv_y
        return gad_os7mp_adv_y(k, True, deltaT, vTrans, vFld_k, maskLocS, localTij, af, cfg=kc, grid=grid)
    if advectionScheme in _PPM:                                                 # :653-658
        return gad_ppm_adv_y(advectionScheme, k, True, deltaT, vFld_k, vTrans, localTij, af, cfg=kc, grid=grid)
    if advectionScheme in _PQM:                                                 # :659-664
        return gad_pqm_adv_y(advectionScheme, k, True, deltaT, vFld_k, vTrans, localTij, af, cfg=kc, grid=grid)
    raise ValueError("GAD_ADVECTION: adv. scheme incompatible with mutli-dim")  # :666-667  STOP


def _with_data(A, data):
    """A with its storage replaced (same declaration)."""
    return FArray(data, A.name, tiled=A.tiled, _dims=A.dims)


def _cs_tile_flags(g):
    """gad_advection.F:254-259 on every tile: nCFace = exch2_myFace(myTile), [N,S,E,W]_edge = exch2_is*edge(myTile)
    .EQ.1 (myTile = W2_myTileList(bi,bj): the grid's W2 tile view, mitjax/model/grid.py), as [T,1,1] arrays; and
    the corner flags of FILL_CS_CORNER_* (fill_cs_corner_tr_rl.F:75-82: SW, SE, NW, NE) as [T,4]."""
    import jax.numpy as jnp
    nCFace = g.exch2_myFace.data
    N_edge, S_edge = g.exch2_isNedge.data == 1, g.exch2_isSedge.data == 1
    E_edge, W_edge = g.exch2_isEedge.data == 1, g.exch2_isWedge.data == 1
    corners = jnp.stack([(W_edge & S_edge)[:, 0, 0], (E_edge & S_edge)[:, 0, 0],
                         (W_edge & N_edge)[:, 0, 0], (E_edge & N_edge)[:, 0, 0]], axis=-1)
    return {"nCFace": nCFace, "N_edge": N_edge, "S_edge": S_edge, "E_edge": E_edge, "W_edge": W_edge,
            "corners": corners}


def _cs_pass_flags(nCFace, ipass):
    """gad_advection.F:349-365 (the useCubedSphereExchange arm) on every tile: (overlapOnly, interiorOnly,
    calc_fluxes_X, calc_fluxes_Y) as [T,1,1] bool arrays (nCFace >= 1: MOD is the non-negative remainder)."""
    import jax.numpy as jnp
    m3 = jnp.mod(nCFace, 3)
    if ipass == 1:                                                              # :351-355
        overlapOnly = m3 == 0
        interiorOnly = m3 != 0
        calc_fluxes_X = (nCFace == 6) | (nCFace == 1) | (nCFace == 2)
        calc_fluxes_Y = (nCFace == 3) | (nCFace == 4) | (nCFace == 5)
    elif ipass == 2:                                                            # :356-360
        overlapOnly = m3 == 2
        interiorOnly = m3 == 1
        calc_fluxes_X = (nCFace == 2) | (nCFace == 3) | (nCFace == 4)
        calc_fluxes_Y = (nCFace == 5) | (nCFace == 6) | (nCFace == 1)
    else:                                                                       # :361-364
        overlapOnly = jnp.zeros_like(nCFace, bool)                              # :347 (not set in pass 3)
        interiorOnly = jnp.ones_like(nCFace, bool)
        calc_fluxes_X = (nCFace == 5) | (nCFace == 6)
        calc_fluxes_Y = (nCFace == 2) | (nCFace == 3)
    return overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y


def _cs_fill(fill4dir, A, flag, cs, kc):
    """FILL_CS_CORNER_TR_RL( fill4dir, .FALSE., A ) on the tiles whose [T,1,1] `flag` is set (the others skip the
    call: their corner flags are cleared)."""
    corners = cs["corners"] & flag[:, 0, 0, None]
    return _with_data(A, fill_cs_corner_tr_rl(fill4dir, False, A.data, corners, True, sNx=kc.sNx, sNy=kc.sNy,
                                              OLx=kc.OLx, OLy=kc.OLy))


def _cs_passes(cs, k, deltaT, advectionScheme, uTrans, vTrans, uFld_k, vFld_k, maskLocW, maskLocS,
               localTij, localVol, af, tracer, compressible, kc, g, recip_rhoFacC):
    """The three cube passes of one level (gad_advection.F:342-810), every tile at once: each tile's flags
    (_cs_pass_flags) select, per tile, the corner fills (FILL_CS_CORNER_TR_RL), whether the fluxes are computed (af
    keeps its value otherwise, as the Fortran's local does), and the points the update writes (the overlapOnly /
    interiorOnly ranges of :548-578 and :771-799 with the edges, as [T,j,i] masks): the update statement is evaluated
    on its full DO range and written only where the Fortran's loops run (each point reads only af and the transports
    at itself and its i+1 / j+1 neighbour and its own localTij, localVol: a pointwise statement)."""
    import jax.numpy as jnp
    sNx, sNy, OLx, OLy = kc.sNx, kc.sNy, kc.OLx, kc.OLy
    N_edge, S_edge, E_edge, W_edge = cs["N_edge"], cs["S_edge"], cs["E_edge"], cs["W_edge"]
    jj = jnp.arange(1-OLy, sNy+OLy+1)[None, :, None]                            # Fortran j of each storage row
    ii = jnp.arange(1-OLx, sNx+OLx+1)[None, None, :]
    jA, iA = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)
    for ipass in range(1, NPASS_CS+1):                                          # :342
        overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y = _cs_pass_flags(cs["nCFace"], ipass)

#--   X direction                                                         # :384-591
        doFlux = calc_fluxes_X & (~overlapOnly | N_edge | S_edge)               # :384, :389
        localTij = _cs_fill(1, localTij, doFlux & overlapOnly, cs, kc)          # :392-395
        afX = af.at[iA, jA].set(0.)                                             # :398-403  0.
        afX = _scheme_x(advectionScheme, k, deltaT, uTrans, uFld_k,             # :406-447
                        maskLocW, localTij, afX, kc, g)
        af = _with_data(af, jnp.where(doFlux, afX.data, af.data))
        localTij = _cs_fill(2, localTij, doFlux & overlapOnly & (ipass == 1), cs, kc)   # :459-462
        # update ranges (:474-578)
        iMinUpd = jnp.where(overlapOnly & W_edge, 1, 1-OLx+1)                   # :475, :479
        iMaxUpd = jnp.where(overlapOnly & E_edge, sNx, sNx+OLx-1)               # :476, :480
        ovl = ((S_edge & (jj <= 0)) | (N_edge & (jj >= sNy+1))) & (ii >= iMinUpd) & (ii <= iMaxUpd)   # :482-547
        jMinUpd = jnp.where(interiorOnly & S_edge, 1, 1-OLy)                    # :550-553
        jMaxUpd = jnp.where(interiorOnly & N_edge, sNy, sNy+OLy)
        full = (jj >= jMinUpd) & (jj <= jMaxUpd) & (ii >= 1-OLx+1) & (ii <= sNx+OLx-1)    # :554-555
        upd = calc_fluxes_X & jnp.where(overlapOnly, ovl, full)
        j = loop_j(1-OLy, sNy+OLy)
        i = loop_i(1-OLx+1, sNx+OLx-1)
        sel = upd[:, :, 1:-1]
        if compressible:                                                        # :507-519, :556-568
            tmpTrac = (localTij[i, j]*localVol[i, j]
                       - deltaT*(af[i+1, j] - af[i, j])
                       * g.maskInC[i, j])
            newVol = (localVol[i, j]
                      - deltaT*(uTrans[i+1, j] - uTrans[i, j])
                      * g.maskInC[i, j])
            localVol = localVol.at[i, j].set(jnp.where(sel, newVol, localVol[i, j]))
            localTij = localTij.at[i, j].set(jnp.where(sel, tmpTrac/newVol, localTij[i, j]))
        else:                                                                   # :520-526, :569-576
            newTij = (localTij[i, j]
                      - deltaT*recip_rhoFacC[k]
                      * g.recip_hFacC[i, j, k]*g.recip_drF[k]
                      * g.recip_rA[i, j]*g.recip_deepFac2C[k]
                      * (af[i+1, j]-af[i, j]
                         - tracer[i, j, k]*(uTrans[i+1, j]-uTrans[i, j])
                         )*g.maskInC[i, j])
            localTij = localTij.at[i, j].set(jnp.where(sel, newTij, localTij[i, j]))

#--   Y direction                                                         # :605-808
        doFlux = calc_fluxes_Y & (~overlapOnly | E_edge | W_edge)               # :605, :610
        localTij = _cs_fill(2, localTij, doFlux & overlapOnly, cs, kc)          # :613-616
        afY = af.at[iA, jA].set(0.)                                             # :619-624  0.
        afY = _scheme_y(advectionScheme, k, deltaT, vTrans, vFld_k,             # :627-668
                        maskLocS, localTij, afY, kc, g)
        af = _with_data(af, jnp.where(doFlux, afY.data, af.data))
        localTij = _cs_fill(1, localTij, doFlux & overlapOnly & (ipass == 1), cs, kc)   # :680-683
        jMinUpd = jnp.where(overlapOnly & S_edge, 1, 1-OLy+1)                   # :696, :700
        jMaxUpd = jnp.where(overlapOnly & N_edge, sNy, sNy+OLy-1)               # :697, :701
        ovl = ((W_edge & (ii <= 0)) | (E_edge & (ii >= sNx+1))) & (jj >= jMinUpd) & (jj <= jMaxUpd)   # :703-768
        iMinUpd = jnp.where(interiorOnly & W_edge, 1, 1-OLx)                    # :773-776
        iMaxUpd = jnp.where(interiorOnly & E_edge, sNx, sNx+OLx)
        full = (ii >= iMinUpd) & (ii <= iMaxUpd) & (jj >= 1-OLy+1) & (jj <= sNy+OLy-1)    # :777-778
        upd = calc_fluxes_Y & jnp.where(overlapOnly, ovl, full)
        j = loop_j(1-OLy+1, sNy+OLy-1)
        i = loop_i(1-OLx, sNx+OLx)
        sel = upd[:, 1:-1, :]
        if compressible:                                                        # :720-721, :752-753, :779-791
            tmpTrac = (localTij[i, j]*localVol[i, j]
                       - deltaT*(af[i, j+1] - af[i, j])
                       * g.maskInC[i, j])
            newVol = (localVol[i, j]
                      - deltaT*(vTrans[i, j+1] - vTrans[i, j])
                      * g.maskInC[i, j])
            localVol = localVol.at[i, j].set(jnp.where(sel, newVol, localVol[i, j]))
            localTij = localTij.at[i, j].set(jnp.where(sel, tmpTrac/newVol, localTij[i, j]))
        else:                                                                   # :792-799
            newTij = (localTij[i, j]
                      - deltaT*recip_rhoFacC[k]
                      * g.recip_hFacC[i, j, k]*g.recip_drF[k]
                      * g.recip_rA[i, j]*g.recip_deepFac2C[k]
                      * (af[i, j+1]-af[i, j]
                         - tracer[i, j, k]*(vTrans[i, j+1]-vTrans[i, j])
                         )*g.maskInC[i, j])
            localTij = localTij.at[i, j].set(jnp.where(sel, newTij, localTij[i, j]))
    return localTij, localVol, af


def gad_advection(implicitAdvection, advectionSchArg, vertAdvecSchArg, trIdentity, deltaTLev,
                  uFld, vFld, wFld, tracer, gTracer, myTime, myIter, *, cfg, grid, params):
    """GAD_ADVECTION( implicitAdvection, advectionSchArg, vertAdvecSchArg, trIdentity, deltaTLev,
                      uFld, vFld, wFld, tracer, gTracer, bi,bj, myTime,myIter,myThid )
    @63cdc0b pkg/generic_advdiff/gad_advection.F:10-1100

    C Calculates the tendency of a tracer due to advection.
    C It uses the multi-dimensional method given in \\ref{sect:multiDimAdvection}
    C and can only be used for the non-linear advection schemes such as the
    C direct-space-time method and flux-limiters.
    C  theta^(n+1/3) = theta^(n)     - dt ( d_x (u theta^(n))     - theta^(n) d_x u )
    C  theta^(n+2/3) = theta^(n+1/3) - dt ( d_y (v theta^(n+1/3)) - theta^(n) d_y v )
    C  theta^(n+3/3) = theta^(n+2/3) - dt ( d_r (w theta^(n+2/3)) - theta^(n) d_r w )
    C  G_theta = ( theta^(n+3/3) - theta^(n) )/dt
    C The tendency (output) is over-written by this routine.
    C  implicitAdvection :: implicit vertical advection (later on)
    C  advectionSchArg   :: advection scheme to use (Horizontal plane)
    C  vertAdvecSchArg   :: advection scheme to use (vertical direction)
    C  trIdentity        :: tracer identifier
    C  uFld, vFld, wFld  :: Advection velocity field, 3 components
    C  tracer            :: tracer field
    C  gTracer           :: tendency array

    Returns gTracer. uFld, vFld, wFld, gTracer: (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr) FArrays; tracer: the State field
    (same declaration, all tiles); deltaTLev(Nr) not tiled. `implicitAdvection`, the schemes and `trIdentity` are
    static; myTime, myIter are read only by the debug branch (:854-862, not ported).

    Ported (the M1 advect variants): a grid without the cubed-sphere exchange (npass = 2, :268-275: overlapOnly,
    interiorOnly and the N/S/E/W edges are .FALSE., so FILL_CS_CORNER_* are not called and the update ranges are
    the full ones of :548-553 and :771-776); both GAD_MULTIDIM_COMPRESSIBLE arms (advect_xy defines it, advect_xz
    does not); the X/Y schemes 77 (GAD_FLUXLIMIT_ADV_*), 33 (GAD_DST3FL_ADV_*), 40-42 (GAD_PPM_ADV_*), 50-52
    (GAD_PQM_ADV_*); the vertical step for PPM/PQM (GAD_PPM_ADV_R / GAD_PQM_ADV_R on the 3-D field, :884-917, copied
    level by level :1009-1015) and for implicitAdvection (the tendency of the horizontal passes only, :819-831);
    the surface interface (k = 1) with any scheme. GOADK lane: the X/Y fluxes of scheme 30 (GAD_DST3_ADV_X/Y,
    replay-gated in reference/replay_goadk; global_ocean.90x40x15/code_ad) and the ALLOW_AUTODIFF arms: localTij
    (and localVol) = 0 before the k loop (:242-247), af = 0 before every X and Y block whatever calc_fluxes_X/Y
    (:375-382, :596-603, in place of the reset inside the IF, :398-403, :619-624), fVerT = 0 before the vertical
    loop (:928-935), PPM/PQM not compiled (the scheme STOP); the inAdMode scheme swap (:193-205) never runs in a
    forward step.
    M3 Task 30: scheme 7 (GAD_OS7MP_ADV_X/Y, :428-431 / :648-651, and GAD_OS7MP_ADV_R for the interior interfaces,
    :1003-1006; tutorial_reentrant_channel). Lane M4ADLAB (lab_sea/input_ad): the vertical DST3 of the interior
    interfaces (GAD_DST3_ADV_R, :995-998; levels 1, 2 and Nr peeled from the level scan).
    Raise: the other schemes (DST2/U1; the vertical kernels GAD_*_ADV_R of DST2/U1 and the flux limiter for the
    interior interfaces), useCubedSphereExchange, ALLOW_OBCS with useOBCS
    (OBCS_U1_ADV_TRACER and the maskIn masks of :323-326), ALLOW_AIM, useDiagnostics (DIAGNOSTICS_FILL /
    LAYERS_FILL: output only), implicitAdvection with GAD_MULTIDIM_COMPRESSIBLE (a Fortran STOP, :820-823).

    Loops: the horizontal k loop (:295-870) treats each level on its own (each level reads and writes only its
    own level of localT3d / locVol3d and the 2-D work arrays it re-initialises); it calls level routines with 2-D
    sections, so it is a level scan in the Fortran's order (KERNEL_GUIDE §4). The vertical loop DO
    k=Nr,1,-1 (:938-1096) carries rTrans and the kUp/kDown slots of fVerT from one level to the next: also a
    level scan in the Fortran's order, the levels with static branches peeled. All point loops are independent per point (pure stencil reads of
    arrays written in an earlier statement), written on their DO ranges. `0.` and `1.` are REAL*4 literals
    (exact).
    """
    if cfg.cpp.flag("DISABLE_MULTIDIM_ADVECTION", _OPT):                       # :91-92
        raise ValueError("GAD_ADVECTION is empty with DISABLE_MULTIDIM_ADVECTION (STOP)")
    ad = bool(cfg.cpp.flag("ALLOW_AUTODIFF", _OPT))                          # GOADK lane: the AD arms below
    if cfg.cpp.flag("ALLOW_OBCS", _OPT) and params.useOBCS:
        raise NotImplementedError("GAD_ADVECTION: OBCS (maskInW/S, OBCS_U1_ADV_TRACER) is not ported")
    if cfg.cpp.flag("ALLOW_AIM", _OPT):
        raise NotImplementedError("GAD_ADVECTION: the ALLOW_AIM vertical-transport condition (:953-960) is not "
                                  "ported")
    if cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics:
        raise NotImplementedError("GAD_ADVECTION: DIAGNOSTICS_FILL (useDiagnostics) is not ported")
    if cfg.cpp.ALLOW_DEPTH_CONTROL:
        raise NotImplementedError("GAD_ADVECTION: ALLOW_DEPTH_CONTROL (_hFacW, _recip_hFacC macros) is not ported")
    compressible = cfg.cpp.flag("GAD_MULTIDIM_COMPRESSIBLE", _OPT)
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    kc = gad_kernel_cfg(cfg)
    g = grid
    rhoFacC, recip_rhoFacC, rhoFacF = params.rhoFacC, params.recip_rhoFacC, params.rhoFacF

    advectionScheme = advectionSchArg                                           # :191
    vertAdvecScheme = vertAdvecSchArg                                           # :192
    # :193-205 (ALLOW_AUTODIFF_TAMC) IF ( inAdMode .AND. useApproxAdvectionInAdMode ): inAdMode is .FALSE. in every
    # forward step (AUTODIFF_INADMODE_UNSET, forward_step.F:433-435), so the scheme swap never runs here
    if ad:                                                                      # GOADK lane: PPM/PQM not compiled
        for sch in (advectionScheme, vertAdvecScheme):
            if sch in _PPM or sch in _PQM:                                      # :431-444, :652-665, :877-919
                raise ValueError("GAD_ADVECTION: adv. scheme incompatible with multi-dim (STOP: PPM/PQM are not "
                                 "compiled with ALLOW_AUTODIFF)")

    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    maskLocW = _local2(uFld, "maskLocW")
    maskLocS = _local2(uFld, "maskLocS")
    xA = _local2(uFld, "xA")
    yA = _local2(uFld, "yA")
    uTrans = _local2(uFld, "uTrans")
    vTrans = _local2(uFld, "vTrans")
    rTrans = _local2(uFld, "rTrans")
    rTransKp = _local2(uFld, "rTransKp")
    af = _local2(uFld, "af")
    localTij = _local2(uFld, "localTij")
    fVerT = _local_kupdw(uFld, "fVerT")
    localT3d = uFld.local("localT3d")
    localVol = _local2(uFld, "localVol")
    locVol3d = uFld.local("locVol3d")

#--   Set up work arrays with valid (i.e. not NaN) values                # :225-246
    rTrans = rTrans.at[iA, jA].set(0.)                                          # :239  0. _d 0
    fVerT = fVerT.at[iA, jA, 1].set(0.)                                         # :240
    fVerT = fVerT.at[iA, jA, 2].set(0.)                                         # :241
    if ad:                                                                      # :242-247 (GOADK lane)
        if compressible:
            localVol = localVol.at[iA, jA].set(0.)                              # :244  0. _d 0
        localTij = localTij.at[iA, jA].set(0.)                                  # :246  0. _d 0

#--   Set tile-specific parameters for horizontal fluxes                  # :248-275
    cube = bool(params.useCubedSphereExchange)
    if cube:                                                                    # :252-267 (ADVECT lane, M2)
        if not cfg.cpp.ALLOW_EXCH2:
            raise NotImplementedError("GAD_ADVECTION: the cube without ALLOW_EXCH2 (nCFace = bi) is not ported")
        npass = 3                                                               # :253
        cs = _cs_tile_flags(g)                                                  # :254-259 nCFace, [N,S,E,W]_edge
        af = af.at[iA, jA].set(0.)      # finite masked lanes: tiles that skip a flux keep af, never read (below)
    else:
        npass = 2                                                               # :269
    overlapOnly = False                                                         # :346-347, :367-369 (not CubedSphere)
    interiorOnly = False

#--   Start of k loop for horizontal fluxes                                # :295
    # DO k=1,Nr as a level scan (KERNEL_GUIDE §4; no level branches); `c` carries what the levels share
    def level_h(k, c):
        xA, yA, uTrans, vTrans, localTij, localVol, maskLocW, maskLocS, af, localT3d, locVol3d, gTracer = c
#--   Get temporary terms used by tendency routines                       # :297-305
        xA = xA.at[iA, jA].set(g.dyG[iA, jA]*g.deepFacC[k]
                               * g.drF[k]*g.hFacW[iA, jA, k])
        yA = yA.at[iA, jA].set(g.dxG[iA, jA]*g.deepFacC[k]
                               * g.drF[k]*g.hFacS[iA, jA, k])
#--   Calculate "volume transports" through tracer cell faces.            # :306-313
        uTrans = uTrans.at[iA, jA].set(uFld[iA, jA, k]*xA[iA, jA]*rhoFacC[k])
        vTrans = vTrans.at[iA, jA].set(vFld[iA, jA, k]*yA[iA, jA]*rhoFacC[k])

#--   Make local copy of tracer array and mask West & South               # :315-332
        localTij = localTij.at[iA, jA].set(tracer[iA, jA, k])                   # :318
        if compressible:                                                        # :319-322
            localVol = localVol.at[iA, jA].set(
                g.rA[iA, jA]*g.deepFac2C[k]
                * rhoFacC[k]*g.drF[k]*g.hFacC[iA, jA, k]
                + (oneRS - g.maskC[iA, jA, k]))
        maskLocW = maskLocW.at[iA, jA].set(g.maskW[iA, jA, k])                  # :328
        maskLocS = maskLocS.at[iA, jA].set(g.maskS[iA, jA, k])                  # :329
        if cube:                                                                # :334-338
            withSigns = False
            mW, mS = fill_cs_corner_uv_rs(withSigns, maskLocW.data, maskLocS.data, cs["corners"], True,
                                          sNx=sNx, sNy=sNy, OLx=OLx, OLy=OLy)
            maskLocW = _with_data(maskLocW, mW)
            maskLocS = _with_data(maskLocS, mS)

        uFld_k, vFld_k = _level(uFld, k), _level(vFld, k)
        deltaT = deltaTLev[k]
        if cube:                                                                # :342-810 (ADVECT lane, M2)
            localTij, localVol, af = _cs_passes(
                cs, k, deltaT, advectionScheme, uTrans, vTrans, uFld_k, vFld_k, maskLocW, maskLocS,
                localTij, localVol, af, tracer, compressible, kc, g, recip_rhoFacC)
#--   Multiple passes for different directions on different tiles         # :342
        for ipass in (range(1, npass+1) if not cube else ()):
            calc_fluxes_X = ipass % 2 == 1                                      # :368  MOD(ipass,2).EQ.1
            calc_fluxes_Y = not calc_fluxes_X                                   # :369

#--   X direction
            if ad:                                                              # :375-382 (GOADK lane) always
                af = af.at[iA, jA].set(0.)                                      #  0.
            if calc_fluxes_X:                                                   # :384
                if not ad:
                    af = af.at[iA, jA].set(0.)                                  # :398-403  0.
                af = _scheme_x(advectionScheme, k, deltaT, uTrans, uFld_k,      # :406-447
                               maskLocW, localTij, af, kc, g)
                # overlapOnly is .FALSE. (not CubedSphere): update everywhere   # :548-578
                jMinUpd = 1-OLy                                                 # :550
                jMaxUpd = sNy+OLy                                               # :551
                j = loop_j(jMinUpd, jMaxUpd)                                    # :554
                i = loop_i(1-OLx+1, sNx+OLx-1)                                  # :555
                if compressible:                                                # :556-568
                    tmpTrac = (localTij[i, j]*localVol[i, j]
                               - deltaT*(af[i+1, j] - af[i, j])
                               * g.maskInC[i, j])
                    localVol = localVol.at[i, j].set(
                        localVol[i, j]
                        - deltaT*(uTrans[i+1, j] - uTrans[i, j])
                        * g.maskInC[i, j])
                    localTij = localTij.at[i, j].set(tmpTrac/localVol[i, j])
                else:                                                           # :569-576
                    localTij = localTij.at[i, j].set(
                        localTij[i, j]
                        - deltaT*recip_rhoFacC[k]
                        * g.recip_hFacC[i, j, k]*g.recip_drF[k]
                        * g.recip_rA[i, j]*g.recip_deepFac2C[k]
                        * (af[i+1, j]-af[i, j]
                           - tracer[i, j, k]*(uTrans[i+1, j]-uTrans[i, j])
                           )*g.maskInC[i, j])
                # :580-585 afx = af (kept for the diagnostics only)

#--   Y direction
            if ad:                                                              # :596-603 (GOADK lane) always
                af = af.at[iA, jA].set(0.)                                      #  0.
            if calc_fluxes_Y:                                                   # :605
                if not ad:
                    af = af.at[iA, jA].set(0.)                                  # :619-624  0.
                af = _scheme_y(advectionScheme, k, deltaT, vTrans, vFld_k,      # :627-668
                               maskLocS, localTij, af, kc, g)
                iMinUpd = 1-OLx                                                 # :773
                iMaxUpd = sNx+OLx                                               # :777
                j = loop_j(1-OLy+1, sNy+OLy-1)                                  # :774
                i = loop_i(iMinUpd, iMaxUpd)                                    # :778
                if compressible:                                                # :779-791
                    tmpTrac = (localTij[i, j]*localVol[i, j]
                               - deltaT*(af[i, j+1] - af[i, j])
                               * g.maskInC[i, j])
                    localVol = localVol.at[i, j].set(
                        localVol[i, j]
                        - deltaT*(vTrans[i, j+1] - vTrans[i, j])
                        * g.maskInC[i, j])
                    localTij = localTij.at[i, j].set(tmpTrac/localVol[i, j])
                else:                                                           # :792-799
                    localTij = localTij.at[i, j].set(
                        localTij[i, j]
                        - deltaT*recip_rhoFacC[k]
                        * g.recip_hFacC[i, j, k]*g.recip_drF[k]
                        * g.recip_rA[i, j]*g.recip_deepFac2C[k]
                        * (af[i, j+1]-af[i, j]
                           - tracer[i, j, k]*(vTrans[i, j+1]-vTrans[i, j])
                           )*g.maskInC[i, j])
                # :803-808 afy = af (kept for the diagnostics only)

        if implicitAdvection:                                                   # :819-831
            if compressible:
                raise ValueError("GAD_ADVECTION: missing code for implicitAdvection (STOP)")   # :822
            gTracer = gTracer.at[iA, jA, k].set(
                (localTij[iA, jA] - tracer[iA, jA, k])/deltaT)
        else:                                                                   # :832-839
            if compressible:
                locVol3d = locVol3d.at[iA, jA, k].set(localVol[iA, jA])         # :835
            localT3d = localT3d.at[iA, jA, k].set(localTij[iA, jA])             # :837
        return xA, yA, uTrans, vTrans, localTij, localVol, maskLocW, maskLocS, af, localT3d, locVol3d, gTracer

    c = (xA, yA, uTrans, vTrans, localTij, localVol, maskLocW, maskLocS, af, localT3d, locVol3d, gTracer)
    c = scan_levels(level_h, c, 1, Nr)
    xA, yA, uTrans, vTrans, localTij, localVol, maskLocW, maskLocS, af, localT3d, locVol3d, gTracer = c
#--   End of K loop for horizontal fluxes

    if implicitAdvection:                                                       # :875
        return gTracer

    usePPMvertAdv = (not ad) and vertAdvecScheme in _PPM                         # :878-880 (#ifndef ALLOW_AUTODIFF)
    usePQMvertAdv = (not ad) and vertAdvecScheme in _PQM                         # :881-883
    afr = None
    if usePPMvertAdv or usePQMvertAdv:                                          # :884-917
        rTran3d = uFld.local("rTran3d")
        for k in range(1, Nr+1):                                                # :886
            if k == 1:                                                          # :887-893
                rTran3d = rTran3d.at[iA, jA, k].set(0.)                         #  0. _d 0
            else:                                                               # :894-902
                rTran3d = rTran3d.at[iA, jA, k].set(wFld[iA, jA, k]*g.rA[iA, jA]
                                                    * g.deepFac2F[k]*rhoFacF[k]
                                                    * g.maskC[iA, jA, k-1])
        afr = uFld.local("afr")
        if usePPMvertAdv:                                                       # :904-910
            afr = gad_ppm_adv_r(vertAdvecScheme, deltaTLev, wFld, rTran3d, localT3d, afr, cfg=kc, grid=g)
        if usePQMvertAdv:                                                       # :911-916
            afr = gad_pqm_adv_r(vertAdvecScheme, deltaTLev, wFld, rTran3d, localT3d, afr, cfg=kc, grid=g)

    if ad:                                                                      # :928-935 (GOADK lane)
        fVerT = fVerT.at[iA, jA, 1].set(0.)                                     #  0.
        fVerT = fVerT.at[iA, jA, 2].set(0.)
#--   Start of k loop for vertical flux                                   # :938
    # DO k=Nr,1,-1 as a level scan (KERNEL_GUIDE §4): k = Nr and k = 1 static (their branches), and with
    # GAD_DST3FL_ADV_R also k = Nr-1, 3, 2 (its MAX(1,k-2), MIN(Nr,k+1)); `c` carries what the Fortran carries
    def level_r(k, c):
        rTransKp, rTrans, fVerT, localTij, localVol, gTracer = c
        kUp = 1+(k+1) % 2                                                       # :941  1+MOD(k+1,2)
        kDown = 1+k % 2                                                         # :942  1+MOD(k,2)
        kp1Msk = 1.                                                             # :943  1.
        if k == Nr:                                                             # :944
            kp1Msk = 0.                                                         #       0.

        if k == 1:                                                              # :961-970  Surface interface
            rTransKp = rTransKp.at[iA, jA].set(kp1Msk*rTrans[iA, jA])
            rTrans = rTrans.at[iA, jA].set(0.)
            fVerT = fVerT.at[iA, jA, kUp].set(0.)
        else:                                                                   # :972-983  Interior interface
            rTransKp = rTransKp.at[iA, jA].set(kp1Msk*rTrans[iA, jA])
            rTrans = rTrans.at[iA, jA].set(wFld[iA, jA, k]*g.rA[iA, jA]
                                           * g.deepFac2F[k]*rhoFacF[k]
                                           * g.maskC[iA, jA, k-1])
            fVerT = fVerT.at[iA, jA, kUp].set(0.)
#-    Compute vertical advective flux in the interior:                   # :985-1019
            if usePPMvertAdv or usePQMvertAdv:                                  # :1009-1015
                fVerT = fVerT.at[iA, jA, kUp].set(afr[iA, jA, k])
            elif vertAdvecScheme == ENUM_DST3_FLUX_LIMIT:                       # :999-1002 (PTRACERS lane)
                from mitjax.pkg.generic_advdiff.gad_dst3fl_adv_r import gad_dst3fl_adv_r
                wT = gad_dst3fl_adv_r(k, deltaTLev[k], rTrans, _level(wFld, k), localT3d, _level(fVerT, kUp),
                                      cfg=kc, grid=g)
                fVerT = fVerT.at[iA, jA, kUp].set(wT[iA, jA])
            elif vertAdvecScheme == ENUM_OS7MP:                                 # :1003-1006 (M3 Task 30)
                from mitjax.pkg.generic_advdiff.gad_os7mp_adv_r import gad_os7mp_adv_r
                wT = gad_os7mp_adv_r(k, deltaTLev[k], rTrans, _level(wFld, k), localT3d, _level(fVerT, kUp),
                                     cfg=kc, grid=g)
                fVerT = fVerT.at[iA, jA, kUp].set(wT[iA, jA])
            elif vertAdvecScheme == ENUM_DST3:                                  # :995-998 (lane M4ADLAB)
                from mitjax.pkg.generic_advdiff.gad_dst3_adv_r import gad_dst3_adv_r
                wT = gad_dst3_adv_r(k, deltaTLev[k], rTrans, _level(wFld, k), localT3d, _level(fVerT, kUp),
                                    cfg=kc, grid=g)
                fVerT = fVerT.at[iA, jA, kUp].set(wT[iA, jA])
            elif vertAdvecScheme in (ENUM_UPWIND_1RST, ENUM_DST2, ENUM_FLUX_LIMIT):   # :987-994
                raise NotImplementedError(f"GAD_ADVECTION: the vertical kernel of scheme {vertAdvecScheme} "
                                          "(GAD_*_ADV_R) is not ported")
            else:
                raise ValueError("GAD_ADVECTION: adv. scheme incompatible with mutli-dim")   # :1018  STOP

#--   Divergence of vertical fluxes                                       # :1038-1080
        if compressible:                                                        # :1038-1062
            tmpTrac = (localT3d[iA, jA, k]*locVol3d[iA, jA, k]
                       - deltaTLev[k]*(fVerT[iA, jA, kDown]-fVerT[iA, jA, kUp])
                       * g.rkSign*g.maskInC[iA, jA])
            localVol = localVol.at[iA, jA].set(
                locVol3d[iA, jA, k]
                - deltaTLev[k]*(rTransKp[iA, jA] - rTrans[iA, jA])
                * g.rkSign*g.maskInC[iA, jA])
            localTij = localTij.at[iA, jA].set(tmpTrac/localVol[iA, jA])        # :1047
            gTracer = gTracer.at[iA, jA, k].set(                                # :1055-1060
                (tmpTrac - tracer[iA, jA, k]*localVol[iA, jA])
                * g.recip_rA[iA, jA]*g.recip_deepFac2C[k]
                * g.recip_drF[k]*g.recip_hFacC[iA, jA, k]
                * recip_rhoFacC[k]
                / deltaTLev[k])
        else:                                                                   # :1063-1077
            localTij = localTij.at[iA, jA].set(
                localT3d[iA, jA, k]
                - deltaTLev[k]*recip_rhoFacC[k]
                * g.recip_hFacC[iA, jA, k]*g.recip_drF[k]
                * g.recip_rA[iA, jA]*g.recip_deepFac2C[k]
                * (fVerT[iA, jA, kDown]-fVerT[iA, jA, kUp]
                   - tracer[iA, jA, k]*(rTransKp[iA, jA]-rTrans[iA, jA])
                   )*g.rkSign*g.maskInC[iA, jA])
            gTracer = gTracer.at[iA, jA, k].set(
                (localTij[iA, jA] - tracer[iA, jA, k])/deltaTLev[k])
        return rTransKp, rTrans, fVerT, localTij, localVol, gTracer

    peel = (3, 2) if vertAdvecScheme == ENUM_DST3_FLUX_LIMIT else (1, 1)
    if vertAdvecScheme == ENUM_DST3:      # lane M4ADLAB: GAD_DST3_ADV_R's km2 = MAX(1,k-2) (:70) is static for k >= 3
        peel = (2, 1)
    if vertAdvecScheme == ENUM_OS7MP:     # M3 Task 30: GAD_OS7MP_ADV_R's clamps MAX(1,k-4) .. MIN(Nr,k+3) (:48-54)
        peel = (5, 4)                     # are static only for 6 <= k <= Nr-4
    c = scan_levels(level_r, (rTransKp, rTrans, fVerT, localTij, localVol, gTracer), 1, Nr, down=True, peel=peel)
    rTransKp, rTrans, fVerT, localTij, localVol, gTracer = c
#--   End of K loop for vertical flux
    return gTracer
