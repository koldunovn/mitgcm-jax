"""INI_MASKS_ETC: model/src/ini_masks_etc.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.eesupp.exch_rs import EXCH_UV_XY_RS, EXCH_UV_XYZ_RS
from mitjax.model.grid import declare, local, oneRL, oneRS, zeroRL, zeroRS, halfRL
from mitjax.model.src.add_walls2masks import add_walls2masks
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.safe import safe_div


def ini_masks_etc(grid, *, cfg, params, ex):
    """INI_MASKS_ETC( myThid )   @63cdc0b model/src/ini_masks_etc.F:7-508

    C     | SUBROUTINE INI_MASKS_ETC
    C     | o Initialise masks and topography factors
    C     | These arrays are used throughout the code and describe
    C     | the topography of the domain through masks (0s and 1s)
    C     | and fractional height factors (0<hFac<1). The latter
    C     | distinguish between the lopped-cell and full-step
    C     | topographic representations.

    Reads R_low, Ro_surf (INI_DEPTHS), rF, drF, recip_drF (INI_VERTICAL_GRID), dxG, dyG (INI_GRID) from `grid`;
    returns `grid` with hFacC/W/S, recip_hFacC/W/S, maskC/W/S, maskInC/W/S, R_low and Ro_surf (re-computed),
    rLowW/S, rSurfW/S, recip_Rcol, kSurfC/W/S, kLowC, and h0FacC/W/S under NONLIN_FRSURF. `ex`: the experiment's
    exchanger (mitjax/eesupp); EXCH_UV_XYZ_RS / EXCH_UV_XY_RS (mitjax/eesupp/exch_rs.py) are called at :402-404.

    Vectorisation: the hFac loops (:107-124, :147-165, :271-323) and the mask/reciprocal loop (:457-483) compute
    each (k,j,i) point from inputs only, so their k iterations are independent and run as one k-vectorised nest
    (`loops_kji`); `hFacMnSz` depends on k only. The column sums `tmpVar = tmpVar + drF(k)*hFac(k)` (:132-138,
    :177-184, :361-367, :381-387) and the k-index searches (kLowC :181, kSurfC :185-191, kSurfW/S :413-416) are
    recursions in k and run as Python loops over k in the Fortran order (each level vectorised over i,j). The tile
    loop is implicit. `MAX`/`MIN` are mitjax.ops.fortran_minmax.MAX/MIN with the per-site winner of ties and NaN (p=).
    Pointwise IFs are `where`s; `1. _d 0 / x` under `IF (x .NE. 0)` or `IF (x .LE. 0) ... ELSE` is guarded before
    the division (mitjax/ops/safe.py), so every lane is finite.

    Ported for the M1 variants (docs/coverage/*.md): `selectSigmaCoord` = 0 (:65-431; INI_SIGMA_HFAC :433-436
    raises), `useMin4hFacEdges` = .FALSE. (Method-2, :266-324; Method-1 :243-265 raises), ADD_WALLS2MASKS. Not
    compiled in any M1 build: ALLOW_SHELFICE (:53-60), ALLOW_STEEP_ICECAVITY (:61-63) (raise if defined and used).
    PLOT_FIELD_XYRS / PLOT_FIELD_XYZRS (:206-213, :448-452, plotLevel >= debLevB, STDOUT only) are not ported: they
    write no model variable.
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    rF, drF, recip_drF = grid.rF, grid.drF, grid.recip_drF
    R_low, Ro_surf = grid.R_low, grid.Ro_surf
    hFacMin, hFacMinDr = params.hFacMin, params.hFacMinDr

    if cfg.cpp.ALLOW_SHELFICE and cfg.use_flag("useShelfIce"):     # :53-60
        raise NotImplementedError("INI_MASKS_ETC: SHELFICE_INIT_DEPTHS is not ported")
    if cfg.cpp.ALLOW_STEEP_ICECAVITY:                               # :61-63
        raise NotImplementedError("INI_MASKS_ETC: STIC_INIT_DEPTHS is not ported")
    if params.selectSigmaCoord != 0:                                # :432-437
        raise NotImplementedError("INI_MASKS_ETC: INI_SIGMA_HFAC (selectSigmaCoord /= 0) is not ported")

    hFacC = declare("hFacC", sz)
    hFacW = declare("hFacW", sz)
    hFacS = declare("hFacS", sz)
    rLowW = declare("rLowW", sz)
    rLowS = declare("rLowS", sz)
    rSurfW = declare("rSurfW", sz)
    rSurfS = declare("rSurfS", sz)
    kSurfC = declare("kSurfC", sz)
    kSurfW = declare("kSurfW", sz)
    kSurfS = declare("kSurfS", sz)
    kLowC = declare("kLowC", sz)
    maskInC = declare("maskInC", sz)
    maskInW = declare("maskInW", sz)
    maskInS = declare("maskInS", sz)
    recip_Rcol = declare("recip_Rcol", sz)
    tmpVar = local("tmpVar", sz, i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))   # :45  _RL tmpVar(...)

    # :68-100  rLow & reference rSurf at Western & Southern edges (not final: hFacMin ignored)
    rEmpty = rF[1]                                                  # :70
    i = 1-OLx                                                       # :73-77
    j = loop_j(1-OLy, sNy+OLy)
    rLowW = rLowW.at[i, j].set(rEmpty)
    rSurfW = rSurfW.at[i, j].set(rEmpty)
    j = 1-OLy                                                       # :78-82
    i = loop_i(1-OLx, sNx+OLx)
    rLowS = rLowS.at[i, j].set(rEmpty)
    rSurfS = rSurfS.at[i, j].set(rEmpty)
    j = loop_j(1-OLy, sNy+OLy)                                      # :83-90
    i = loop_i(2-OLx, sNx+OLx)
    rLowW = rLowW.at[i, j].set(MAX(R_low[i-1, j], R_low[i, j], p="b"))           # :85-86
    rSurfW = rSurfW.at[i, j].set(MIN(Ro_surf[i-1, j], Ro_surf[i, j], p="b"))     # :87-88
    j = loop_j(2-OLy, sNy+OLy)                                      # :91-98
    i = loop_i(1-OLx, sNx+OLx)
    rLowS = rLowS.at[i, j].set(MAX(R_low[i, j-1], R_low[i, j], p="b"))           # :93-94
    rSurfS = rSurfS.at[i, j].set(MIN(Ro_surf[i, j-1], Ro_surf[i, j], p="b"))     # :95-96

    # :105-124  lopping factor hFacC: over-estimate the part inside of the domain (lower_R boundary)
    k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
    hFacMnSz = MAX(hFacMin, MIN(hFacMinDr*recip_drF[k], oneRL, p="b"), p="b")     # :108
    hFac_loc = (rF[k]-R_low[i, j])*recip_drF[k]
    hFac_loc = MIN(MAX(hFac_loc, zeroRL, p="b"), oneRL, p="a")                  # :114
    hFacC = hFacC.at[i, j, k].set(jnp.where((hFac_loc < hFacMnSz*halfRL) | (R_low[i, j] >= Ro_surf[i, j]),
                                            zeroRS, MAX(hFac_loc, hFacMnSz, p="b")))  # :120

    # :126-143  re-calculate the lower-R boundary position, taking into account hFacC
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    tmpVar = tmpVar.at[i, j].set(0.)                                # 0. _d 0
    for kk in range(1, Nr + 1):
        tmpVar = tmpVar.at[i, j].set(tmpVar[i, j] + drF[kk]*hFacC[i, j, kk])
    R_low = R_low.at[i, j].set(rF[1] - tmpVar[i, j])

    # :145-165  lopping factor hFacC: remove the part outside of the domain (reference surface Ro_surf)
    k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
    hFacMnSz = MAX(hFacMin, MIN(hFacMinDr*recip_drF[k], oneRL, p="b"), p="b")     # :148
    hFac_loc = (rF[k]-Ro_surf[i, j])*recip_drF[k]
    hFac_loc = hFacC[i, j, k] - MAX(hFac_loc, zeroRL, p="b")                    # :154
    hFac_loc = MAX(hFac_loc, zeroRL, p="a")                                     # :156
    hFacC = hFacC.at[i, j, k].set(jnp.where(hFac_loc < hFacMnSz*halfRL, zeroRS,
                                            MAX(hFac_loc, hFacMnSz, p="b")))    # :161

    # :167-200  reference surface position, column thickness, kSurfC, kLowC, maskInC
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    tmpVar = tmpVar.at[i, j].set(0.)
    kSurfC = kSurfC.at[i, j].set(Nr+1)
    kLowC = kLowC.at[i, j].set(0)
    for kk in range(1, Nr + 1):                                     # :177-184
        tmpVar = tmpVar.at[i, j].set(tmpVar[i, j] + drF[kk]*hFacC[i, j, kk])
        kLowC = kLowC.at[i, j].set(jnp.where(hFacC[i, j, kk] != zeroRS, kk, kLowC[i, j]))
    for kk in range(Nr, 0, -1):                                     # :185-191
        kSurfC = kSurfC.at[i, j].set(jnp.where(hFacC[i, j, kk] != zeroRS, kk, kSurfC[i, j]))
    Ro_surf = Ro_surf.at[i, j].set(R_low[i, j] + tmpVar[i, j])      # :192-200
    maskInC = maskInC.at[i, j].set(0.)
    maskInC = maskInC.at[i, j].set(jnp.where(kSurfC[i, j] <= Nr, 1., maskInC[i, j]))

    # :217-233  quantities derived from the XY depth map
    tmpVar = tmpVar.at[i, j].set(Ro_surf[i, j] - R_low[i, j])
    recip_Rcol = recip_Rcol.at[i, j].set(
        safe_div(1.0, tmpVar[i, j], ~(tmpVar[i, j] <= zeroRL), fill=zeroRS))   # 1. _d 0 / tmpVar(i,j)

    # :235-324  hFacW, hFacS
    if params.useMin4hFacEdges:                                     # :243-265  Method-1
        raise NotImplementedError("INI_MASKS_ETC: useMin4hFacEdges = .TRUE. (Method-1) is not ported")
    else:                                                           # :266-324  Method-2
        k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
        hFacMnSz = MAX(hFacMin, MIN(hFacMinDr*recip_drF[k], oneRL, p="b"), p="b")     # :272
        hFac1tmp = (rF[k] - rLowW[i, j])*recip_drF[k]               # :276-295  W
        hFac_loc = MIN(hFac1tmp, oneRL, p="a")                      # :277
        hFac1tmp = jnp.where((hFac_loc < hFacMnSz*halfRL) | (rLowW[i, j] >= rSurfW[i, j]),
                             0., MAX(hFac_loc, hFacMnSz, p="b"))    # :284
        hFac2tmp = (rF[k] - rSurfW[i, j])*recip_drF[k]
        hFac_loc = hFac1tmp - MAX(hFac2tmp, zeroRL, p="b")          # :289
        hFacW = hFacW.at[i, j, k].set(jnp.where(hFac_loc < hFacMnSz*halfRL, zeroRS,
                                                MAX(hFac_loc, hFacMnSz, p="b")))    # :294
        hFac1tmp = (rF[k] - rLowS[i, j])*recip_drF[k]               # :298-322  S
        hFac_loc = MIN(hFac1tmp, oneRL, p="a")                      # :302
        hFac1tmp = jnp.where((hFac_loc < hFacMnSz*halfRL) | (rLowS[i, j] >= rSurfS[i, j]),
                             0., MAX(hFac_loc, hFacMnSz, p="b"))    # :309
        hFac2tmp = (rF[k] - rSurfS[i, j])*recip_drF[k]
        hFac_loc = hFac1tmp - MAX(hFac2tmp, zeroRL, p="b")          # :314
        hFacS = hFacS.at[i, j, k].set(jnp.where(hFac_loc < hFacMnSz*halfRL, zeroRS,
                                                MAX(hFac_loc, hFacMnSz, p="b")))    # :319

    # :326-349  update rLow & reference rSurf at Western & Southern edges (hFacMin-adjusted R_low, Ro_surf)
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(2-OLx, sNx+OLx)
    rLowW = rLowW.at[i, j].set(MAX(R_low[i-1, j], R_low[i, j], p="b"))           # :332-333
    rSurfW = rSurfW.at[i, j].set(MIN(Ro_surf[i-1, j], Ro_surf[i, j], p="b"))     # :334-335
    rSurfW = rSurfW.at[i, j].set(MAX(rSurfW[i, j], rLowW[i, j], p="b"))         # :336-337
    j = loop_j(2-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    rLowS = rLowS.at[i, j].set(MAX(R_low[i, j-1], R_low[i, j], p="b"))           # :342-343
    rSurfS = rSurfS.at[i, j].set(MIN(Ro_surf[i, j-1], Ro_surf[i, j], p="b"))     # :344-345
    rSurfS = rSurfS.at[i, j].set(MAX(rSurfS[i, j], rLowS[i, j], p="b"))         # :346-347

    # :351-396  adjust the reference rSurf at W & S edges to the integrated level thickness (the `IF (useShelfIce)`
    # around it is commented out in the Fortran, :351 and :396: it always runs)
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    tmpVar = tmpVar.at[i, j].set(0.)                                # :356-360
    for kk in range(1, Nr + 1):                                     # :361-367
        tmpVar = tmpVar.at[i, j].set(tmpVar[i, j] + drF[kk]*hFacW[i, j, kk])
    rSurfW = rSurfW.at[i, j].set(rLowW[i, j] + tmpVar[i, j])        # :369-373
    tmpVar = tmpVar.at[i, j].set(0.)                                # :376-380
    for kk in range(1, Nr + 1):                                     # :381-387
        tmpVar = tmpVar.at[i, j].set(tmpVar[i, j] + drF[kk]*hFacS[i, j, kk])
    rSurfS = rSurfS.at[i, j].set(rLowS[i, j] + tmpVar[i, j])        # :389-393

    hFacW, hFacS = EXCH_UV_XYZ_RS(hFacW, hFacS, False, ex=ex)      # :402
    rSurfW, rSurfS = EXCH_UV_XY_RS(rSurfW, rSurfS, False, ex=ex)   # :403
    rLowW, rLowS = EXCH_UV_XY_RS(rLowW, rLowS, False, ex=ex)       # :404

    # :406-424  surface k index for interfaces W & S, maskInW/S
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    kSurfW = kSurfW.at[i, j].set(Nr+1)
    kSurfS = kSurfS.at[i, j].set(Nr+1)
    for kk in range(Nr, 0, -1):
        kSurfW = kSurfW.at[i, j].set(jnp.where(hFacW[i, j, kk] != zeroRS, kk, kSurfW[i, j]))
        kSurfS = kSurfS.at[i, j].set(jnp.where(hFacS[i, j, kk] != zeroRS, kk, kSurfS[i, j]))
    maskInW = maskInW.at[i, j].set(zeroRS)
    maskInW = maskInW.at[i, j].set(jnp.where(kSurfW[i, j] <= Nr, oneRS, maskInW[i, j]))
    maskInS = maskInS.at[i, j].set(zeroRS)
    maskInS = maskInS.at[i, j].set(jnp.where(kSurfS[i, j] <= Nr, oneRS, maskInS[i, j]))

    # :426-430  additional closing of Western and Southern grid-cell edges
    (hFacW, hFacS, rLowW, rLowS, rSurfW, rSurfS, kSurfW, kSurfS, maskInW, maskInS) = add_walls2masks(
        rEmpty, hFacW, hFacS, rLowW, rLowS, rSurfW, rSurfS, kSurfW, kSurfS, maskInW, maskInS,
        cfg=cfg, grid=grid, params=params)

    # :454-500  masks and reciprocals of hFac[CWS]
    recip_hFacC = declare("recip_hFacC", sz)
    recip_hFacW = declare("recip_hFacW", sz)
    recip_hFacS = declare("recip_hFacS", sz)
    maskC = declare("maskC", sz)
    maskW = declare("maskW", sz)
    maskS = declare("maskS", sz)
    k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
    wet = hFacC[i, j, k] != zeroRS                                  # :460-466
    recip_hFacC = recip_hFacC.at[i, j, k].set(safe_div(1.0, hFacC[i, j, k], wet, fill=zeroRS))
    maskC = maskC.at[i, j, k].set(jnp.where(wet, oneRS, zeroRS))
    wet = hFacW[i, j, k] != zeroRS                                  # :467-473
    recip_hFacW = recip_hFacW.at[i, j, k].set(safe_div(1.0, hFacW[i, j, k], wet, fill=zeroRS))
    maskW = maskW.at[i, j, k].set(jnp.where(wet, oneRS, zeroRS))
    wet = hFacS[i, j, k] != zeroRS                                  # :474-480
    recip_hFacS = recip_hFacS.at[i, j, k].set(safe_div(1.0, hFacS[i, j, k], wet, fill=zeroRS))
    maskS = maskS.at[i, j, k].set(jnp.where(wet, oneRS, zeroRS))
    out = dict(hFacC=hFacC, hFacW=hFacW, hFacS=hFacS, recip_hFacC=recip_hFacC, recip_hFacW=recip_hFacW,
               recip_hFacS=recip_hFacS, maskC=maskC, maskW=maskW, maskS=maskS, maskInC=maskInC, maskInW=maskInW,
               maskInS=maskInS, R_low=R_low, Ro_surf=Ro_surf, rLowW=rLowW, rLowS=rLowS, rSurfW=rSurfW,
               rSurfS=rSurfS, recip_Rcol=recip_Rcol, kSurfC=kSurfC, kSurfW=kSurfW, kSurfS=kSurfS, kLowC=kLowC)
    if cfg.cpp.NONLIN_FRSURF:                                       # :484-497  h0Fac = initial hFac (fixed in time)
        out["h0FacC"] = declare("h0FacC", sz).at[i, j, k].set(hFacC[i, j, k])
        out["h0FacW"] = declare("h0FacW", sz).at[i, j, k].set(hFacW[i, j, k])
        out["h0FacS"] = declare("h0FacS", sz).at[i, j, k].set(hFacS[i, j, k])
    return grid.replace(**out)
