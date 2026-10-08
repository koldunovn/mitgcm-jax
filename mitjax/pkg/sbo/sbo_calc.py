"""SBO_CALC: pkg/sbo/sbo_calc.F @63cdc0b, and the SBO.h common block it fills."""

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp

from mitjax.eesupp import global_sum as gs
from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.model.grid import deg2rad
from mitjax.model.src.rotate_uv2en import rotate_uv2en_rl

ae = 6.3710e6               # sbo_calc.F:106  PARAMETER ( ae        = 6.3710 _d 6    )
sbo_omega = 7.292115e-5     # sbo_calc.F:107  PARAMETER ( sbo_omega = 7.292115 _d -5 )
MINUS_ZERO = -0.0           # the additive identity of IEEE round-to-nearest: a skipped (masked) term


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SboCommon:
    """SBO.h (pkg/sbo/SBO.h:20-40) common block scalars, Fortran names."""
    xoamc: jax.Array
    yoamc: jax.Array
    zoamc: jax.Array
    xoamp: jax.Array
    yoamp: jax.Array
    zoamp: jax.Array
    mass: jax.Array
    xcom: jax.Array
    ycom: jax.Array
    zcom: jax.Array
    sboarea: jax.Array
    xoamp_fw: jax.Array
    yoamp_fw: jax.Array
    zoamp_fw: jax.Array
    mass_fw: jax.Array
    xcom_fw: jax.Array
    ycom_fw: jax.Array
    zcom_fw: jax.Array
    xoamc_si: jax.Array
    yoamc_si: jax.Array
    zoamc_si: jax.Array
    mass_si: jax.Array
    mass_gc: jax.Array

    @staticmethod
    def names():
        return tuple(f.name for f in fields(SboCommon))


def _chain(terms):
    """Per-tile partial sums of a `tile(bi,bj) = tile(bi,bj) + term` chain: terms [tile, nj, ...] in the Fortran
    loop order (j outer, then the remaining axes in order, innermost last), starting from `tile = 0.0`
    (eesupp/global_sum.tile_sum_fortran)."""
    t = jnp.asarray(terms)
    return gs.tile_sum_fortran(t.reshape(t.shape[0], t.shape[1], -1))


def _gsum(part, ex):
    """GLOBAL_SUM_TILE_RL (eesupp/src/global_sum_tile.F; fixed tile order)."""
    return gs.global_sum_tile(part) if ex is None else ex.global_sum_tile(part)


def sbo_calc(myTime, myIter, *, cfg, grid, fp, state, ff, ex=None):
    """SBO_CALC( myTime, myIter, myThid )   @63cdc0b pkg/sbo/sbo_calc.F:10-475

    C     | SUBROUTINE SBO_CALC                                      |
    C     | o Do SBO diagnostic output.                              |
    C     |    calc_sbo calculates the core products of the IERS Special Bureau
    C     |    for the Oceans including oceanic mass, center-of-mass, and angular
    C     |    momentum.

    Reads GRID.h (rA, maskC, drF, hFacC as the step left it, R_low, xC, yC, angleCosC/SinC), the State (etaN, uVel,
    vVel, rhoInSitu), FFIELDS.h sIceLoad, PARAMS.h rhoConst (`fp`). Returns the SBO.h scalars (SboCommon).

    Sums: every `tile_x(bi,bj) = tile_x(bi,bj) + term` (or `- term`, the addition of the exactly negated term) is
    one sequential chain in the Fortran loop order j, i (k innermost where the k loop sits inside), from 0.; a point
    the `IF ( maskC(i,j,1,bi,bj) .NE. 0. )` of :317 skips contributes -0.0, the exact additive identity; tile partials
    are added by GLOBAL_SUM_TILE_RL in tile order (mitjax/eesupp/global_sum.py). The :241-243 and :424-447 sums are
    independent, so their order does not matter. ALLOW_SEAICE (:187-196) raises when useSEAICE (no M1 build
    compiles pkg/seaice); UEice = VNice = 0. (:197-207) otherwise."""
    sz = cfg.size
    if cfg.cpp.ALLOW_SEAICE:
        raise NotImplementedError("SBO_CALC: ALLOW_SEAICE (sea-ice OAM) is not ported")
    rhoConst = fp.rhoConst

    # :182-185  CALL ROTATE_UV2EN_RL( uVel, vVel, UE, VN, .TRUE., .TRUE., .FALSE., Nr, mythid )
    UE = state.uVel.local("UE")
    VN = state.vVel.local("VN")
    _, _, UE, VN = rotate_uv2en_rl(state.uVel, state.vVel, UE, VN, True, True, False, sz.Nr,
                                   cfg=cfg, grid=grid, fp=fp)

    # interior loops DO j = 1, sNy; DO i = 1, sNx (:225-226, :314-315)
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    kk, jk, ik = loops_kji((1, sz.Nr), (1, sz.sNy), (1, sz.sNx))
    to_jik = lambda a: jnp.moveaxis(a, 1, -1)                                   # noqa: E731  [t,k,j,i] -> [t,j,i,k]

    rA = grid.rA[i, j]
    maskC1 = grid.maskC[i, j, 1]
    etaN = state.etaN[i, j]
    sIceLoad = ff.sIceLoad[i, j]
    # :220-240
    darea = rA*maskC1                                                           # :227
    tile_sboarea = _chain(darea)                                                # :228
    fw_a = rhoConst*etaN*darea                                                  # :229-231 tile + a + b:
    fw_b = sIceLoad*darea                                                       #   two additions per point
    tile_FWload = _chain(jnp.stack([fw_a, fw_b], axis=-1))
    dvolume = grid.rA[ik, jk]*grid.drF[kk]*grid.hFacC[ik, jk, kk]               # :233
    tile_GCload = _chain(to_jik(state.rhoInSitu[ik, jk, kk]*dvolume))           # :234-235
    FWload = _gsum(tile_FWload, ex)                                             # :241
    sboarea = _gsum(tile_sboarea, ex)                                           # :242
    GCload = _gsum(tile_GCload, ex)                                             # :243
    FWload = FWload/sboarea                                                     # :244
    GCload = -1.0 * GCload/sboarea                                              # :245  (-1.0 REAL*4, exact)

    # :248-262 Mload on every point (halos included)
    jh = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    ih = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    Mload = ff.sIceLoad.local("Mload")
    Mload = Mload.at[ih, jh].set(rhoConst*state.etaN[ih, jh]
                                 + ff.sIceLoad[ih, jh]
                                 + GCload - grid.R_low[ih, jh]*rhoConst)       # :252-254
    for k in range(1, sz.Nr + 1):                                               # :255-258 (recursion in k)
        Mload = Mload.at[ih, jh].set(Mload[ih, jh]
                                     + state.rhoInSitu[ih, jh, k]*grid.drF[k]*grid.hFacC[ih, jh, k])

    # :265-278 cos/sin of lat, lon on every point
    lat = grid.yC[i, j] * deg2rad                                               # :269 (interior used below)
    lon = grid.xC[i, j] * deg2rad                                               # :270
    COSlat, SINlat, COSlon, SINlon = jnp.cos(lat), jnp.sin(lat), jnp.cos(lon), jnp.sin(lon)   # :271-274

    # :284-421 main loops; `wet` is the IF of :317
    wet = maskC1 != 0.
    skip = lambda term: jnp.where(wet, term, MINUS_ZERO)                        # noqa: E731
    darea = rA*maskC1                                                           # :320
    M = Mload[i, j]
    tile_mass = _chain(skip(M*darea))                                           # :323-324
    tile_mass_gc = _chain(skip(GCload*darea))                                   # :325-326
    tile_mass_si = _chain(skip(sIceLoad*darea))                                 # :327-328
    tile_xcom = _chain(skip(M*COSlat*COSlon * ae * darea))                      # :331-333
    tile_ycom = _chain(skip(M*COSlat*SINlon * ae * darea))                      # :334-336
    tile_zcom = _chain(skip(M*SINlat * ae * darea))                             # :337-339
    # :343-359 the k loop inside the wet IF: one chain over (j, i, k) per accumulator
    dvol = grid.rA[ik, jk]*grid.drF[kk] * grid.maskC[ik, jk, kk]*grid.hFacC[ik, jk, kk]   # :344-345
    UEk, VNk = UE[ik, jk, kk], VN[ik, jk, kk]
    SINlat3, COSlat3 = SINlat[:, None], COSlat[:, None]
    SINlon3, COSlon3 = SINlon[:, None], COSlon[:, None]
    wet3 = jnp.broadcast_to(wet[:, None], dvol.shape)
    skip3 = lambda term: to_jik(jnp.where(wet3, term, MINUS_ZERO))             # noqa: E731
    tile_xoamc = _chain(skip3((VNk*SINlon3 - UEk*SINlat3*COSlon3) * rhoConst * ae * dvol))   # :346-350
    tile_yoamc = _chain(skip3((-VNk*COSlon3 - UEk*SINlat3*SINlon3) * rhoConst * ae * dvol))  # :351-355
    tile_zoamc = _chain(skip3(UEk*COSlat3 * rhoConst * ae * dvol))                           # :356-358
    # :362-374 sea-ice motion (UEice = VNice = 0. on every point, :197-207)
    UEice = jnp.zeros_like(rA)
    VNice = jnp.zeros_like(rA)
    tile_xoamc_si = _chain(skip((VNice*SINlon - UEice*SINlat*COSlon) * sIceLoad * ae * darea))
    tile_yoamc_si = _chain(skip((-VNice*COSlon - UEice*SINlat*SINlon) * sIceLoad * ae * darea))
    tile_zoamc_si = _chain(skip(UEice*COSlat * sIceLoad * ae * darea))
    # :377-385 pressure OAM (`tile - x` is `tile + (-x)` exactly)
    tile_xoamp = _chain(skip(-(SINlat*COSlat*COSlon * sbo_omega * M * ae*ae * darea)))       # :377-379
    tile_yoamp = _chain(skip(-(SINlat*COSlat*SINlon * sbo_omega * M * ae*ae * darea)))       # :380-382
    tile_zoamp = _chain(skip(COSlat * COSlat * sbo_omega * M * ae*ae * darea))               # :383-385
    # :388-411 real fresh-water flux
    tile_mass_fw = _chain(skip(FWload * darea))                                              # :388-389
    tile_xcom_fw = _chain(skip(FWload * COSlat * COSlon * ae * darea))                       # :392-394
    tile_ycom_fw = _chain(skip(FWload * COSlat * SINlon * ae * darea))                       # :395-397
    tile_zcom_fw = _chain(skip(FWload * SINlat * ae * darea))                                # :398-400
    tile_xoamp_fw = _chain(skip(-(SINlat*COSlat*COSlon * sbo_omega * FWload * ae*ae * darea)))   # :403-405
    tile_yoamp_fw = _chain(skip(-(SINlat*COSlat*SINlon * sbo_omega * FWload * ae*ae * darea)))   # :406-408
    tile_zoamp_fw = _chain(skip(COSlat * COSlat * sbo_omega * FWload * ae*ae * darea))           # :409-411

    # :424-447
    mass, xcom, ycom, zcom = (_gsum(t, ex) for t in (tile_mass, tile_xcom, tile_ycom, tile_zcom))
    xoamc, yoamc, zoamc = (_gsum(t, ex) for t in (tile_xoamc, tile_yoamc, tile_zoamc))
    xoamp, yoamp, zoamp = (_gsum(t, ex) for t in (tile_xoamp, tile_yoamp, tile_zoamp))
    xoamc_si, yoamc_si, zoamc_si, mass_si = (_gsum(t, ex) for t in (tile_xoamc_si, tile_yoamc_si, tile_zoamc_si,
                                                                     tile_mass_si))
    mass_fw, xcom_fw, ycom_fw, zcom_fw = (_gsum(t, ex) for t in (tile_mass_fw, tile_xcom_fw, tile_ycom_fw,
                                                                 tile_zcom_fw))
    xoamp_fw, yoamp_fw, zoamp_fw = (_gsum(t, ex) for t in (tile_xoamp_fw, tile_yoamp_fw, tile_zoamp_fw))
    mass_gc = _gsum(tile_mass_gc, ex)

    # :453-463 (pointwise IF on a traced scalar: select, both branches finite where mass != 0; the guarded
    # denominator keeps the unused branch finite)
    nz = mass != 0.
    m_safe = jnp.where(nz, mass, 1.)
    xcom, ycom, zcom = (jnp.where(nz, c / m_safe, c) for c in (xcom, ycom, zcom))
    nzf = mass_fw != 0.
    mf_safe = jnp.where(nzf, mass_fw, 1.)
    xcom_fw, ycom_fw, zcom_fw = (jnp.where(nzf, c / mf_safe, c) for c in (xcom_fw, ycom_fw, zcom_fw))
    # :466-468
    xoamc = xoamc + xoamc_si
    yoamc = yoamc + yoamc_si
    zoamc = zoamc + zoamc_si
    return SboCommon(xoamc=xoamc, yoamc=yoamc, zoamc=zoamc, xoamp=xoamp, yoamp=yoamp, zoamp=zoamp, mass=mass,
                     xcom=xcom, ycom=ycom, zcom=zcom, sboarea=sboarea, xoamp_fw=xoamp_fw, yoamp_fw=yoamp_fw,
                     zoamp_fw=zoamp_fw, mass_fw=mass_fw, xcom_fw=xcom_fw, ycom_fw=ycom_fw, zcom_fw=zcom_fw,
                     xoamc_si=xoamc_si, yoamc_si=yoamc_si, zoamc_si=zoamc_si, mass_si=mass_si, mass_gc=mass_gc)
