"""INI_VEL of solid-body.cs-32x32x1: verification/solid-body.cs-32x32x1/code/ini_vel.F @63cdc0b (the experiment's own
version; lane B, plan Task 25)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_UV_XYZ_RL
from mitjax.eesupp.global_sum import _zero_plus
from mitjax.farray import loops_kji
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL


def ini_vel(state, *, cfg, grid, params, ex, rw):
    """INI_VEL( myThid )   @63cdc0b verification/solid-body.cs-32x32x1/code/ini_vel.F:6-92

    A solid-body rotation (5-day period) from the stream function psi = fac*fCoriG (the statement function of :38):
    uVel = 0. + (psi(i,j) - psi(i,jp1))*recip_dyG, vVel = 0. + (psi(ip1,j) - psi(i,j))*recip_dxG with
    jp1 = MIN(j+1,sNy+OLy), ip1 = MIN(i+1,sNx+OLx) on every point (:48-65); the optional input files (:69-73), the
    unconditional EXCH_UV_XYZ_RL( uVel, vVel, .TRUE. ) (:75; its IF is commented out) and the masks (:78-89).
    `0. _d 0 + x` is the IEEE sum (+0 for x = -0; mitjax/eesupp/global_sum._zero_plus: XLA folds a constant 0 away).
    MIN of the indices is static (the last row / column uses itself)."""
    sz = cfg.size
    gp, ip = params.grid, params.init
    rSphere, Omega = gp.rSphere, gp.omega
    omegaprime = 80.0 / rSphere                                                # :48  80. _d 0 / rSphere
    fac = -(rSphere*rSphere)*omegaprime/(2.0*Omega)                            # :49
    psi = fac*jnp.asarray(grid.fCoriG.data)                                    # :38  psi(i,j) = fac*fCoriG(i,j)
    psi_jp1 = jnp.concatenate([psi[:, 1:, :], psi[:, -1:, :]], axis=1)         # psi(i,jp1), jp1 = MIN(j+1,sNy+OLy)
    psi_ip1 = jnp.concatenate([psi[:, :, 1:], psi[:, :, -1:]], axis=2)         # psi(ip1,j), ip1 = MIN(i+1,sNx+OLx)
    u2 = _zero_plus((psi - psi_jp1)*jnp.asarray(grid.recip_dyG.data))          # :57-58
    v2 = _zero_plus((psi_ip1 - psi)*jnp.asarray(grid.recip_dxG.data))          # :59-60
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    uVel = state.uVel.at[i, j, k].set(jnp.broadcast_to(u2[:, None], state.uVel.data.shape))
    vVel = state.vVel.at[i, j, k].set(jnp.broadcast_to(v2[:, None], state.vVel.data.shape))
    if str(ip.uVelInitFile).strip() != "":                                      # :69-70
        uVel = READ_FLD_XYZ_RL(ip.uVelInitFile, " ", uVel, 0, rw=rw)
    if str(ip.vVelInitFile).strip() != "":                                      # :72-73
        vVel = READ_FLD_XYZ_RL(ip.vVelInitFile, " ", vVel, 0, rw=rw)
    uVel, vVel = EXCH_UV_XYZ_RL(uVel, vVel, True, ex=ex)                       # :75
    uVel = uVel.at[i, j, k].set(uVel[i, j, k]*grid.maskW[i, j, k])             # :83
    vVel = vVel.at[i, j, k].set(vVel[i, j, k]*grid.maskS[i, j, k])             # :84
    return state.replace(uVel=uVel, vVel=vVel)
