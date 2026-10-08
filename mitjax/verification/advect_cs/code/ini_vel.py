"""INI_VEL of advect_cs: verification/advect_cs/code/ini_vel.F @63cdc0b (the experiment's own version, which the build
compiles instead of model/src/ini_vel.F)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_UV_XYZ_RL
from mitjax.farray import loops_kji
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL


def ini_vel(state, *, cfg, grid, params, ex, rw):
    """INI_VEL( myThid )   @63cdc0b verification/advect_cs/code/ini_vel.F:6-92

    C     | SUBROUTINE INI_VEL
    C     | o Initialize flow field (either to zero or from input files)

    A solid-body rotation from the stream function psi(i,j,bi,bj) = fac*fCoriG(i,j,bi,bj) (statement function, :38,
    expanded at each use as gfortran does): :48 omegaprime = 38.60328935834681 _d 0 / rSphere (a 12-day rotation),
    :49 fac = -(rSphere*rSphere)*omegaprime/(2. _d 0*Omega); :50-65 on every point uVel = 0. _d 0 +
    (psi(i,j)-psi(i,jp1))*recip_dyG, vVel = 0. _d 0 + (psi(ip1,j)-psi(i,j))*recip_dxG with jp1 = MIN(j+1,sNy+OLy),
    ip1 = MIN(i+1,sNx+OLx) (at the last row / column the difference is psi - psi); :69-73 the input files; :75
    EXCH_UV_XYZ_RL(.TRUE.) (outside the commented-out IF, so always); :78-89 the masks. `0. _d 0 +` is kept (it
    turns -0 into +0). Pointwise; the integer MIN of the index is the shifted read (MINMAX-INT: integer index)."""
    sz = cfg.size
    ip, gp = params.init, params.grid
    g = grid
    rSphere, Omega = gp.rSphere, gp.omega
    omegaprime = 38.60328935834681 / rSphere                                    # :48
    fac = -(rSphere*rSphere)*omegaprime/(2.*Omega)                              # :49
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    fG = g.fCoriG.data[:, None]                                                 # [tile, 1, j, i]
    # jp1 = MIN(j+1,sNy+OLy), ip1 = MIN(i+1,sNx+OLx) (:54, :56; MINMAX-INT: integer index): the next row / column,
    # the last row / column itself at the upper bound
    fG_jp1 = jnp.concatenate([fG[:, :, 1:, :], fG[:, :, -1:, :]], axis=2)
    fG_ip1 = jnp.concatenate([fG[:, :, :, 1:], fG[:, :, :, -1:]], axis=3)
    uVel = state.uVel.at[i, j, k].set(0.                                        # :57-58
                                      + (fac*fG - fac*fG_jp1)*g.recip_dyG.data[:, None])
    vVel = state.vVel.at[i, j, k].set(0.                                        # :59-60
                                      + (fac*fG_ip1 - fac*fG)*g.recip_dxG.data[:, None])
    if str(ip.uVelInitFile).strip() != "":                                      # :69-70
        uVel = READ_FLD_XYZ_RL(ip.uVelInitFile, " ", uVel, 0, rw=rw)
    if str(ip.vVelInitFile).strip() != "":                                      # :72-73
        vVel = READ_FLD_XYZ_RL(ip.vVelInitFile, " ", vVel, 0, rw=rw)
    uVel, vVel = EXCH_UV_XYZ_RL(uVel, vVel, True, ex=ex)                        # :75
    uVel = uVel.at[i, j, k].set(uVel[i, j, k]*g.maskW[i, j, k])                 # :83
    vVel = vVel.at[i, j, k].set(vVel[i, j, k]*g.maskS[i, j, k])                 # :84
    return state.replace(uVel=uVel, vVel=vVel)
