"""DWNSLP_CALC_RHO: pkg/down_slope/dwnslp_calc_rho.F @63cdc0b (lane M4ADLAB session 3)."""

import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.model.src.find_rho import find_rho_2d


def dwnslp_calc_rho(tFld, sFld, rhoLoc, k, *, cfg, grid, params, eos, state):
    """DWNSLP_CALC_RHO( tFld, sFld, rhoLoc, k, bi, bj, myTime, myIter, myThid )
    @63cdc0b pkg/down_slope/dwnslp_calc_rho.F:6-83

    C     | o Compute the in-situ density at level k; below the bottom (and above the surface) of a water column
    C     |   from the bottom (surface) values of that column

    `tFld`, `sFld` 3-D FArrays (theta, salt), `rhoLoc` the 2-D level-k slab of rhoInSitu, `k` a level index (int or
    the traced KIdx of a level scan). Returns rhoLoc. :60-72 on every point (1-OLx:sNx+OLx, 1-OLy:sNy+OLy): `kl =
    MIN( MAX(k,kSurfC), MAX(kLowC,1) )` (INTEGER MIN/MAX: no tie or NaN case), tLoc / sLoc = tFld / sFld at level
    kl; then FIND_RHO_2D (:74-78) with kRef = k on the full range, as DO_OCEANIC_PHYS's own call does."""
    sz = cfg.size
    kv = getattr(k, "value", k)
    kSurfC = grid.kSurfC.data
    kLowC = grid.kLowC.data
    kl = jnp.minimum(jnp.maximum(kv, kSurfC), jnp.maximum(kLowC, 1))      # :68  MINMAX-RAW: INTEGER MIN/MAX
    idx = (kl - tFld.dims[2][1])[:, None]                                  # [T, 1, j, i] level index of kl

    def at_kl(fld):
        d = jnp.take_along_axis(fld.data, idx, axis=1)[:, 0]               # :69-70  fld(i,j,kl,bi,bj)
        return FArray(d, fld.name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))
    tLoc, sLoc = at_kl(tFld), at_kl(sFld)
    return find_rho_2d(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, k, tLoc, sLoc, rhoLoc, k,   # :74-78
                       cfg=cfg, grid=grid, params=params, eos=eos, state=state)
