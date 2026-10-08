"""EXF_FILTER_RL: pkg/exf/exf_filter_rl.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j


def exf_filter_rl(arr, ckind, *, cfg, grid, params):
    """EXF_FILTER_RL( arr, ckind, myThid )   @63cdc0b pkg/exf/exf_filter_rl.F:3-89

    C     o Apply a mask (c: maskC, w: maskW, s: maskS at the surface level) to a forcing field: 0 where masked.

    `ckind` static. Returns arr."""
    sz = cfg.size
    if ckind == " ":                                                           # :38
        return arr
    ks = sz.Nr if params.usingPCoords else 1                                   # :40-41
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    msk = {"c": grid.maskC, "w": grid.maskW, "s": grid.maskS}.get(ckind)       # :58-82
    if msk is None:
        return arr
    return arr.at[i, j].set(jnp.where(msk[i, j, ks] == 0., 0., arr[i, j]))
