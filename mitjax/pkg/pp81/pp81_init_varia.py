"""PP81_INIT_VARIA: pkg/pp81/pp81_init_varia.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.pkg.pp81.pp81_h import declare


def pp81_init_varia(*, cfg, params, pp):
    """PP81_INIT_VARIA( myThid )   @63cdc0b pkg/pp81/pp81_init_varia.F:3-50

    C     | SUBROUTINE PP81_INIT_VARIA
    C     | o Routine to initialize PP81 parameters and variables.
    C     | Initialize PP81 parameters and variables.

    Returns `pp` (PP81.h) with PPviscAr(i,j,k) = viscArNr(k) and PPdiffKr(i,j,k) = diffKrNrS(k) at every point of
    every tile, halos included (:34-45). The k loop's iterations are independent (each level reads only the PARAMS.h
    vectors), so it is written for all levels at once.
    """
    sz = cfg.size
    T = sz.nSx*sz.nSy
    shape = (T, sz.Nr, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx)
    vis = jnp.broadcast_to(jnp.asarray(params.viscArNr.data)[None, :, None, None], shape)    # :39
    dif = jnp.broadcast_to(jnp.asarray(params.diffKrNrS.data)[None, :, None, None], shape)   # :40
    return pp.replace(PPviscAr=declare("PPviscAr", sz, vis), PPdiffKr=declare("PPdiffKr", sz, dif))
