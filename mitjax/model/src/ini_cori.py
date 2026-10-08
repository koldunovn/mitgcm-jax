"""INI_CORI: model/src/ini_cori.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import declare, deg2rad


def ini_cori(grid, *, cfg, params):
    """INI_CORI( myThid )   @63cdc0b model/src/ini_cori.F:8-219

    C     | SUBROUTINE INI_CORI
    C     | o Initialise coriolis parameter
    C     | Set Coriolis parameter (fCori at grid Center point,
    C     |  fCoriG at grid Corner point and fCoriCos for NH terms)

    Returns `grid` with fCori, fCoriG, fCoriCos. `selectCoriMap` (set_parms.F:73-81 default) 0, 1 and 2 are ported
    (:55-100) and any other value zeroes the fields (:104-117); 3 (read from files, :119-181) raises. The point loops
    are pointwise and run vectorised over the full tile; association as in the Fortran (f0+(beta*yC)*facGrid,
    (2. _d 0*omega)*sin(yC*deg2rad)). SIN, COS: XLA:CPU's are bitwise glibc 2.28 (Task 7c). The ALLOW_MONITOR
    statistics printout (:183-216, STDOUT only) is not ported: it writes no model variable.
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    yC, yG = grid.yC, grid.yG
    f0, beta, fPrime, omega = params.f0, params.beta, params.fPrime, params.omega
    fCori, fCoriG, fCoriCos = (declare(n, sz) for n in ("fCori", "fCoriG", "fCoriCos"))
    j = loop_j(1-OLy, sNy+OLy)                                      # DO j=1-Oly,sNy+Oly (Fortran spells OLy Oly)
    i = loop_i(1-OLx, sNx+OLx)
    sel = params.selectCoriMap
    if sel == 0:                                                    # :55-67  Constant F case
        fCori = fCori.at[i, j].set(f0)
        fCoriG = fCoriG.at[i, j].set(f0)
        fCoriCos = fCoriCos.at[i, j].set(fPrime)
    elif sel == 1:                                                  # :68-83  Beta plane case
        facGrid = 1.0                                               # 1. _d 0
        if params.usingSphericalPolarGrid or params.usingCurvilinearGrid:
            facGrid = deg2rad*params.rSphere
        fCori = fCori.at[i, j].set(f0+beta*yC[i, j]*facGrid)
        fCoriG = fCoriG.at[i, j].set(f0+beta*yG[i, j]*facGrid)
        fCoriCos = fCoriCos.at[i, j].set(fPrime)
    elif sel == 2:                                                  # :84-103  Spherical case
        fCori = fCori.at[i, j].set(2.0*omega*jnp.sin(yC[i, j]*deg2rad))
        fCoriG = fCoriG.at[i, j].set(2.0*omega*jnp.sin(yG[i, j]*deg2rad))
        fCoriCos = fCoriCos.at[i, j].set(2.0*omega*jnp.cos(yC[i, j]*deg2rad))
    else:                                                           # :104-117  Initialise to zero
        fCori = fCori.at[i, j].set(0.)
        fCoriG = fCoriG.at[i, j].set(0.)
        fCoriCos = fCoriCos.at[i, j].set(0.)
    if sel == 3:                                                    # :119-181
        raise NotImplementedError("INI_CORI: selectCoriMap = 3 (fCori read from files) is not ported")
    return grid.replace(fCori=fCori, fCoriG=fCoriG, fCoriCos=fCoriCos)

