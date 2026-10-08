"""LOAD_GRID_SPACING: model/src/load_grid_spacing.F @63cdc0b."""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.model.grid import UNSET_RL


def _blank(s):
    return str(s).strip() == ""


def load_grid_spacing(*, cfg, params):
    """LOAD_GRID_SPACING( myThid )   @63cdc0b model/src/load_grid_spacing.F:11-182

    C     !DESCRIPTION:
    C     load grid-spacing (vector array) delX, delY, delR or delRc from file.

    Returns (delX, delY, latBandClimRelax): SET_GRID.h `delX(grid_maxNx)`, `delY(grid_maxNy)` as non-tiled FArrays
    (as INI_PARMS left them: ini_parms_grid), and PARAMS.h `latBandClimRelax` after its default (:166-175).

    Ported for the M1 variants: no grid-spacing file (`delXFile`, `delYFile`, `delRFile`, `delRcFile`,
    `hybSigmFile` all blank; :59-106 not executed, docs/coverage/*.md), so each file branch raises; the delX/delY
    checks (:108-164) and the latBandClimRelax default (:166-175) are ported. `gridNx` is exch2_mydNx(1) under
    ALLOW_EXCH2 (:48-54). Host-side checks: a STOP of the Fortran raises ValueError with its message.
    """
    sz = cfg.size
    if cfg.cpp.ALLOW_EXCH2:                                         # :48-54
        gridNx = params.exch2.exch2_mydNx[0]
        gridNy = params.exch2.exch2_mydNy[0]
    else:
        gridNx = sz.Nx
        gridNy = sz.Ny

    for key in ("delXFile", "delYFile", "delRFile", "delRcFile", "hybSigmFile"):   # :59-106
        if not _blank(getattr(params, key)):
            raise NotImplementedError(f"LOAD_GRID_SPACING: {key} = {getattr(params, key)!r} is not ported")

    delX = np.asarray(params.delX, dtype=np.float64)
    delY = np.asarray(params.delY, dtype=np.float64)

    if not params.usingCurvilinearGrid:                             # :109-164
        n = 0                                                       # :114-137 check delX
        msgs = []
        for i in range(1, gridNx + 1):
            if delX[i - 1] == UNSET_RL:
                n = n + 1
                msgs.append(f"S/R LOAD_GRID_SPACING: No value for delX at i ={i:5d}")
            if delX[i - 1] <= 0.:
                n = n + 1
                msgs.append(f"S/R LOAD_GRID_SPACING: delX(i={i:5d})={delX[i - 1]:16.8E} : MUST BE >0")
        if n >= 1:
            raise ValueError("\n".join(msgs + [f"S/R LOAD_GRID_SPACING: found{n:5d} invalid delX values",
                                               "ABNORMAL END: S/R LOAD_GRID_SPACING"]))
        n = 0                                                       # :139-162 check delY
        msgs = []
        for j in range(1, gridNy + 1):
            if delY[j - 1] == UNSET_RL:
                n = n + 1
                msgs.append(f"S/R LOAD_GRID_SPACING: No value for delY at j ={j:5d}")
            if delY[j - 1] <= 0.:
                n = n + 1
                msgs.append(f"S/R LOAD_GRID_SPACING: delY(j={j:5d})={delY[j - 1]:16.8E} : MUST BE >0")
        if n >= 1:
            raise ValueError("\n".join(msgs + [f"S/R LOAD_GRID_SPACING: found{n:5d} invalid delY values",
                                               "ABNORMAL END: S/R LOAD_GRID_SPACING"]))

    latBandClimRelax = params.latBandClimRelax
    if params.usingCartesianGrid or params.usingSphericalPolarGrid:  # :167-175
        delYsum = 0.
        for j in range(1, gridNy + 1):                              # sequential sum, Fortran order
            delYsum = delYsum + float(delY[j - 1])
        if latBandClimRelax == UNSET_RL:
            latBandClimRelax = delYsum*3.0                          # :173  delYsum*3. _d 0

    delX = FArray(jnp.asarray(delX), "delX", i=(1, delX.shape[0]), tiled=False)
    delY = FArray(jnp.asarray(delY), "delY", j=(1, delY.shape[0]), tiled=False)
    return delX, delY, latBandClimRelax
