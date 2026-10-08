"""INI_VERTICAL_GRID: model/src/ini_vertical_grid.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.model.grid import UNSET_RS, declare


def ini_vertical_grid(grid, *, cfg, params):
    """INI_VERTICAL_GRID( myThid )   @63cdc0b model/src/ini_vertical_grid.F:6-415

    C     *==========================================================*
    C     | SUBROUTINE INI_VERTICAL_GRID
    C     | o Initialise vertical gridding arrays
    C     *==========================================================*

    Returns `grid` with rkSign, gravitySign, drF, drC, rF, rC, recip_drF, recip_drC and the hybrid-sigma
    coefficients (aHybSigmF ... dBHybSigC). The k loops are recursions (rF, rC) or one statement per level; they run
    as Python loops over k on the non-tiled (Nr) / (Nr+1) arrays, in the Fortran order.

    Ported for the M1 variants (docs/coverage/*.md): `setInterFDr` = T and `setCenterDr` = F (delR given, delRc not),
    so :71-87 and :120-128 run and :89-100 / :103-119 raise; `rF(1)` set (z coordinates of water, :136-147) or the
    axis from rF(Nr+1) = top_Pres / seaLev_Z (:155-168; p coordinates, lane B Task 25); `selectSigmaCoord` = 0 (:203-215; the hybrid-sigma set-up :216-409
    raises). The relative-position check :170-192 is ported (STOP -> ValueError).
    """
    Nr = cfg.size.Nr
    delR = FArray(jnp.asarray(params.delR), "delR", k=(1, Nr), tiled=False)

    # :53-57
    rkSign = jnp.float64(-1.0)                                      # rkSign = -1. _d 0
    gravitySign = jnp.float64(-1.0)                                 # gravitySign = -1. _d 0
    if params.usingPCoords:
        gravitySign = jnp.float64(1.0)                              # gravitySign = 1. _d 0

    if not (params.setInterFDr or params.setCenterDr):              # :59-67
        raise ValueError("S/R INI_VERTICAL_GRID: neither delR nor delRc are defined\n"
                         "S/R INI_VERTICAL_GRID: Need at least 1 of the 2 (delR,delRc)\n"
                         "ABNORMAL END: S/R INI_VERTICAL_GRID")

    drF = declare("drF", cfg.size)
    drC = declare("drC", cfg.size)
    rF = declare("rF", cfg.size)
    rC = declare("rC", cfg.size)
    recip_drF = declare("recip_drF", cfg.size)
    recip_drC = declare("recip_drC", cfg.size)

    if params.setInterFDr:                                          # :71-87
        for k in range(1, Nr + 1):
            drF = drF.at[k].set(delR[k])
        for k in range(1, Nr + 1):
            if params.delR[k - 1] <= 0.:
                raise ValueError(f"S/R INI_VERTICAL_GRID: delR(k={k:4d} )={params.delR[k - 1]:16.8E}\n"
                                 "S/R INI_VERTICAL_GRID: Vert. grid spacing MUST BE > 0\n"
                                 "ABNORMAL END: S/R INI_VERTICAL_GRID")
    else:                                                           # :88-101
        raise NotImplementedError("INI_VERTICAL_GRID: drF from delRc (setInterFDr = F) is not ported")

    if params.setCenterDr:                                          # :103-119
        raise NotImplementedError("INI_VERTICAL_GRID: drC from delRc (setCenterDr = T) is not ported")
    else:                                                           # :120-128
        drC = drC.at[1].set(0.5*delR[1])                            # drC(1)  = 0.5 _d 0 *delR(1)
        for k in range(2, Nr + 1):
            drC = drC.at[k].set(0.5*(delR[k-1]+delR[k]))
        drC = drC.at[Nr+1].set(0.5*delR[Nr])

    # :136-168  r-position of interface (rF) and cell centre (rC)
    rF1 = params.rF1                                                # rF(1) as INI_PARMS left it
    if rF1 == UNSET_RS and params.usingZCoords and params.fluidIsWater:     # :136-139
        rF1 = params.seaLev_Z
    if rF1 != UNSET_RS:                                             # :140-147
        rF = rF.at[1].set(rF1)
        for k in range(1, Nr + 1):
            rF = rF.at[k+1].set(rF[k] + rkSign*drF[k])
        rC = rC.at[1].set(rF[1] + rkSign*drC[1])
        for k in range(2, Nr + 1):
            rC = rC.at[k].set(rC[k-1] + rkSign*drC[k])
    else:                                                           # :155-168 (lane B, Task 25: p coords)
        if params.usingPCoords:
            rF = rF.at[Nr+1].set(params.top_Pres)                   # :157  rF(Nr+1) = top_Pres
        else:
            rF = rF.at[Nr+1].set(params.seaLev_Z)                   # :159  rF(Nr+1) = seaLev_Z
        for k in range(Nr, 0, -1):                                  # :161-163
            rF = rF.at[k].set(rF[k+1] - rkSign*drF[k])
        rC = rC.at[Nr].set(rF[Nr+1] - rkSign*drC[Nr+1])             # :164
        for k in range(Nr, 1, -1):                                  # :165-167
            rC = rC.at[k-1].set(rC[k] - rkSign*drC[k])

    # :170-192  check the vertical discretization (host-side)
    checkRatio2 = 100.                                              # :171  100. (REAL*4, exact)
    checkRatio1 = 1.0 / checkRatio2                                 # :172  1. _d 0 / checkRatio2
    for k in range(1, Nr + 1):
        rCk, rFk, rFkp1 = float(rC[k]), float(rF[k]), float(rF[k+1])
        tmpRatio = 0.
        if (rCk - rFkp1) != 0.:
            tmpRatio = (rFk - rCk) / (rCk - rFkp1)
        if tmpRatio < checkRatio1 or tmpRatio > checkRatio2:
            raise ValueError(f"S/R INI_VERTICAL_GRID: Invalid relative position, level k={k:4d} :{tmpRatio:16.8E}\n"
                             "ABNORMAL END: S/R INI_VERTICAL_GRID")

    # :194-200  reciprocals
    for k in range(1, Nr + 2):
        recip_drC = recip_drC.at[k].set(1.0/drC[k])                 # 1. _d 0/drC(k)
    for k in range(1, Nr + 1):
        recip_drF = recip_drF.at[k].set(1.0/drF[k])

    # :202-409  hybrid-sigma vertical coordinate
    if params.selectSigmaCoord == 0:                                # :203-215
        hyb = {name: declare(name, cfg.size) for name in ("aHybSigmF", "bHybSigmF", "dAHybSigC", "dBHybSigC",
                                                           "aHybSigmC", "bHybSigmC", "dAHybSigF", "dBHybSigF")}
        for k in range(1, Nr + 2):
            for name in ("aHybSigmF", "bHybSigmF", "dAHybSigC", "dBHybSigC"):
                hyb[name] = hyb[name].at[k].set(0.0)                # 0. _d 0
        for k in range(1, Nr + 1):
            for name in ("aHybSigmC", "bHybSigmC", "dAHybSigF", "dBHybSigF"):
                hyb[name] = hyb[name].at[k].set(0.0)
    else:                                                           # :216-409
        raise NotImplementedError("INI_VERTICAL_GRID: hybrid-sigma coordinate (selectSigmaCoord /= 0) is not "
                                  "ported")

    return grid.replace(rkSign=rkSign, gravitySign=gravitySign, drF=drF, drC=drC, rF=rF, rC=rC,
                        recip_drF=recip_drF, recip_drC=recip_drC, **hyb)
