"""COST_ATLANTIC_HEAT: pkg/cost/cost_atlantic_heat.F @63cdc0b (GOADK lane, M2: global_ocean.90x40x15/code_ad, the only
cost term of the input_ad family)."""

import jax.numpy as jnp
import numpy as np

# cost_atlantic_heat.F:39-48 PARAMETERs (ALLOW_COST_ATLANTIC_HEAT_DOMASS undefined: the #else values)
petawatt = 1.e+15                   # :39  parameter( petawatt = 1. _d +15 )
isecbeg, isecend, jsec = 69, 87, 28  # :42  parameter( isecbeg = 69, isecend = 87, jsec = 28 )
jsecbeg, jsecend, isec = 10, 27, 59  # :44  (read only by the zonal-transport branch, MERID_TRANSPORT undefined)
kmaxdepth = 14                      # :48  parameter ( kmaxdepth = 14 )


def cost_atlantic_heat(*, cfg, grid, params, cost, myXGlobalLo, myYGlobalLo):
    """cost_atlantic_heat( myThid )   @63cdc0b pkg/cost/cost_atlantic_heat.F:3-196

    C     | subroutine cost_atlantic_heat
    C     | o This routine computes the meridional heat transport.
    C     |   The current indices are for North Atlantic 29N
    C     |   2x2 global setup.

    Returns objf_atl [tile] (cost.h /COST_OBJFUNCTIONS/ objf_atl(nSx,nSy)). `grid`: GRID.h (dxG, maskS, maskC, drF);
    `params`: PARAMS.h (HeatCapacity_Cp, rhoConst: traced); `cost`: cost.h (cMeanVVel, cMeanThetaVVel);
    myXGlobalLo, myYGlobalLo: EEPARAMS.h (static ints; a single-process build, nPx = nPy = 1).

    Branches (global_ocean.90x40x15/code_ad COST_OPTIONS.h): ALLOW_COST_ATLANTIC_HEAT defined, ALLOW_COST_ATLANTIC_HEAT
    _DOMASS undefined; the routine's own `#define MERID_TRANSPORT` / `#undef ENERGYNORM` (:63-67) select the
    meridional transport across the row jg = jsec of :85-127 and objf_atl = sum*HeatCapacity_Cp*rhoConst/petawatt
    (:186-187). Raise: ALLOW_COST_ATLANTIC_HEAT_DOMASS (:116-120, :182-184) and a build without
    ALLOW_COST_ATLANTIC_HEAT (the routine is then empty and objf_atl keeps its value: not ported).

    Vectorisation: the bi,bj loop is the tile axis; the global indices jg, ig of every tile are static. The i loop
    (:94-112) accumulates vVel_bar, thetaVvel_bar, countTV, countV in the Fortran order (one add per i in turn) over
    the row j with jg = jsec of each tile (at most one; a tile without it adds nothing: its countTV stays 0, so the
    k loop :115-127 adds nothing, as in the Fortran, where the IF (jg .eq. jsec) skips it). Points outside
    isecbeg <= ig <= isecend add +0: x + (+0) = x for every x the running sums take (they start at +0 and a sum of +0
    with anything is never -0), so the masked chain gives the Fortran's values bit for bit. `0.0` is a REAL*4
    literal, exact. vVel_bar and countV are computed as the Fortran computes them (no output reads them).
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr, nSx, nSy = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr, sz.nSx, sz.nSy
    if not cfg.cpp.flag("ALLOW_COST_ATLANTIC_HEAT", "COST_OPTIONS.h"):
        raise NotImplementedError("COST_ATLANTIC_HEAT: a build without ALLOW_COST_ATLANTIC_HEAT is not ported")
    if cfg.cpp.flag("ALLOW_COST_ATLANTIC_HEAT_DOMASS", "COST_OPTIONS.h"):
        raise NotImplementedError("COST_ATLANTIC_HEAT: ALLOW_COST_ATLANTIC_HEAT_DOMASS is not ported")
    if sz.nPx != 1 or sz.nPy != 1:
        raise NotImplementedError("COST_ATLANTIC_HEAT: a multi-process tiling is not ported")

    T = nSx * nSy
    # static per-tile row and column selection (tile t = (bj-1)*nSx + (bi-1))
    jrow = np.zeros(T, dtype=np.int64)                              # storage index of the row jg = jsec
    hasrow = np.zeros(T, dtype=bool)
    incol = np.zeros((T, sNx), dtype=bool)                          # isecbeg <= ig <= isecend, i = 1..sNx
    for t in range(T):
        bi, bj = t % nSx + 1, t // nSx + 1
        for j in range(1, sNy + 1):                                 # :85-87
            jg = myYGlobalLo-1+(bj-1)*sNy+j
            if jg == jsec:
                jrow[t], hasrow[t] = j-1+OLy, True
        for i in range(1, sNx + 1):                                 # :94-97
            ig = myXGlobalLo-1+(bi-1)*sNx+i
            incol[t, i-1] = (ig >= isecbeg) and (ig <= isecend)
    sel = incol & hasrow[:, None]                                   # [tile, i]
    tix = np.arange(T)

    def row(a, k):
        """Level k, row jrow[t] of tile t, interior columns i = 1..sNx: [tile, sNx]."""
        d = a.data[:, k-1] if a.data.ndim == 4 else a.data
        return d[tix, jrow, OLx:OLx+sNx]

    dxG = grid.dxG
    sumT = jnp.zeros(T)                                             # :61  sum = 0.0
    for k in range(1, Nr + 1):                                      # :89
        vVel_bar = jnp.zeros(T)                                     # :90-93  0.0
        thetaVvel_bar = jnp.zeros(T)
        countV = jnp.zeros(T)
        countTV = jnp.zeros(T)
        vV, tV = row(cost.cMeanVVel, k), row(cost.cMeanThetaVVel, k)
        dx, mS, mC = row(dxG, k), row(grid.maskS, k), row(grid.maskC, k)
        for i in range(sNx):                                        # :94-112
            m = sel[:, i]
            vVel_bar = vVel_bar + jnp.where(m, vV[:, i]*dx[:, i]
                                            * mS[:, i], 0.)          # :98-100
            thetaVvel_bar = thetaVvel_bar + jnp.where(m, tV[:, i]*dx[:, i]
                                                      * mS[:, i]*mC[:, i], 0.)   # :102-104
            countTV = countTV + jnp.where(m, mS[:, i]*mC[:, i], 0.)     # :106-107
            countV = countV + jnp.where(m, mS[:, i], 0.)                # :108-109
        if k <= kmaxdepth:                                          # :122-125
            nz = countTV != 0
            sumT = jnp.where(nz, sumT
                             + thetaVvel_bar*grid.drF[k]/jnp.where(nz, countTV, 1.), sumT)
    return sumT*params.HeatCapacity_Cp*params.rhoConst/petawatt      # :186-187
