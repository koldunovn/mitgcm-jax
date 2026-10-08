"""ECCO_COST_FINAL   @63cdc0b pkg/ecco/ecco_cost_final.F:6-171 (lane M4ADCOL session 3)

Called by COST_FINAL under useECCO (cost_final.F:97) with ifc .NE. -1 on the master thread; ECCO_VERBOSE and
ALLOW_ECCO_OLD_FC_PRINT are undefined (ECCO_OPTIONS.h:58, :61). Traced part `ecco_cost_final`, host printing
`ecco_cost_final_lines`.
"""

import jax.numpy as jnp

from mitjax.eesupp.global_sum import global_sum_tile
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.ecco.ecco_readparms import NGENCOST


def ecco_cost_final(cost, res, *, ep, mult=None, ex=None):
    """ECCO_COST_FINAL( ifc, optimcycle, myThid ) on cost.h (`cost`, updated) and COST_GENCOST_ALL's {k: (objf_gencost,
    num_gencost)} [tile] (`res`; the other k keep ECCO_COST_INIT_VARIA's zeros, ecco_cost_init_varia.F:47-52).
    :67-70 f_gencost = no_gencost = 0. _d 0; :73-87 per tile, num_var = 1..NGENCOST in order: tile_fc = tile_fc +
    mult_gencost(num_var)*objf_gencost, f_gencost += objf_gencost, no_gencost += num_gencost; :91-94 _GLOBAL_SUM_RL
    (one process: the tile sum). `mult`: {k: traced mult_gencost} (default the namelist's).
    Returns (cost, [(num_var, f_gencost, no_gencost, mult_gencost, gencost_name)] for every num_var)."""
    nT = cost.tile_fc.shape[0]
    z = jnp.zeros((nT,), jnp.float64)
    tile_fc = cost.tile_fc
    out = []
    gs = global_sum_tile if ex is None else ex.global_sum_tile
    for g in ep.gencost:                                                     # :76-84 (num_var = g.k)
        objf, num = res.get(g.k, (z, z))
        m = g.mult_gencost if mult is None else mult[g.k]
        tile_fc = tile_fc + m * objf
        out.append((g.k, gs(objf), gs(num), m, g.gencost_name))
    assert len(out) == NGENCOST
    cost.tile_fc = tile_fc
    return cost, out


def ecco_cost_final_lines(out):
    """(STDOUT records without the PID prefix, costfunction_ecco.0000 records) of :97-107 and :124-133 (host):
    for num_var with no_gencost .GT. 0, '(A,1PE22.14,I3,1X,1PE9.2,3A)' ' --> f_gencost  =', f, num_var, mult,
    ' (', name(1:IL), ')' and '(2A,I3.0,A,1PE22.14,1PE22.14,1X,1PE9.2)' name(1:MAX(IL,15)), ' (gencost ', num_var,
    ') = ', f, no, mult."""
    std, cf = [], []
    for nv, f, no, m, name in out:
        if float(no) > 0:                                                    # :98
            std.append(fortran_write("(A,1PE22.14,I3,1X,1PE9.2,3A)", " --> f_gencost  =", float(f), nv, float(m),
                                     " (", name.rstrip(" "), ")"))           # :99-104
    for nv, f, no, m, name in out:
        if float(no) > 0:                                                    # :125
            IL = max(len(name.rstrip(" ")), 15)   # :126-127  MINMAX-INT: INTEGER lengths
            cf.append(fortran_write("(2A,I3.0,A,1PE22.14,1PE22.14,1X,1PE9.2)", name.ljust(IL)[:IL], " (gencost ", nv,
                                    ") = ", float(f), float(no), float(m)))  # :128-131
    return std, cf
