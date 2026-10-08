"""CTRL_COST_GEN2D, CTRL_COST_GEN3D   @63cdc0b pkg/ctrl/ctrl_cost_gen.F:12-181, :187-339 (lane M4ADCOL)"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.pkg.cost.cost_h import chain_sum


def ctrl_cost_gen2d(startrec, endrec, records, xx_gen_weight, dodimensionalcost, xx_gen_mask2D, *, sz):
    """ctrl_cost_gen2d( startrec, endrec, xx_gen_file, xx_gen_dummy, xx_gen_period, xx_gen_weight,
                        dodimensionalcost, num_gen_anom, objf_gen_anom, xx_gen_mask2D, myThid )

    `records`: {irec: FArray} the records startrec..endrec of `xx_<name>.<optimcycle>` (the control vector, whose
    interior ACTIVE_READ_XY reads into tmpfld2d, :139-141; tmpfld2d is zeroed first, :111-119, and only its interior
    is read). Returns (num_gen_anom, objf_gen_anom) [tile]: per record (:136) and tile, fctile = 0. _d 0 (:151),
    fctile = fctile + tmpx*tmpx (or weight*tmpx*tmpx with dodimensionalcost, :157-161) over j, i where
    mask2D .ne. 0 (:152-168, a skipped point adds nothing: here +0. _d 0, exact), num + 1. _d 0 where also the weight
    .ne. 0 (:162-164), objf = objf + fctile (:170)."""
    j, i = loop_j(1, sz.sNy), loop_i(1, sz.sNx)
    m = xx_gen_mask2D[i, j] != 0.0                                     # :154
    w = xx_gen_weight[i, j]
    nT = w.shape[0]
    num = jnp.zeros((nT,), jnp.float64)                                # :130
    objf = jnp.zeros((nT,), jnp.float64)                               # :131
    for irec in range(startrec, endrec + 1):                           # :136
        tmpx = records[irec][i, j]                                     # :139-141, :156
        term = w * tmpx * tmpx if dodimensionalcost else tmpx * tmpx   # :157-161
        fctile = chain_sum(jnp.where(m, term, 0.0))                    # :151-168
        num = num + chain_sum(jnp.where(m & (w != 0.0), 1.0, 0.0))     # :162-164  1. _d 0
        objf = objf + fctile                                           # :170
    return num, objf


def ctrl_cost_gen3d(record, xx_gen_weight, dodimensionalcost, xx_gen_mask, *, sz):
    """ctrl_cost_gen3d( xx_gen_file, xx_gen_dummy, xx_gen_weight, dodimensionalcost, num_gen, objf_gen,
                        xx_gen_mask, myThid )

    `record`: record 1 of `xx_<name>.<optimcycle>` (ACTIVE_READ_XYZ, :297-302; interior read). Returns (num_gen,
    objf_gen) [tile]: objf_gen = objf_gen + tmpx*tmpx (weight*tmpx*tmpx with dodimensionalcost) over k, j, i where
    mask .ne. 0 (:314-331), num_gen + 1. _d 0 where also the weight .ne. 0 (:326-327)."""
    k, j, i = loops_kji((1, sz.Nr), (1, sz.sNy), (1, sz.sNx))
    m = xx_gen_mask[i, j, k] != 0.0                                    # :317
    w = xx_gen_weight[i, j, k]
    tmpx = record[i, j, k]                                             # :318
    term = w * tmpx * tmpx if dodimensionalcost else tmpx * tmpx       # :319-325
    objf = chain_sum(jnp.where(m, term, 0.0))                          # :311-312, :314-331
    num = chain_sum(jnp.where(m & (w != 0.0), 1.0, 0.0))               # :326-327
    return num, objf
