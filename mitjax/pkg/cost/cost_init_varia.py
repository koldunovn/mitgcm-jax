"""COST_INIT_VARIA and COST_DEPENDENT_INIT (forward part)
    @63cdc0b pkg/cost/cost_init_varia.F:3-97, pkg/cost/cost_dependent_init.F:3-...
"""

import jax.numpy as jnp

from mitjax.farray import FArray, loops_kji
from mitjax.pkg.cost.cost_h import CostCommon


def cost_init_varia(*, cfg, ntiles):
    """COST_INIT_VARIA( mythid )

    c--   Initialize the tiled cost function contributions.

    tile_fc = 0 (:41), objf_atl = 0 (:42), objf_test = 0 (:43), objf_tracer = 0 (:44),
    cMean* = 0 on the interior (:52-62, ALLOW_COST), fc = glofc = 0 (:90-91). ALLOW_COST_VECTOR, ALLOW_COST_STATE_FINAL, ALLOW_SEAICE,
    ALLOW_THSICE undefined (lane M4ADCOL: under ALLOW_SEAICE SEAICE_COST_INIT_VARIA, :80-82).
    objf_temp_tut/objf_hflux_tut are not initialised by the Fortran (COST_TEMP/COST_HFLUX
    set them in COST_FINAL): NaN here, so a read before COST_FINAL shows up. The cMean halos are never written or
    read by the Fortran: zero storage here."""
    sz = cfg.size
    if cfg.cpp.ALLOW_COST_VECTOR or cfg.cpp.ALLOW_COST_STATE_FINAL:
        raise NotImplementedError("COST_INIT_VARIA: ALLOW_COST_VECTOR/STATE_FINAL not ported")
    zero = jnp.zeros((ntiles, sz.Nr, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx), jnp.float64)
    dims = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy), k=(1, sz.Nr))
    k, j, i = loops_kji((1, sz.Nr), (1, sz.sNy), (1, sz.sNx))
    mean = {}
    for n in ("cMeanTheta", "cMeanUVel", "cMeanVVel", "cMeanThetaUVel", "cMeanThetaVVel"):
        mean[n] = FArray(zero, n, **dims).at[i, j, k].set(0.0)         # :55-59  0. _d 0
    cost = CostCommon(fc=jnp.float64(0.0), glofc=jnp.float64(0.0),             # :90-91
                      tile_fc=jnp.zeros((ntiles,), jnp.float64),               # :41
                      objf_temp_tut=jnp.full((ntiles,), jnp.nan), objf_hflux_tut=jnp.full((ntiles,), jnp.nan),
                      objf_atl=jnp.zeros((ntiles,), jnp.float64),              # :42 (GOADK lane)
                      objf_tracer=jnp.zeros((ntiles,), jnp.float64),           # :44 (PTRACERS lane)
                      objf_test=jnp.zeros((ntiles,), jnp.float64),             # :43 (GO lane)
                      **mean)
    if cfg.cpp.flag("ALLOW_SEAICE"):                                   # :80-82 (lane M4ADCOL)
        from mitjax.pkg.seaice.seaice_cost_init_varia import seaice_cost_init_varia
        cost = seaice_cost_init_varia(cost, cfg=cfg, ntiles=ntiles)    # :81
    return cost


def cost_dependent_init(cost, *, cfg):
    """COST_DEPENDENT_INIT( myThid ): `fc = 0. _d 0` (cost_dependent_init.F:44). The ALLOW_AUTODIFF part (:47-...,
    adfc and the adjoint seeds) is TAF's seeding of the reverse sweep, not a forward value: the JAX driver seeds the
    cotangent of fc (Task 16)."""
    cost.fc = jnp.float64(0.0)
    return cost
