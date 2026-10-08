"""SEAICE_COST_INIT_VARIA   @63cdc0b pkg/seaice/seaice_cost_init_varia.F:3-54 (lane M4ADCOL)"""

import jax.numpy as jnp


def seaice_cost_init_varia(cost, *, cfg, ntiles):
    """SEAICE_COST_INIT_VARIA( myThid )

    C     o Initialise the variable cost function part.

    SEAICE_COST.h /SEAICE_COST_R/ objf_ice, objf_ice_export, num_ice (nSx,nSy) -> [tile], kept in the cost.h pytree
    (`CostCommon`, the cost state of a run): objf_ice = objf_ice_export = num_ice = 0. _d 0 (:37-39; ALLOW_COST,
    called from COST_INIT_VARIA :81). ALLOW_SEAICE_COST_EXPORT (:29, :40) is undefined in 1D_ocean_ice_column/code_ad
    (its block is not compiled)."""
    if cfg.cpp.flag("ALLOW_SEAICE_COST_EXPORT", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_COST_INIT_VARIA: ALLOW_SEAICE_COST_EXPORT (:40-47) is not ported")
    z = jnp.zeros((ntiles,), jnp.float64)
    cost.objf_ice = z                                                  # :37  0. _d 0
    cost.objf_ice_export = z                                           # :38  0. _d 0
    cost.num_ice = z                                                   # :39  0. _d 0
    return cost
