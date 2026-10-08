"""SET_GRID_FACTORS: model/src/set_grid_factors.F @63cdc0b."""

from mitjax.model.grid import declare


def set_grid_factors(grid, *, cfg, params):
    """SET_GRID_FACTORS( myThid )   @63cdc0b model/src/set_grid_factors.F:6-98

    C     | SUBROUTINE SET_GRID_FACTORS
    C     | o Initialise grid factors for the deep model

    Returns `grid` with deepFacC, deepFac2C, recip_deepFacC, recip_deepFac2C (Nr) and deepFacF, deepFac2F,
    recip_deepFacF, recip_deepFac2F (Nr+1), all 1. _d 0 (:48-59). `deepAtmosphere` = .FALSE. in every M1 variant
    (:62-90 not executed, docs/coverage/*.md): the deep-model branch (:60-92) raises.
    """
    Nr = cfg.size.Nr
    f = {}
    for name in ("deepFacC", "deepFac2C", "recip_deepFacC", "recip_deepFac2C"):      # :48-53
        a = declare(name, cfg.size)
        for k in range(1, Nr + 1):
            a = a.at[k].set(1.0)
        f[name] = a
    for name in ("deepFacF", "deepFac2F", "recip_deepFacF", "recip_deepFac2F"):      # :54-59
        a = declare(name, cfg.size)
        for k in range(1, Nr + 2):
            a = a.at[k].set(1.0)
        f[name] = a
    if params.deepAtmosphere:                                                        # :60-92
        raise NotImplementedError("SET_GRID_FACTORS: deepAtmosphere is not ported")
    return grid.replace(**f)
