"""cost.h (the parts tutorial_global_oce_optim/input_ad uses) and the shared helpers of pkg/cost.

    @63cdc0b pkg/cost/cost.h

Common-block fields with their Fortran names in `CostCommon` (a pytree: the traced cost state of a run):
    /COST_R/ fc, glofc (scalars), /COST_FINAL_R/ tile_fc(nSx,nSy) -> [tile],
    /COST_OBJFUNCTIONS/ objf_hflux_tut, objf_temp_tut (nSx,nSy) -> [tile],
    /COST_MEAN_R/ cMeanTheta, cMeanUVel, cMeanVVel, cMeanThetaUVel, cMeanThetaVVel
                  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy) FArrays (only the interior is ever written or read).
Namelist values (data.cost, COST_NML): mult_temp_tut, mult_hflux_tut, lastinterval: `_RL` (cost.h), traced floats.

Experiment-specific code: MITgcm compiles an experiment's own `code_ad/cost_*.F` in place of (or in addition to)
pkg/cost files. Here such a port lives in this package under the Fortran name, and its module sets
`CODE_DIR = "<experiment>/<code dir>"`; `require_code_dir(cfg, CODE_DIR)` refuses to run it for any other build, so
a routine of one experiment can never be used for another by accident (cost_temp.py, cost_hflux.py, cost_weights.py:
tutorial_global_oce_optim/code_ad).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mitjax.eesupp.global_sum import tile_sum_fortran


@jax.tree_util.register_dataclass
@dataclass
class CostCommon:
    fc: object
    glofc: object
    tile_fc: object
    objf_temp_tut: object
    objf_hflux_tut: object
    cMeanTheta: object
    cMeanUVel: object
    cMeanVVel: object
    cMeanThetaUVel: object
    cMeanThetaVVel: object
    objf_atl: object = None             # GOADK lane: /COST_OBJFUNCTIONS/ objf_atl(nSx,nSy) -> [tile] (cost.h:60)
    objf_tracer: object = None          # PTRACERS lane: /COST_OBJFUNCTIONS/ objf_tracer(nSx,nSy) -> [tile] (cost.h:62)
    objf_test: object = None            # GO lane: /COST_OBJFUNCTIONS/ objf_test(nSx,nSy) -> [tile] (cost.h:61)
    # lane M4ADCOL: SEAICE_COST.h /SEAICE_COST_R/ objf_ice, objf_ice_export, num_ice (nSx,nSy) -> [tile]
    # (SEAICE_COST.h:20-23), carried with the cost state (SEAICE_COST_INIT_VARIA, SEAICE_COST_TEST, SEAICE_COST_FINAL)
    objf_ice: object = None
    objf_ice_export: object = None
    num_ice: object = None


def require_code_dir(cfg, code_dir):
    """An experiment-specific routine runs only in the build that compiles it."""
    if f"{cfg.experiment}/{cfg.code_dir}" != code_dir:
        raise RuntimeError(f"{code_dir} routine called for {cfg.experiment}/{cfg.code_dir}")


def chain_sum(a):
    """`s = 0. _d 0; DO ...; s = s + a(...)` over the trailing axes of a [T, ..., ni] array in C order (the Fortran
    loop order when the axes are given outer to inner) -> [T]: the ordered add chain of mitjax/eesupp/global_sum."""
    a = jnp.asarray(a)
    return tile_sum_fortran(a.reshape(a.shape[0], -1, a.shape[-1]))
