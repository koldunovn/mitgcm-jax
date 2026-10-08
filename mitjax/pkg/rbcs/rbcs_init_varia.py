"""RBCS_INIT_VARIA: pkg/rbcs/rbcs_init_varia.F @63cdc0b (M3 Task 30)."""

import jax.numpy as jnp

from mitjax.farray import loops_kji
from mitjax.pkg.rbcs.rbcs_fields_h import rbcs_array


def rbcs_init_varia(*, cfg):
    """RBCS_INIT_VARIA( myThid )   @63cdc0b pkg/rbcs/rbcs_init_varia.F:4-85

    C     | SUBROUTINE RBCS_INIT_VARIA
    C     | o Routine to initialize RBCS data structures

    Returns the RBCS_FIELDS.h state RBCS_FIELDS_LOAD updates: rbcsLdRec(bi,bj) = 0 and the loaded records rbct0/1,
    rbcs0/1 and RBCtemp, RBCsalt zero on every point (:36-60); with DISABLE_RBCS_MOM undefined also rbcu0/1,
    rbcv0/1, RBCuVel, RBCvVel (:42-49). ALLOW_PTRACERS (:62-80) raises."""
    if cfg.cpp.ALLOW_PTRACERS:                                          # :62-80
        raise NotImplementedError("RBCS_INIT_VARIA: ALLOW_PTRACERS (rbcptr0/1, RBC_ptracers) is not ported")
    sz = cfg.size
    k3, j3, i3 = loops_kji((1, sz.Nr), (1 - sz.OLy, sz.sNy + sz.OLy), (1 - sz.OLx, sz.sNx + sz.OLx))
    names = ["rbct0", "rbcs0", "rbct1", "rbcs1", "RBCtemp", "RBCsalt"]                 # :50-55
    if not cfg.cpp.DISABLE_RBCS_MOM:
        names = ["rbcu0", "rbcv0", "rbcu1", "rbcv1", "RBCuVel", "RBCvVel"] + names    # :43-48
    out = {n: rbcs_array(n, sz).at[i3, j3, k3].set(0.) for n in names}                 # 0. _d 0
    out["rbcsLdRec"] = jnp.zeros((sz.nSx * sz.nSy,), jnp.int32)                         # :38
    return out
