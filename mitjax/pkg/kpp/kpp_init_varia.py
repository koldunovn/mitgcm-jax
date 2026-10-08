"""KPP_INIT_VARIA: pkg/kpp/kpp_init_varia.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j, loops_kji


def kpp_init_varia(kpp, kppf, *, cfg, grid, params):
    """KPP_INIT_VARIA( myThid )   @63cdc0b pkg/kpp/kpp_init_varia.F:7-95

    C     | SUBROUTINE KPP_INIT_VARIA
    C     | o Routine to initialize KPP parameters and variables.
    C     | Initialize KPP parameters and variables.

    `kpp`: the Kpp (KPP_PARAMS.h) of KPP_INIT_FIXED; `kppf`: the KPP.h fields {name: FArray} (kpp_params_h.kpp_fields;
    KPPfrac keeps its value: the routine does not set it without SHORTWAVE_HEATING). Returns (kpp with nzmax =
    kLowC (:42, INTEGER (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy), int32), kppf with KPPhbl = 0. (:63), KPPghat = 0.,
    KPPviscAz = viscArNr(1), KPPdiffKzS = KPPdiffKzT = 0. (:69-72); the REAL*4 literals 0. are exact). Every point,
    halos included; the points are independent."""
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :40-41
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    kLowC = jnp.asarray(grid.kLowC[i, j]).astype(jnp.int32)
    nzmax = FArray(kLowC, "nzmax", i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))   # :42
    kppf = dict(kppf)
    kppf["KPPhbl"] = kppf["KPPhbl"].at[i, j].set(0.)                            # :63
    k3, j3, i3 = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :66-68
    kppf["KPPghat"] = kppf["KPPghat"].at[i3, j3, k3].set(0.)                    # :69
    kppf["KPPviscAz"] = kppf["KPPviscAz"].at[i3, j3, k3].set(params.viscArNr[1])   # :70
    kppf["KPPdiffKzS"] = kppf["KPPdiffKzS"].at[i3, j3, k3].set(0.)              # :71
    kppf["KPPdiffKzT"] = kppf["KPPdiffKzT"].at[i3, j3, k3].set(0.)              # :72
    return kpp.replace(nzmax=nzmax), kppf
