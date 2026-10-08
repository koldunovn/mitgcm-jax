"""KPP_CALC_DUMMY: pkg/kpp/kpp_calc.F @63cdc0b, routine KPP_CALC_DUMMY (kpp_calc.F:723-799; the other routines of
kpp_calc.F, KPP_CALC and its callees, are M3 work)."""

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.model.src.calc_3d_diffusivity import calc_3d_diffusivity
from mitjax.pkg.generic_advdiff.gad_h import GAD_SALINITY, GAD_TEMPERATURE

KPP_FIELDS = ("KPPhbl", "KPPfrac", "KPPghat", "KPPviscAz", "KPPdiffKzS", "KPPdiffKzT")   # KPP.h:28-38


def kpp_calc_dummy(myTime, myIter, *, cfg, grid, params, state, kpp):
    """KPP_CALC_DUMMY( bi, bj, myTime, myIter, myThid )   @63cdc0b pkg/kpp/kpp_calc.F:726-799

    C     | SUBROUTINE KPP_CALC_DUMMY                                |
    C     | o Compute all KPP fields defined in KPP.h                |
    C     | o Dummy routine for TAF                                  |

    DO_OCEANIC_PHYS calls it when pkg/kpp is compiled but useKPP is .FALSE. (do_oceanic_phys.F:959-962). `kpp`: the
    KPP.h fields {name: FArray} (KPP_FIELDS); returns the new dict: KPPhbl = 1.0, KPPfrac = 0.0, KPPghat = 0.0,
    KPPviscAz = viscArNr(1) on every point (:767-779; REAL*4 literals, exact), KPPdiffKzS and KPPdiffKzT from
    CALC_3D_DIFFUSIVITY with GAD_SALINITY / GAD_TEMPERATURE, no GM/Redi and no KPP, over the whole tile
    (:786-795). The forward model reads none of these when useKPP is .FALSE. (only TAF's adjoint does), but they are
    KPP.h state and are ported. Raise: ALLOW_SALT_PLUME (KPPplumefrac, :771-773). The CADJ STOREs (:783-784) are
    TAF directives. The (i, j, k) points are independent (each writes its own point from constants)."""
    if not cfg.cpp.flag("ALLOW_KPP"):                                           # :759
        return kpp
    if cfg.cpp.flag("ALLOW_SALT_PLUME"):
        raise NotImplementedError("KPP_CALC_DUMMY: ALLOW_SALT_PLUME (KPPplumefrac) is not ported")
    sz = cfg.size
    kpp = dict(kpp)
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :767
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :768
    kpp["KPPhbl"] = kpp["KPPhbl"].at[i, j].set(1.0)                             # :769
    kpp["KPPfrac"] = kpp["KPPfrac"].at[i, j].set(0.0)                           # :770
    k3, j3, i3 = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :774
    kpp["KPPghat"] = kpp["KPPghat"].at[i3, j3, k3].set(0.0)                     # :775
    kpp["KPPviscAz"] = kpp["KPPviscAz"].at[i3, j3, k3].set(params.viscArNr[1])  # :776
    kpp["KPPdiffKzS"] = calc_3d_diffusivity(                                    # :786-790
        1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, GAD_SALINITY, False, False, kpp["KPPdiffKzS"],
        cfg=cfg, grid=grid, params=params, state=state)
    kpp["KPPdiffKzT"] = calc_3d_diffusivity(                                    # :791-795
        1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, GAD_TEMPERATURE, False, False, kpp["KPPdiffKzT"],
        cfg=cfg, grid=grid, params=params, state=state)
    return kpp
