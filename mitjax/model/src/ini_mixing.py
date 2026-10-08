"""INI_MIXING: model/src/ini_mixing.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_XYZ_RL
from mitjax.farray import loops_kji
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL


def ini_mixing(state, diffKrFile, *, cfg, params, ex, rw):
    """INI_MIXING( myThid )   @63cdc0b model/src/ini_mixing.F:6-73

    C     | SUBROUTINE INI_MIXING
    C     | o Initialise diffusivity to default constant value.

    Returns the State. Under ALLOW_3D_DIFFKR (:39-55): DYNVARS.h diffKr(i,j,k) = diffKrNrS(k) on every point (`params`:
    PARAMS.h diffKrNrS of INI_PARMS), then, if diffKrFile (PARAMS.h, PARM05; set_defaults.F:370 ' ') is set, the field
    read from it (READ_FLD_XYZ_RL) and exchanged (_EXCH_XYZ_RL). The points are independent (k-vectorised). Raise:
    ALLOW_BL79_LAT_VARY (:57-70, BL79LatArray). PTRACERS lane (tutorial_tracer_adjsens)."""
    if cfg.cpp.ALLOW_BL79_LAT_VARY:
        raise NotImplementedError("INI_MIXING: ALLOW_BL79_LAT_VARY (BL79LatArray) is not ported")
    if not cfg.cpp.ALLOW_3D_DIFFKR:
        return state
    sz = cfg.size
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :40-44
    diffKr = state.diffKr.at[i, j, k].set(params.diffKrNrS[k])                  # :45
    if str(diffKrFile).strip() != "":                                           # :51
        diffKr = READ_FLD_XYZ_RL(diffKrFile, " ", diffKr, 0, rw=rw)             # :52
        diffKr = EXCH_XYZ_RL(diffKr, ex=ex)                                     # :53
    return state.replace(diffKr=diffKr)
