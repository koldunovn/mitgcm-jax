"""GMREDI_INIT_FIXED: pkg/gmredi/gmredi_init_fixed.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.pkg.gmredi.gmredi_h import declare


def gmredi_init_fixed(gm, *, cfg):
    """GMREDI_INIT_FIXED( myThid )   @63cdc0b pkg/gmredi/gmredi_init_fixed.F:6-153

    C     | SUBROUTINE GMREDI_INIT_FIXED
    C     | o Routine to initialize GM/Redi variables
    C     |   that are kept fixed during the run.

    Returns `gm` with GM_isoFac2d, GM_bolFac2d (every point, halos included) and GM_isoFac1d, GM_bolFac1d set.
    Ported: :44-59, :74-77. Not ported (raise): the input files GM_iso2dFile, GM_bol2dFile (READ_FLD_XY_RS +
    EXCH_XY_RS, :62-69), GM_iso1dFile, GM_bol1dFile (READ_GLVEC_RS, :80-87), the K3D input files GM_K3dRediFile,
    GM_K3dGMFile (:107-112, :130-135) and GM_GEOM_VARIABLE_K (:50-55); no M1/M2 run sets them. GOADK lane (M2,
    global_ocean.90x40x15/code_ad): #ifdef GM_READ_K3D_REDI / GM_READ_K3D_GM the background 3-D diffusivities
    GM_inpK3dRedi = GM_isopycK*GM_isoFac1d(k)*GM_isoFac2d(i,j), GM_inpK3dGM = GM_background_K*GM_bolFac1d(k)
    *GM_bolFac2d(i,j) at every point (:93-105, :116-128; constants per level, the k loop k-vectorised). Not ported
    (output only): GMREDI_MNC_INIT (:140-145, ALLOW_MNC) and GMREDI_DIAGNOSTICS_INIT (:146-150, which only
    registers diagnostics).
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    if cfg.cpp.GM_GEOM_VARIABLE_K:
        raise NotImplementedError("GMREDI_INIT_FIXED: GM_GEOM_VARIABLE_K (GEOM_taper) is not ported")

    GM_isoFac2d = declare("GM_isoFac2d", sz)
    GM_bolFac2d = declare("GM_bolFac2d", sz)
    GM_isoFac1d = declare("GM_isoFac1d", sz)
    GM_bolFac1d = declare("GM_bolFac1d", sz)

    j = loop_j(1-OLy, sNy+OLy)                                      # :44-59
    i = loop_i(1-OLx, sNx+OLx)
    GM_isoFac2d = GM_isoFac2d.at[i, j].set(1.)                      # 1. _d 0
    GM_bolFac2d = GM_bolFac2d.at[i, j].set(1.)                      # 1. _d 0

    for name in ("GM_iso2dFile", "GM_bol2dFile"):                   # :62-69
        if getattr(gm, name).strip() != "":
            raise NotImplementedError(f"GMREDI_INIT_FIXED: {name} (READ_FLD_XY_RS) is not ported")

    for k in range(1, Nr + 1):                                      # :74-77
        GM_isoFac1d = GM_isoFac1d.at[k].set(1.)                     # 1. _d 0
        GM_bolFac1d = GM_bolFac1d.at[k].set(1.)                     # 1. _d 0

    for name in ("GM_iso1dFile", "GM_bol1dFile"):                   # :80-87
        if getattr(gm, name).strip() != "":
            raise NotImplementedError(f"GMREDI_INIT_FIXED: {name} (READ_GLVEC_RS) is not ported")

    out = {}
    if cfg.cpp.GM_READ_K3D_REDI:                                    # :93-115 (GOADK lane)
        k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))   # :95-105
        GM_inpK3dRedi = gm.GM_inpK3dRedi if "GM_inpK3dRedi" in gm else declare("GM_inpK3dRedi", sz)
        out["GM_inpK3dRedi"] = GM_inpK3dRedi.at[i, j, k].set(gm.GM_isopycK
                                                             * GM_isoFac1d[k]*GM_isoFac2d[i, j])
        if gm.GM_K3dRediFile.strip() != "":                          # :107-112
            raise NotImplementedError("GMREDI_INIT_FIXED: GM_K3dRediFile (READ_FLD_XYZ_RL) is not ported")
    if cfg.cpp.GM_READ_K3D_GM:                                      # :116-138 (GOADK lane)
        k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))   # :118-128
        GM_inpK3dGM = gm.GM_inpK3dGM if "GM_inpK3dGM" in gm else declare("GM_inpK3dGM", sz)
        out["GM_inpK3dGM"] = GM_inpK3dGM.at[i, j, k].set(gm.GM_background_K
                                                         * GM_bolFac1d[k]*GM_bolFac2d[i, j])
        if gm.GM_K3dGMFile.strip() != "":                            # :130-135
            raise NotImplementedError("GMREDI_INIT_FIXED: GM_K3dGMFile (READ_FLD_XYZ_RL) is not ported")
    return gm.replace(GM_isoFac2d=GM_isoFac2d, GM_bolFac2d=GM_bolFac2d, GM_isoFac1d=GM_isoFac1d,
                      GM_bolFac1d=GM_bolFac1d, **out)
