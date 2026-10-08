"""GMREDI_INIT_VARIA: pkg/gmredi/gmredi_init_varia.F @63cdc0b."""

from mitjax.farray import loops_kji
from mitjax.pkg.gmredi.gmredi_h import declare


def gmredi_init_varia(gm, *, cfg):
    """GMREDI_INIT_VARIA( myThid )   @63cdc0b pkg/gmredi/gmredi_init_varia.F:9-226

    C     | SUBROUTINE GMREDI_INIT_VARIA
    C     | o Routine to initialize GM/Redi variables that are
    C     |   time-dependent (i.e. not just a function of the grid)

    Returns `gm` with the tensor Kwx, Kwy, Kwz, Kux, Kvy (and Kuz, Kvz under GM_EXTRA_DIAGONAL, GM_PsiX, GM_PsiY under
    GM_BOLUS_ADVEC) zero at every point, halos included (:101-132). Under ALLOW_AUTODIFF the body is inside
    `IF ( useGMRedi )` (:70-76, :220-222); the routine is only called with useGMRedi (packages_init_variables.F), so
    the IF holds. The k loop (:105-132) writes constants: its iterations are independent (one k-vectorised nest).
    GOADK lane (M2, global_ocean.90x40x15/code_ad): ALLOW_KAPREDI/KAPGM_CONTROL with GM_READ_K3D_* reset
    GM_inpK3dRedi / GM_inpK3dGM to their background values at every point (:38-67, before the IF ( useGMRedi ));
    the re-read of GM_K3dRediFile / GM_K3dGMFile (:84-99) raises (no run sets them). M3 lane MLAdjust:
    ALLOW_GM_LEITH_QG zeroes GM_LeithQG_K (:121-123). Not ported (raise):
    GM_BATES_K3D, GM_GEOM_VARIABLE_K, GM_VISBECK_VARIABLE_K (:121-144, :167-218). Not ported (output only): the
    WRITE_GLVEC_RS / WRITE_FLD_XY_RS of the 1-D and 2-D factors when their input files are set (:152-165; the inputs
    themselves raise in GMREDI_INIT_FIXED).
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    for opt in ("GM_BATES_K3D", "GM_GEOM_VARIABLE_K", "GM_VISBECK_VARIABLE_K"):
        if getattr(cfg.cpp, opt):
            raise NotImplementedError(f"GMREDI_INIT_VARIA: {opt} is not ported")
    out = {}
    # GOADK lane (M2, global_ocean.90x40x15/code_ad): the K3D control re-initialisation :38-67 (both k loops write
    # constants per level: k-vectorised)
    kapredi = cfg.cpp.ALLOW_CTRL and cfg.cpp.flag("ALLOW_KAPREDI_CONTROL", "CTRL_OPTIONS.h") and \
        cfg.cpp.GM_READ_K3D_REDI
    kapgm = cfg.cpp.ALLOW_CTRL and cfg.cpp.flag("ALLOW_KAPGM_CONTROL", "CTRL_OPTIONS.h") and cfg.cpp.GM_READ_K3D_GM
    if kapredi:                                                     # :38-52
        k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
        out["GM_inpK3dRedi"] = gm.GM_inpK3dRedi.at[i, j, k].set(gm.GM_isopycK
                                                                * gm.GM_isoFac1d[k]*gm.GM_isoFac2d[i, j])
    if kapgm:                                                       # :53-67
        k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
        out["GM_inpK3dGM"] = gm.GM_inpK3dGM.at[i, j, k].set(gm.GM_background_K
                                                            * gm.GM_bolFac1d[k]*gm.GM_bolFac2d[i, j])
    if cfg.cpp.ALLOW_AUTODIFF and not cfg.use_flag("useGMRedi"):    # :75  IF ( useGMRedi ) THEN
        return gm.replace(**out)
    if kapredi and gm.GM_K3dRediFile.strip() != "":                 # :84-91
        raise NotImplementedError("GMREDI_INIT_VARIA: GM_K3dRediFile (READ_FLD_XYZ_RL) is not ported")
    if kapgm and gm.GM_K3dGMFile.strip() != "":                     # :92-99
        raise NotImplementedError("GMREDI_INIT_VARIA: GM_K3dGMFile (READ_FLD_XYZ_RL) is not ported")

    names = ["Kwx", "Kwy", "Kwz", "Kux", "Kvy"]
    if cfg.cpp.GM_EXTRA_DIAGONAL:                                   # :113-116
        names += ["Kuz", "Kvz"]
    if cfg.cpp.GM_BOLUS_ADVEC:                                      # :117-120
        names += ["GM_PsiX", "GM_PsiY"]
    if cfg.cpp.ALLOW_GM_LEITH_QG:                                   # :121-123 (M3 lane MLAdjust)
        names += ["GM_LeithQG_K"]
    k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))  # :105-132
    for name in names:
        A = getattr(gm, name) if name in gm else declare(name, sz)
        out[name] = A.at[i, j, k].set(0.)                           # 0. _d 0
    return gm.replace(**out)
