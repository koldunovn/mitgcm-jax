"""GMREDI_CALC_PSI_BOLUS: pkg/gmredi/gmredi_calc_psi_bolus.F @63cdc0b (GOADK lane, M2:
global_ocean.90x40x15/code_ad)."""

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import halfRL
from mitjax.pkg.gmredi.gmredi_h import op25
from mitjax.pkg.gmredi.gmredi_slope_psi import gmredi_slope_psi


def gmredi_calc_psi_bolus(iMin, iMax, jMin, jMax, sigmaX, sigmaY, sigmaR, locK3dGM, ldd97_LrhoW, ldd97_LrhoS,
                          *, cfg, grid, params, gm):
    """GMREDI_CALC_PSI_BOLUS( bi, bj, iMin, iMax, jMin, jMax, sigmaX, sigmaY, sigmaR, locK3dGM, ldd97_LrhoW,
                              ldd97_LrhoS, myThid )
    @63cdc0b pkg/gmredi/gmredi_calc_psi_bolus.F:16-201

    C     | SUBROUTINE GMREDI_CALC_PSI_BOLUS
    C     | o Calculate stream-functions for GM bolus velocity

    sigmaX, sigmaY, sigmaR, locK3dGM: FArrays (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr), tiled (the caller's per-tile
    locals); ldd97_LrhoW/S (1-OLx:sNx+OLx, 1-OLy:sNy+OLy). `grid`: GRID.h (maskW, maskS, rF, gravitySign);
    `params`: PARAMS.h (usingZCoords static; wUnit2rVel, rVel2wUnit for GMREDI_SLOPE_PSI); `gm`: GMREDI.h.
    Returns `gm` with GM_PsiX, GM_PsiY written at levels k = 2..Nr on the points of :149-169 / :170-190 (level 1 and
    the first column / row keep their values: GMREDI_CALC_TENSOR zeroes them first, :223-232). iMin..jMax are not
    read (as in the Fortran).

    Branches ported (global_ocean.90x40x15/code_ad, GMREDI_OPTIONS.h: GM_READ_K3D_GM defined, so GM_USE_K3D_GM,
    :8-11; GM_GEOM_VARIABLE_K undefined; ALLOW_EDDYPSI undefined in CPP_OPTIONS.h): the whole routine under
    GM_BOLUS_ADVEC, IF ( GM_AdvForm ) (:87-196), the GM_USE_K3D_GM form of the stream functions (:152-155,
    :173-176); PTRACERS lane (tutorial_tracer_adjsens/code_ad): without GM_USE_K3D_GM the half_K*GM_bolFac2d form
    (:147-148, :156-159, :177-180; GM_bolFac1d/2d of GMREDI.h). Not ported (raise): GM_GEOM_VARIABLE_K (:160-162, :181-183), ALLOW_EDDYPSI
    (:165-167, :186-188). Under ALLOW_AUTODIFF the locals SlopeX, SlopeY, dSigmaDrW, dSigmaDrS are zeroed at every
    point of each level first (:94-103); without it they are NaN where the Fortran leaves them unset (outer column /
    row), which GMREDI_SLOPE_PSI never reads.

    The k loop (:91-193) calls GMREDI_SLOPE_PSI once per level; nothing is carried between levels except through the
    output arrays, so it runs as a level scan in the Fortran order (KERNEL_GUIDE §4), each level vectorised over
    (i, j).
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    maskW, maskS, rF, gravitySign = grid.maskW, grid.maskS, grid.rF, grid.gravitySign
    if not cfg.cpp.GM_BOLUS_ADVEC:                                  # :63
        return gm
    useK3dGM = (cfg.cpp.GM_READ_K3D_GM or cfg.cpp.GM_VISBECK_VARIABLE_K or cfg.cpp.GM_BATES_K3D
                or cfg.cpp.GM_GEOM_VARIABLE_K
                or cfg.cpp.ALLOW_GM_LEITH_QG)                       # :8-11  GM_USE_K3D_GM (M3 lane MLAdjust)
    # PTRACERS lane (tutorial_tracer_adjsens/code_ad): without GM_USE_K3D_GM the half_K*GM_bolFac2d form (#else arms
    # of :151-161, :172-182)
    for opt in ("GM_GEOM_VARIABLE_K", "ALLOW_EDDYPSI"):
        if getattr(cfg.cpp, opt):
            raise NotImplementedError(f"GMREDI_CALC_PSI_BOLUS: {opt} is not ported")

    GM_PsiX, GM_PsiY = gm.GM_PsiX, gm.GM_PsiY
    if not gm.GM_AdvForm:                                           # :87
        return gm

    def loc2d(name):
        # GO lane (P=N): the device's tile count (see gmredi_calc_tensor.loc2d)
        return grid.R_low.local(name)

    SlopeX, SlopeY = loc2d("SlopeX"), loc2d("SlopeY")               # :74-79 locals
    dSigmaDrW, dSigmaDrS = loc2d("dSigmaDrW"), loc2d("dSigmaDrS")
    taperX, taperY = loc2d("taperX"), loc2d("taperY")

    halfSign = halfRL*gravitySign                                   # :89

    # :91  DO k=2,Nr as a level scan (KERNEL_GUIDE §4; no level branches); `c` carries what the levels share
    def level_k(k, c):
        SlopeX, SlopeY, dSigmaDrW, dSigmaDrS, taperX, taperY, GM_PsiX, GM_PsiY = c
        km1 = k-1                                                   # :92
        if cfg.cpp.ALLOW_AUTODIFF:                                  # :94-103
            j = loop_j(1-OLy, sNy+OLy)
            i = loop_i(1-OLx, sNx+OLx)
            SlopeX = SlopeX.at[i, j].set(0.)                        # 0. _d 0
            SlopeY = SlopeY.at[i, j].set(0.)
            dSigmaDrW = dSigmaDrW.at[i, j].set(0.)
            dSigmaDrS = dSigmaDrS.at[i, j].set(0.)

        j = loop_j(1-OLy, sNy+OLy)                                  # :105-113  Gradient of Sigma below U and V points
        i = loop_i(1-OLx+1, sNx+OLx)
        SlopeX = SlopeX.at[i, j].set((sigmaX[i, j, km1]+sigmaX[i, j, k])*halfRL
                                     * maskW[i, j, km1]*maskW[i, j, k])
        dSigmaDrW = dSigmaDrW.at[i, j].set((sigmaR[i-1, j, k]+sigmaR[i, j, k])*halfSign
                                           * maskW[i, j, km1]*maskW[i, j, k])
        j = loop_j(1-OLy+1, sNy+OLy)                                # :114-121
        i = loop_i(1-OLx, sNx+OLx)
        SlopeY = SlopeY.at[i, j].set((sigmaY[i, j, km1]+sigmaY[i, j, k])*halfRL
                                     * maskS[i, j, km1]*maskS[i, j, k])
        dSigmaDrS = dSigmaDrS.at[i, j].set((sigmaR[i, j-1, k]+sigmaR[i, j, k])*halfSign
                                           * maskS[i, j, km1]*maskS[i, j, k])

        if params.usingZCoords:                                     # :123-128  rDepth for 'ldd97' tapering
            rDepth = rF[1] - rF[k]
        else:
            rDepth = rF[k] - rF[Nr+1]
        taperX, taperY, SlopeX, SlopeY, dSigmaDrW, dSigmaDrS = gmredi_slope_psi(   # :130-135
            taperX, taperY, SlopeX, SlopeY, dSigmaDrW, dSigmaDrS, ldd97_LrhoW, ldd97_LrhoS, rDepth, k,
            cfg=cfg, params=params, gm=gm)

        # :145-190  the 2 stream-function components (GM bolus vel.); SlopeX,Y are masked, PsiX,Y are not masked again
        # :147-148  half_K = GM_background_K*(GM_bolFac1d(km1)+GM_bolFac1d(k))*op25: read only without GM_USE_K3D_GM
        half_K = gm.GM_background_K \
            * (gm.GM_bolFac1d[km1]+gm.GM_bolFac1d[k])*op25
        j = loop_j(1-OLy, sNy+OLy)                                  # :149-169
        i = loop_i(1-OLx+1, sNx+OLx)
        if useK3dGM:
            GM_PsiX = GM_PsiX.at[i, j, k].set(SlopeX[i, j]*taperX[i, j]
                                              * (op25
                                                 * (locK3dGM[i-1, j, km1]+locK3dGM[i, j, km1]
                                                    + locK3dGM[i-1, j, k]+locK3dGM[i, j, k])
                                                 ))
        else:                                                       # :156-159 (PTRACERS lane)
            GM_PsiX = GM_PsiX.at[i, j, k].set(SlopeX[i, j]*taperX[i, j]
                                              * (half_K
                                                 * (gm.GM_bolFac2d[i-1, j]+gm.GM_bolFac2d[i, j])
                                                 ))
        j = loop_j(1-OLy+1, sNy+OLy)                                # :170-190
        i = loop_i(1-OLx, sNx+OLx)
        if useK3dGM:
            GM_PsiY = GM_PsiY.at[i, j, k].set(SlopeY[i, j]*taperY[i, j]
                                              * (op25
                                                 * (locK3dGM[i, j-1, km1]+locK3dGM[i, j, km1]
                                                    + locK3dGM[i, j-1, k]+locK3dGM[i, j, k])
                                                 ))
        else:                                                       # :177-180 (PTRACERS lane)
            GM_PsiY = GM_PsiY.at[i, j, k].set(SlopeY[i, j]*taperY[i, j]
                                              * (half_K
                                                 * (gm.GM_bolFac2d[i, j-1]+gm.GM_bolFac2d[i, j])
                                                 ))
        return SlopeX, SlopeY, dSigmaDrW, dSigmaDrS, taperX, taperY, GM_PsiX, GM_PsiY

    from mitjax.ops.scan_k import scan_levels
    c = scan_levels(level_k, (SlopeX, SlopeY, dSigmaDrW, dSigmaDrS, taperX, taperY, GM_PsiX, GM_PsiY), 2, Nr)
    SlopeX, SlopeY, dSigmaDrW, dSigmaDrS, taperX, taperY, GM_PsiX, GM_PsiY = c

    return gm.replace(GM_PsiX=GM_PsiX, GM_PsiY=GM_PsiY)
