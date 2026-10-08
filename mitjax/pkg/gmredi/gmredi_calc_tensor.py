"""GMREDI_CALC_TENSOR: pkg/gmredi/gmredi_calc_tensor.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.safe import safe_div
from mitjax.ops.scan_k import scan_levels
from mitjax.pkg.gmredi.gmredi_calc_psi_bolus import gmredi_calc_psi_bolus
from mitjax.pkg.gmredi.gmredi_h import op5, op25
from mitjax.pkg.gmredi.gmredi_slope_limit import gmredi_slope_limit


def gmredi_calc_tensor(iMin, iMax, jMin, jMax, sigmaX, sigmaY, sigmaR, myTime, myIter,
                       *, cfg, grid, params, gm, state, visc=None, kppf=None):
    """GMREDI_CALC_TENSOR( iMin, iMax, jMin, jMax, sigmaX, sigmaY, sigmaR, bi, bj, myTime, myIter, myThid )
    @63cdc0b pkg/gmredi/gmredi_calc_tensor.F:24-1129

    C     | SUBROUTINE GMREDI_CALC_TENSOR
    C     | o Calculate tensor elements for GM/Redi tensor.
    C     bi, bj    :: tile indices
    C     myTime    :: Current time in simulation
    C     myIter    :: Current iteration number in simulation

    sigmaX, sigmaY, sigmaR: FArrays (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr), tiled (the caller's per-tile locals).
    `grid`: GRID.h (maskC, maskW, maskS, R_low, rF, rC, gravitySign, kLowC); `params`: PARAMS.h (usingZCoords static;
    wUnit2rVel, rVel2wUnit, rUnit2z, z2rUnit); `gm`: GMREDI.h (gmredi_h.Gmredi); `state`: DYNVARS.h (hMixLayer).
    Returns `gm` with the tensor it writes: Kwx, Kwy, Kwz (:561-570, :650-685), Kux (:786-806), Kvy (:983-1003); under
    ALLOW_AUTODIFF every point of Kwx..Kvz is first set to zero (:263-279; Kuz, Kvz stay zero when GM_ExtraDiag is
    .FALSE.); under GM_BOLUS_ADVEC every point of GM_PsiX, GM_PsiY is set to zero (:223-232), then GM_AdvForm writes
    them through GMREDI_CALC_PSI_BOLUS; GM_ExtraDiag writes Kuz (:815-839), Kvz (:1012-1036). Points the Fortran does
    not write (the outer halo row/column of the tensor) keep their values in `gm`.

    Branches ported for the M1 experiments (docs/coverage/{global_ocean.90x40x15,tutorial_global_oce_optim}.md):
    GM_taper_scheme 'gkw91' / 'dm95' (GMREDI_SLOPE_LIMIT), GM_NON_UNITY_DIAGONAL and GM_EXTRA_DIAGONAL defined,
    GM_ExtraDiag = .FALSE., GM_AdvForm = .FALSE., ALLOW_AUTODIFF on (optim's code_ad) or off. GOADK lane (M2,
    global_ocean.90x40x15/code_ad, docs/coverage/global_ocean.90x40x15-code_ad.md): GM_READ_K3D_REDI and
    GM_READ_K3D_GM (GM_USE_K3D_*: the locals locK3dRedi/locK3dGM :233-259 and their forms of Kredi_tmp/Kgm_tmp
    :665-676, Kux :790-791, Kvy :987-988), GM_ExtraDiag = .TRUE. with both K3D fields (Kuz :815-839, Kvz
    :1012-1036; PTRACERS lane: the GM_isoFac / GM_bolFac forms without them, :822-829, :1019-1026), GM_AdvForm with
    GM_BOLUS_ADVEC (GMREDI_CALC_PSI_BOLUS, :609-634; GM_PsiX/Y returned). M3 lane MLAdjust (input.QGLthGM):
    ALLOW_GM_LEITH_QG (GM_USE_K3D_* with the #else initialisations :237-241, :251-255; the LeithQG block :334-360,
    GMREDI_CALC_QGLEITH with `visc` = MOM_VISC.h). Not ported
    (raise): GMREDI_CALC_PSI_BVP, GM_useBatesK3d with AdvForm,
    SUBMESO_CALC_PSI :637-645, GM_VISBECK_VARIABLE_K,
    GM_BATES_K3D, GM_GEOM_VARIABLE_K (the other K3d paths),
    GMREDI_MASK_SLOPES (:542-551), a build with only one of GM_NON_UNITY_DIAGONAL / GM_EXTRA_DIAGONAL. Lane M4LAB
    (lab_sea/code: GMREDI_OPTIONS.h defines neither; GM_taper_scheme 'ldd97'; useKPP): the ldd97/fm07 Lrho fields
    and kLow_W/S (:164-211; `Cspd/ABS(fCori)` guarded where fCori = 0, the IF's ELSE lanes; INTEGER MIN of kLowC),
    locMixLayer = KPPhbl of KPP.h (`kppf`, :290-297 C, :700-707 U, :896-903 V), the build without both diagonal
    options: the U and V loops (:696-1091) are not compiled and Kux = Kvy = GM_isopycK (:1093-1118, or the
    GM_USE_K3D_REDI average) on (2-OLx:sNx+OLx-1, 2-OLy:sNy+OLy-1), levels independent. Not ported
    (output only, they write no model variable): the DIAGNOSTICS_IS_ON / DIAGNOSTICS_FILL blocks (:143-148, :687-692,
    :849-887, :1046-1084) and GMREDI_DIAGNOSTICS_FILL (:1120-1123).

    Loops: the three k loops (:382-573 k=Nr..2, :731-891 and :929-1088 k=Nr..1) call GMREDI_SLOPE_LIMIT once per level
    and carry the locals (SlopeX, hTransLay, ...) from level to level, so they run as Python loops over k in the
    Fortran order (each level vectorised over i,j). The diffusivity loop :650-685 writes level k from level-k inputs
    only; it also runs as a Python loop over k (its scalars isopycK, bolus_K depend on k). The tile loop is implicit.
    Locals are NaN until the Fortran writes them (mitjax/model/grid.local); kLow_W, kLow_S are written only by the
    ldd97/fm07 branch and passed to GMREDI_SLOPE_LIMIT, which does not read them for the ported schemes. Fortran
    MAX (:727, :804, :924, :1001) is mitjax/ops/fortran_minmax.MAX (gfortran's value on +-0 ties and NaN); the
    integer MAX/MIN of level indices (km1, kp1) are Python's.
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    cpp = cfg.cpp
    maskC, maskW, maskS = grid.maskC, grid.maskW, grid.maskS
    R_low, rF, rC, gravitySign = grid.R_low, grid.rF, grid.rC, grid.gravitySign
    kLowC = grid.kLowC
    hMixLayer = state.hMixLayer
    GM_isoFac1d, GM_isoFac2d = gm.GM_isoFac1d, gm.GM_isoFac2d
    GM_bolFac1d, GM_bolFac2d = gm.GM_bolFac1d, gm.GM_bolFac2d
    scheme = gm.GM_taper_scheme.rstrip()

    # :11-19  GM_USE_K3D_REDI / GM_USE_K3D_GM
    for opt in ("GM_VISBECK_VARIABLE_K", "GM_BATES_K3D",
                "GM_GEOM_VARIABLE_K", "GMREDI_MASK_SLOPES"):
        if getattr(cpp, opt):
            raise NotImplementedError(f"GMREDI_CALC_TENSOR: {opt} is not ported")
    # :696 #if ( defined GM_NON_UNITY_DIAGONAL || defined GM_EXTRA_DIAGONAL ): the U and V loops; :1093
    # #ifndef GM_NON_UNITY_DIAGONAL: Kux = Kvy = GM_isopycK. Ported: both defined (M1-M3 builds) or neither (lane
    # M4LAB, lab_sea/code/GMREDI_OPTIONS.h); one without the other (both blocks) raises.
    diagLoops = bool(cpp.GM_NON_UNITY_DIAGONAL or cpp.GM_EXTRA_DIAGONAL)
    if diagLoops and not (cpp.GM_NON_UNITY_DIAGONAL and cpp.GM_EXTRA_DIAGONAL):
        raise NotImplementedError("GMREDI_CALC_TENSOR: a build with only one of GM_NON_UNITY_DIAGONAL / "
                                  "GM_EXTRA_DIAGONAL is not ported (:696, :1093-1118)")
    useKPP = bool(cpp.ALLOW_KPP and cfg.use_flag("useKPP"))         # :290-306 (lane M4LAB: KPPhbl of KPP.h)
    if useKPP and kppf is None:
        raise ValueError("GMREDI_CALC_TENSOR: useKPP needs the KPP.h fields `kppf` (KPPhbl)")
    # GOADK lane (M2, global_ocean.90x40x15/code_ad): GM_USE_K3D_REDI / GM_USE_K3D_GM from GM_READ_K3D_REDI / _GM
    # (:11-19; the other options that define them raise above), GM_ExtraDiag (Kuz, Kvz) and GM_AdvForm
    # M3 lane MLAdjust (input.QGLthGM): ALLOW_GM_LEITH_QG defines both (:11-19), with the #else arms of the locals'
    # initialisation (:237-241 GM_isopycK*GM_isoFac1d*GM_isoFac2d, :251-255 GM_background_K*GM_bolFac1d*GM_bolFac2d)
    K3dRedi = bool(cpp.GM_READ_K3D_REDI or cpp.ALLOW_GM_LEITH_QG)
    K3dGM = bool(cpp.GM_READ_K3D_GM or cpp.ALLOW_GM_LEITH_QG)
    # PTRACERS lane (tutorial_tracer_adjsens/code_ad): GM_ExtraDiag with the GM_isoFac / GM_bolFac forms of Kuz, Kvz
    # (the #else arms of GM_USE_K3D_REDI / GM_USE_K3D_GM, :821-829, :1018-1026)

    def loc2d(name, fill=float("nan")):
        # GO lane (P=N): the local takes the tile count of the arrays this program holds (the device's block under
        # shard_map), not SIZE.h's nSx*nSy: R_low is a 2-D GRID.h array with the same (i, j) declaration
        return grid.R_low.local(name, fill=fill)

    SlopeX, SlopeY = loc2d("SlopeX"), loc2d("SlopeY")               # :78-97 locals
    dSigmaDx, dSigmaDy, dSigmaDr = loc2d("dSigmaDx"), loc2d("dSigmaDy"), loc2d("dSigmaDr")
    SlopeSqr, taperFct = loc2d("SlopeSqr"), loc2d("taperFct")
    maskFk = loc2d("maskFk")
    ldd97_LrhoC, ldd97_LrhoW, ldd97_LrhoS = loc2d("ldd97_LrhoC"), loc2d("ldd97_LrhoW"), loc2d("ldd97_LrhoS")
    kLow_W = kLowC.local("kLow_W", fill=-999999)                    # :92-93 INTEGER (never written: see docstring)
    kLow_S = kLowC.local("kLow_S", fill=-999999)
    locMixLayer, baseSlope = loc2d("locMixLayer"), loc2d("baseSlope")
    hTransLay, recipLambda = loc2d("hTransLay"), loc2d("recipLambda")
    Kwx, Kwy, Kwz, Kux, Kvy = gm.Kwx, gm.Kwy, gm.Kwz, gm.Kux, gm.Kvy
    out = {}

    # :163-221  ldd97_Lrho (for tapering scheme ldd97)
    if scheme in ("ldd97", "fm07"):                                 # :164-211 (lane M4LAB: lab_sea 'ldd97')
        Cspd = 2.                                                   # :166  2. _d 0
        LrhoInf = 15.e3                                             # :167  15. _d 3
        LrhoSup = 100.e3                                            # :168  100. _d 3
        fCori = grid.fCori
        j = loop_j(1-OLy, sNy+OLy)                                  # :170-179  Tracer point location (center)
        i = loop_i(1-OLx, sNx+OLx)
        nz = fCori[i, j] != 0.                                      # :172  0. (REAL*4, exact)
        ldd97_LrhoC = ldd97_LrhoC.at[i, j].set(jnp.where(nz, safe_div(Cspd, jnp.abs(fCori[i, j]), nz), LrhoSup))
        ldd97_LrhoC = ldd97_LrhoC.at[i, j].set(MAX(LrhoInf, MIN(ldd97_LrhoC[i, j], LrhoSup, p="b"), p="b"))   # :177
        j = loop_j(1-OLy, sNy+OLy)                                  # :180-194  U point location (West)
        i1 = loop_i(1-OLx, 1-OLx)
        kLow_W = kLow_W.at[i1, j].set(0)                            # :182
        ldd97_LrhoW = ldd97_LrhoW.at[i1, j].set(LrhoSup)            # :183
        i = loop_i(1-OLx+1, sNx+OLx)
        kLow_W = kLow_W.at[i, j].set(jnp.minimum(kLowC[i-1, j], kLowC[i, j]))   # :185  MINMAX-RAW: INTEGER MIN
        fCoriLoc = op5*(fCori[i-1, j]+fCori[i, j])                  # :186
        nz = fCoriLoc != 0.                                         # :187  zeroRL
        ldd97_LrhoW = ldd97_LrhoW.at[i, j].set(jnp.where(nz, safe_div(Cspd, jnp.abs(fCoriLoc), nz), LrhoSup))
        ldd97_LrhoW = ldd97_LrhoW.at[i, j].set(MAX(LrhoInf, MIN(ldd97_LrhoW[i, j], LrhoSup, p="b"), p="b"))   # :192
        j1 = loop_j(1-OLy, 1-OLy)                                   # :196-199  V point location (South)
        i = loop_i(1-OLx+1, sNx+OLx)
        kLow_S = kLow_S.at[i, j1].set(0)                            # :197
        ldd97_LrhoS = ldd97_LrhoS.at[i, j1].set(LrhoSup)            # :198
        j = loop_j(1-OLy+1, sNy+OLy)                                # :200-211
        i = loop_i(1-OLx, sNx+OLx)
        kLow_S = kLow_S.at[i, j].set(jnp.minimum(kLowC[i, j-1], kLowC[i, j]))   # :202  MINMAX-RAW: INTEGER MIN
        fCoriLoc = op5*(fCori[i, j-1]+fCori[i, j])                  # :203
        nz = fCoriLoc != 0.                                         # :204  zeroRL
        ldd97_LrhoS = ldd97_LrhoS.at[i, j].set(jnp.where(nz, safe_div(Cspd, jnp.abs(fCoriLoc), nz), LrhoSup))
        ldd97_LrhoS = ldd97_LrhoS.at[i, j].set(MAX(LrhoInf, MIN(ldd97_LrhoS[i, j], LrhoSup, p="b"), p="b"))   # :209
    else:
        j = loop_j(1-OLy, sNy+OLy)                                  # :212-221  Just initialize to zero
        i = loop_i(1-OLx, sNx+OLx)
        ldd97_LrhoC = ldd97_LrhoC.at[i, j].set(0.)                  # 0. _d 0
        ldd97_LrhoW = ldd97_LrhoW.at[i, j].set(0.)
        ldd97_LrhoS = ldd97_LrhoS.at[i, j].set(0.)

    if cpp.GM_BOLUS_ADVEC:                                          # :223-232
        k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
        out["GM_PsiX"] = gm.GM_PsiX.at[i, j, k].set(0.)             # 0. _d 0
        out["GM_PsiY"] = gm.GM_PsiY.at[i, j, k].set(0.)
    # :261  locK3dGM(1) = 0. _d 0  (GM_USE_K3D_GM undefined; read only by GMREDI_CALC_PSI_BOLUS)
    locK3dRedi = locK3dGM = None
    if K3dRedi:                                                     # :233-246 (GOADK lane)  # ifdef GM_USE_K3D_REDI
        k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
        if cpp.GM_READ_K3D_REDI:                                    # # ifdef GM_READ_K3D_REDI
            locK3dRedi = gm.GM_inpK3dRedi.local("locK3dRedi").at[i, j, k].set(gm.GM_inpK3dRedi[i, j, k])
        else:                                                       # # else (M3 lane MLAdjust)
            locK3dRedi = gm.Kwx.local("locK3dRedi").at[i, j, k].set(
                gm.GM_isopycK*GM_isoFac1d[k]*GM_isoFac2d[i, j])
    if K3dGM:                                                       # :247-259 (GOADK lane)  # ifdef GM_USE_K3D_GM
        k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
        if cpp.GM_READ_K3D_GM:                                      # # ifdef GM_READ_K3D_GM
            locK3dGM = gm.GM_inpK3dGM.local("locK3dGM").at[i, j, k].set(gm.GM_inpK3dGM[i, j, k])
        else:                                                       # # else (M3 lane MLAdjust)
            locK3dGM = gm.Kwx.local("locK3dGM").at[i, j, k].set(
                gm.GM_background_K*GM_bolFac1d[k]*GM_bolFac2d[i, j])

    if cpp.ALLOW_AUTODIFF:                                          # :263-279
        k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
        Kwx = Kwx.at[i, j, k].set(0.)                               # 0. _d 0
        Kwy = Kwy.at[i, j, k].set(0.)
        Kwz = Kwz.at[i, j, k].set(0.)
        Kux = Kux.at[i, j, k].set(0.)
        Kvy = Kvy.at[i, j, k].set(0.)
        out["Kuz"] = gm.Kuz.at[i, j, k].set(0.)                     # # ifdef GM_EXTRA_DIAGONAL
        out["Kvz"] = gm.Kvz.at[i, j, k].set(0.)

    j = loop_j(1-OLy, sNy+OLy)                                      # :281-289  Initialise Mixed Layer related array
    i = loop_i(1-OLx, sNx+OLx)
    hTransLay = hTransLay.at[i, j].set(R_low[i, j])
    baseSlope = baseSlope.at[i, j].set(0.)                          # 0. _d 0
    recipLambda = recipLambda.at[i, j].set(0.)
    locMixLayer = locMixLayer.at[i, j].set(0.)
    if useKPP:                                                      # :290-297  #ifdef ALLOW_KPP IF ( useKPP )
        locMixLayer = locMixLayer.at[i, j].set(kppf["KPPhbl"][i, j])
    else:                                                           # :299-306  ELSE / IF ( .TRUE. ) THEN
        locMixLayer = locMixLayer.at[i, j].set(hMixLayer[i, j])

    if cpp.ALLOW_GM_LEITH_QG:                                       # :334-360 (M3 lane MLAdjust)
        GM_LeithQG_K = gm.GM_LeithQG_K
        if cpp.ALLOW_AUTODIFF:                                      # :335-343
            k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
            GM_LeithQG_K = GM_LeithQG_K.at[i, j, k].set(0.)         # 0. _d 0
        if gm.GM_useLeithQG:                                        # :344-359
            from mitjax.pkg.gmredi.gmredi_calc_qgleith import gmredi_calc_qgleith
            if visc is None:
                raise ValueError("GMREDI_CALC_TENSOR: GM_useLeithQG needs `visc` (MOM_VISC.h of MOM_INIT_FIXED)")
            GM_LeithQG_K = gmredi_calc_qgleith(GM_LeithQG_K, myTime, myIter, cfg=cfg, grid=grid,   # :346-348
                                               params=params, state=state, visc=visc)
            k, j, i = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))   # :349-358
            locK3dRedi = locK3dRedi.at[i, j, k].set(locK3dRedi[i, j, k]
                                                    + GM_LeithQG_K[i, j, k]*gm.GM_isoFac_calcK)
            locK3dGM = locK3dGM.at[i, j, k].set(locK3dGM[i, j, k]
                                                + GM_LeithQG_K[i, j, k])
        out["GM_LeithQG_K"] = GM_LeithQG_K

    # :380-573  1rst loop on k : compute Tensor Coeff. at W points.
    # :380-573 DO k=Nr,2,-1 as a level scan (KERNEL_GUIDE §4; no level branches)  [for k in range(Nr, 1, -1):]
    def w_level(k, c):
        SlopeX, SlopeY, dSigmaDx, dSigmaDy, dSigmaDr, SlopeSqr, taperFct, maskFk, hTransLay, baseSlope, recipLambda, Kwx, Kwy, Kwz = c
        j = loop_j(1-OLy, sNy+OLy)                                  # :384-397
        i = loop_i(1-OLx, sNx+OLx)
        if cpp.ALLOW_AUTODIFF:
            SlopeX = SlopeX.at[i, j].set(0.)                        # 0. _d 0
            SlopeY = SlopeY.at[i, j].set(0.)
            dSigmaDx = dSigmaDx.at[i, j].set(0.)
            dSigmaDy = dSigmaDy.at[i, j].set(0.)
            dSigmaDr = dSigmaDr.at[i, j].set(0.)
            SlopeSqr = SlopeSqr.at[i, j].set(0.)
            taperFct = taperFct.at[i, j].set(0.)
        maskFk = maskFk.at[i, j].set(maskC[i, j, k-1]*maskC[i, j, k])

        j = loop_j(1-OLy+1, sNy+OLy-1)                              # :399-410  Gradient of Sigma at rVel points
        i = loop_i(1-OLx+1, sNx+OLx-1)
        dSigmaDx = dSigmaDx.at[i, j].set(op25*(sigmaX[i+1, j, k-1]+sigmaX[i, j, k-1]
                                               + sigmaX[i+1, j, k]+sigmaX[i, j, k]
                                               )*maskFk[i, j])
        dSigmaDy = dSigmaDy.at[i, j].set(op25*(sigmaY[i, j+1, k-1]+sigmaY[i, j, k-1]
                                               + sigmaY[i, j+1, k]+sigmaY[i, j, k]
                                               )*maskFk[i, j])

        j = loop_j(1-OLy, sNy+OLy)                                  # :505-511  Z-Coords: change sign of vertical
        i = loop_i(1-OLx, sNx+OLx)                                  #   Sigma gradient (always downward)
        dSigmaDr = dSigmaDr.at[i, j].set(gravitySign*sigmaR[i, j, k])

        if params.usingZCoords:                                     # :525-530  rDepth for 'ldd97' tapering
            rDepth = rF[1] - rF[k]
        else:
            rDepth = rF[k] - rF[Nr+1]
        (SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda,
         dSigmaDr) = gmredi_slope_limit(                            # :531-540
            SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, dSigmaDr,
            dSigmaDx, dSigmaDy, ldd97_LrhoC, locMixLayer, rDepth, rF, kLowC,
            3, k, myTime, myIter, cfg=cfg, params=params, gm=gm)

        j = loop_j(1-OLy+1, sNy+OLy-1)                              # :561-570  Components of Redi/GM tensor,
        i = loop_i(1-OLx+1, sNx+OLx-1)                              #   first step with just the slope
        Kwx = Kwx.at[i, j, k].set(-gravitySign*SlopeX[i, j]*taperFct[i, j])
        Kwy = Kwy.at[i, j, k].set(-gravitySign*SlopeY[i, j]*taperFct[i, j])
        Kwz = Kwz.at[i, j, k].set(SlopeSqr[i, j]*taperFct[i, j])
        return (SlopeX, SlopeY, dSigmaDx, dSigmaDy, dSigmaDr, SlopeSqr, taperFct, maskFk, hTransLay, baseSlope, recipLambda, Kwx, Kwy, Kwz)

    c = (SlopeX, SlopeY, dSigmaDx, dSigmaDy, dSigmaDr, SlopeSqr, taperFct, maskFk, hTransLay, baseSlope, recipLambda, Kwx, Kwy, Kwz)
    c = scan_levels(w_level, c, 2, Nr, down=True)
    SlopeX, SlopeY, dSigmaDx, dSigmaDy, dSigmaDr, SlopeSqr, taperFct, maskFk, hTransLay, baseSlope, recipLambda, Kwx, Kwy, Kwz = c

    if cpp.GM_BOLUS_ADVEC and gm.GM_AdvForm:                        # :609-634 (GOADK lane)
        if cpp.GM_BOLUS_BVP and gm.GM_useBVP:
            raise NotImplementedError("GMREDI_CALC_TENSOR: GMREDI_CALC_PSI_BVP (GM_UseBVP) is not ported")
        if not cpp.GM_BATES_PASSIVE and gm.GM_useBatesK3d:          # :619-629
            raise NotImplementedError("GMREDI_CALC_TENSOR: GM_useBatesK3d (PsiX/Y from GMREDI_CALC_BATES_K) is not "
                                      "ported")
        gmB = gmredi_calc_psi_bolus(                                # :623-627
            iMin, iMax, jMin, jMax, sigmaX, sigmaY, sigmaR, locK3dGM, ldd97_LrhoW, ldd97_LrhoS,
            cfg=cfg, grid=grid, params=params,
            gm=gm.replace(GM_PsiX=out["GM_PsiX"], GM_PsiY=out["GM_PsiY"]))
        out["GM_PsiX"], out["GM_PsiY"] = gmB.GM_PsiX, gmB.GM_PsiY
    if not cpp.GM_EXCLUDE_SUBMESO and gm.GM_useSubMeso and gm.GM_AdvForm:          # :637-645
        raise NotImplementedError("GMREDI_CALC_TENSOR: SUBMESO_CALC_PSI is not ported")

    # :648-685  Update components of Redi/GM tensor, second step: multiply by Diffusivity
    # :648-685 DO k=1,Nr: k = 1 static, k = 2..Nr a level scan (KERNEL_GUIDE §4)  [for k in range(1, Nr + 1):]
    def k_level(k, c):
        Kwx, Kwy, Kwz = c
        km1 = max(k-1, 1)  # MINMAX-INT: integer (no tie or NaN case)
        isopycK = (gm.GM_isopycK
                   * (GM_isoFac1d[km1]+GM_isoFac1d[k])*op5)
        bolus_K = (gm.GM_background_K
                   * (GM_bolFac1d[km1]+GM_bolFac1d[k])*op5)
        j = loop_j(1-OLy+1, sNy+OLy-1)
        i = loop_i(1-OLx+1, sNx+OLx-1)
        if K3dRedi:                                                 # :665-670  # ifdef GM_USE_K3D_REDI (GOADK)
            Kredi_tmp = op5*(locK3dRedi[i, j, km1] + locK3dRedi[i, j, k])
        else:
            Kredi_tmp = isopycK*GM_isoFac2d[i, j]
        if K3dGM:                                                   # :671-676  # ifdef GM_USE_K3D_GM (GOADK)
            Kgm_tmp = Kredi_tmp + gm.GM_skewflx*op5*(locK3dGM[i, j, km1]+locK3dGM[i, j, k])
        else:
            Kgm_tmp = Kredi_tmp + gm.GM_skewflx*bolus_K*GM_bolFac2d[i, j]
        Kwx = Kwx.at[i, j, k].set(Kgm_tmp*Kwx[i, j, k])
        Kwy = Kwy.at[i, j, k].set(Kgm_tmp*Kwy[i, j, k])
        Kwz = Kwz.at[i, j, k].set(Kredi_tmp*Kwz[i, j, k])
        return (Kwx, Kwy, Kwz)

    c = (Kwx, Kwy, Kwz)
    c = k_level(1, c)                             # km1 = MAX(k-1,1): k = 1 static
    if Nr >= 2:
        c = scan_levels(k_level, c, 2, Nr)
    Kwx, Kwy, Kwz = c

    if diagLoops:                                                   # :696  #if ( GM_NON_UNITY_DIAGONAL ||
        #                                                               GM_EXTRA_DIAGONAL )
        # :696-891  2nd k loop : compute Tensor Coeff. at U point
        j = loop_j(1-OLy, sNy+OLy)                                  # :700-719
        i = loop_i(2-OLx, sNx+OLx)
        if useKPP:                                                  # :700-707  # ifdef ALLOW_KPP IF ( useKPP )
            locMixLayer = locMixLayer.at[i, j].set((kppf["KPPhbl"][i-1, j]
                                                    + kppf["KPPhbl"][i, j])*op5)
        else:                                                       # :711-719  ELSE / IF ( .TRUE. ) THEN
            locMixLayer = locMixLayer.at[i, j].set((hMixLayer[i-1, j]
                                                    + hMixLayer[i, j])*op5)
        j = loop_j(1-OLy, sNy+OLy)                                      # :720-729
        i = loop_i(1-OLx, sNx+OLx)
        hTransLay = hTransLay.at[i, j].set(0.)                          # 0.
        baseSlope = baseSlope.at[i, j].set(0.)
        recipLambda = recipLambda.at[i, j].set(0.)
        i = loop_i(2-OLx, sNx+OLx)
        hTransLay = hTransLay.at[i, j].set(MAX(R_low[i-1, j], R_low[i, j], p="b"))  # :727

        # :731-891 DO k=Nr,1,-1: k = Nr, Nr-1 static, k = Nr-2..1 a level scan (KERNEL_GUIDE §4)  [for k in range(Nr, 0, -1):                                      # :731]
        def u_level(k, c):
            dSigmaDx, dSigmaDy, dSigmaDr, SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, Kux, out = c
            kp1 = min(Nr, k+1)  # MINMAX-INT: integer (no tie or NaN case)
            maskp1 = 1.                                                 # 1. _d 0
            if k >= Nr:
                maskp1 = 0.                                             # 0. _d 0

            j = loop_j(1-OLy+1, sNy+OLy-1)                              # :739-751  Gradient of Sigma at U points
            i = loop_i(1-OLx+1, sNx+OLx-1)
            dSigmaDx = dSigmaDx.at[i, j].set(sigmaX[i, j, k]
                                             * maskW[i, j, k])
            dSigmaDy = dSigmaDy.at[i, j].set(op25*(sigmaY[i-1, j+1, k]+sigmaY[i, j+1, k]
                                                   + sigmaY[i-1, j, k]+sigmaY[i, j, k]
                                                   )*maskW[i, j, k])
            dSigmaDr = dSigmaDr.at[i, j].set(op25*(sigmaR[i-1, j, k]+sigmaR[i, j, k]
                                                   + (sigmaR[i-1, j, kp1]+sigmaR[i, j, kp1])*maskp1
                                                   )*maskW[i, j, k]*gravitySign)

            if params.usingZCoords:                                     # :764-769
                rDepth = rF[1] - rC[k]
            else:
                rDepth = rC[k] - rF[Nr+1]
            (SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda,
             dSigmaDr) = gmredi_slope_limit(                            # :770-779
                SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, dSigmaDr,
                dSigmaDx, dSigmaDy, ldd97_LrhoW, locMixLayer, rDepth, rC, kLow_W,
                1, k, myTime, myIter, cfg=cfg, params=params, gm=gm)

            j = loop_j(1-OLy+1, sNy+OLy-1)                              # :785-807  # ifdef GM_NON_UNITY_DIAGONAL
            i = loop_i(1-OLx+1, sNx+OLx-1)
            if K3dRedi:                                                 # :790-791  #  ifdef GM_USE_K3D_REDI (GOADK)
                Kux = Kux.at[i, j, k].set(
                    op5*(locK3dRedi[i-1, j, k] + locK3dRedi[i, j, k])
                    * taperFct[i, j])
            else:
                Kux = Kux.at[i, j, k].set(
                    gm.GM_isopycK*GM_isoFac1d[k]
                    * op5*(GM_isoFac2d[i-1, j]+GM_isoFac2d[i, j])
                    * taperFct[i, j])
            Kux = Kux.at[i, j, k].set(MAX(Kux[i, j, k], gm.GM_Kmin_horiz, p="a"))  # :804
            if gm.GM_ExtraDiag:                                         # :815-846 (GOADK lane, K3D forms)
                if K3dRedi:                                             # :820
                    redi = op5*(locK3dRedi[i-1, j, k] + locK3dRedi[i, j, k])
                else:                                                   # :822-823 (PTRACERS lane)
                    redi = gm.GM_isopycK*GM_isoFac1d[k] \
                        * op5*(GM_isoFac2d[i-1, j]+GM_isoFac2d[i, j])
                if K3dGM:                                               # :826
                    bol = gm.GM_skewflx*op5*(locK3dGM[i-1, j, k] + locK3dGM[i, j, k])
                else:                                                   # :828-829 (PTRACERS lane)
                    bol = gm.GM_skewflx*gm.GM_background_K*GM_bolFac1d[k] \
                        * op5*(GM_bolFac2d[i-1, j]+GM_bolFac2d[i, j])
                out["Kuz"] = out.get("Kuz", gm.Kuz).at[i, j, k].set(
                    -gravitySign
                    * (redi
                       - bol
                       )*SlopeX[i, j]*taperFct[i, j])
            return (dSigmaDx, dSigmaDy, dSigmaDr, SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, Kux, out)

        c = (dSigmaDx, dSigmaDy, dSigmaDr, SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, Kux, out)
        c = u_level(Nr, c)                            # kp1 = MIN(Nr,k+1), k >= Nr: k = Nr, Nr-1
        if Nr >= 2:                                   #   static
            c = u_level(Nr-1, c)
        if Nr >= 3:
            c = scan_levels(u_level, c, 1, Nr-2, down=True)
        dSigmaDx, dSigmaDy, dSigmaDr, SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, Kux, out = c

        # :893-1088  3rd k loop : compute Tensor Coeff. at V point
        j = loop_j(2-OLy, sNy+OLy)                                  # :896-914
        i = loop_i(1-OLx, sNx+OLx)
        if useKPP:                                                  # :896-903  # ifdef ALLOW_KPP IF ( useKPP )
            locMixLayer = locMixLayer.at[i, j].set((kppf["KPPhbl"][i, j-1]
                                                    + kppf["KPPhbl"][i, j])*op5)
        else:                                                       # :906-914  ELSE / IF ( .TRUE. ) THEN
            locMixLayer = locMixLayer.at[i, j].set((hMixLayer[i, j-1]
                                                    + hMixLayer[i, j])*op5)
        j = loop_j(1-OLy, sNy+OLy)                                      # :915-921
        i = loop_i(1-OLx, sNx+OLx)
        hTransLay = hTransLay.at[i, j].set(0.)                          # 0.
        baseSlope = baseSlope.at[i, j].set(0.)
        recipLambda = recipLambda.at[i, j].set(0.)
        j = loop_j(2-OLy, sNy+OLy)                                      # :922-926
        i = loop_i(1-OLx, sNx+OLx)
        hTransLay = hTransLay.at[i, j].set(MAX(R_low[i, j-1], R_low[i, j], p="b"))  # :924

        # :928-1088 DO k=Nr,1,-1: k = Nr, Nr-1 static, k = Nr-2..1 a level scan (KERNEL_GUIDE §4)  [for k in range(Nr, 0, -1):                                      # :928-929  Gradient of Sigma at V points]
        def v_level(k, c):
            dSigmaDx, dSigmaDy, dSigmaDr, SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, Kvy, out = c
            kp1 = min(Nr, k+1)  # MINMAX-INT: integer (no tie or NaN case)
            maskp1 = 1.                                                 # 1. _d 0
            if k >= Nr:
                maskp1 = 0.                                             # 0. _d 0

            j = loop_j(1-OLy+1, sNy+OLy-1)                              # :937-948
            i = loop_i(1-OLx+1, sNx+OLx-1)
            dSigmaDx = dSigmaDx.at[i, j].set(op25*(sigmaX[i, j, k] + sigmaX[i+1, j, k]
                                                   + sigmaX[i, j-1, k] + sigmaX[i+1, j-1, k]
                                                   )*maskS[i, j, k])
            dSigmaDy = dSigmaDy.at[i, j].set(sigmaY[i, j, k]
                                             * maskS[i, j, k])
            dSigmaDr = dSigmaDr.at[i, j].set(op25*(sigmaR[i, j-1, k]+sigmaR[i, j, k]
                                                   + (sigmaR[i, j-1, kp1]+sigmaR[i, j, kp1])*maskp1
                                                   )*maskS[i, j, k]*gravitySign)

            if params.usingZCoords:                                     # :961-966
                rDepth = rF[1] - rC[k]
            else:
                rDepth = rC[k] - rF[Nr+1]
            (SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda,
             dSigmaDr) = gmredi_slope_limit(                            # :967-976
                SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, dSigmaDr,
                dSigmaDx, dSigmaDy, ldd97_LrhoS, locMixLayer, rDepth, rC, kLow_S,
                2, k, myTime, myIter, cfg=cfg, params=params, gm=gm)

            j = loop_j(1-OLy+1, sNy+OLy-1)                              # :982-1004  # ifdef GM_NON_UNITY_DIAGONAL
            i = loop_i(1-OLx+1, sNx+OLx-1)
            if K3dRedi:                                                 # :987-988  #  ifdef GM_USE_K3D_REDI (GOADK)
                Kvy = Kvy.at[i, j, k].set(
                    op5*(locK3dRedi[i, j-1, k] + locK3dRedi[i, j, k])
                    * taperFct[i, j])
            else:
                Kvy = Kvy.at[i, j, k].set(
                    gm.GM_isopycK*GM_isoFac1d[k]
                    * op5*(GM_isoFac2d[i, j-1]+GM_isoFac2d[i, j])
                    * taperFct[i, j])
            Kvy = Kvy.at[i, j, k].set(MAX(Kvy[i, j, k], gm.GM_Kmin_horiz, p="a"))  # :1001
            if gm.GM_ExtraDiag:                                         # :1012-1043 (GOADK lane, K3D forms)
                if K3dRedi:                                             # :1017
                    redi = op5*(locK3dRedi[i, j-1, k] + locK3dRedi[i, j, k])
                else:                                                   # :1019-1020 (PTRACERS lane)
                    redi = gm.GM_isopycK*GM_isoFac1d[k] \
                        * op5*(GM_isoFac2d[i, j-1]+GM_isoFac2d[i, j])
                if K3dGM:                                               # :1023
                    bol = gm.GM_skewflx*op5*(locK3dGM[i, j-1, k] + locK3dGM[i, j, k])
                else:                                                   # :1025-1026 (PTRACERS lane)
                    bol = gm.GM_skewflx*gm.GM_background_K*GM_bolFac1d[k] \
                        * op5*(GM_bolFac2d[i, j-1]+GM_bolFac2d[i, j])
                out["Kvz"] = out.get("Kvz", gm.Kvz).at[i, j, k].set(
                    -gravitySign
                    * (redi
                       - bol
                       )*SlopeY[i, j]*taperFct[i, j])
            return (dSigmaDx, dSigmaDy, dSigmaDr, SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, Kvy, out)

        c = (dSigmaDx, dSigmaDy, dSigmaDr, SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, Kvy, out)
        c = v_level(Nr, c)                            # kp1 = MIN(Nr,k+1), k >= Nr: k = Nr, Nr-1
        if Nr >= 2:                                   #   static
            c = v_level(Nr-1, c)
        if Nr >= 3:
            c = scan_levels(v_level, c, 1, Nr-2, down=True)
        dSigmaDx, dSigmaDy, dSigmaDr, SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, Kvy, out = c

    if not cpp.GM_NON_UNITY_DIAGONAL:                               # :1093-1118 (lane M4LAB)
        # keep a simplified setting here (used to be inside gmredi_x/ytransport)
        k, j, i = loops_kji((1, Nr), (1-OLy+1, sNy+OLy-1), (1-OLx+1, sNx+OLx-1))   # levels independent
        if K3dRedi:                                                 # # ifdef GM_USE_K3D_REDI
            Kux = Kux.at[i, j, k].set(op5*(locK3dRedi[i-1, j, k] + locK3dRedi[i, j, k]))   # :1099-1101
            Kvy = Kvy.at[i, j, k].set(op5*(locK3dRedi[i, j-1, k] + locK3dRedi[i, j, k]))   # :1109-1111
        else:                                                       # # else
            Kux = Kux.at[i, j, k].set(jnp.broadcast_to(gm.GM_isopycK, Kux[i, j, k].shape))   # :1099, :1103
            Kvy = Kvy.at[i, j, k].set(jnp.broadcast_to(gm.GM_isopycK, Kvy[i, j, k].shape))   # :1109, :1113

    return gm.replace(Kwx=Kwx, Kwy=Kwy, Kwz=Kwz, Kux=Kux, Kvy=Kvy, **out)
