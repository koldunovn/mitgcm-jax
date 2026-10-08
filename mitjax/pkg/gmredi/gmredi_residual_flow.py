"""GMREDI_RESIDUAL_FLOW: pkg/gmredi/gmredi_residual_flow.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def gmredi_residual_flow(uFld, vFld, wFld, myIter, *, cfg, gm, grid=None):
    """GMREDI_RESIDUAL_FLOW( uFld, vFld, wFld, bi, bj, myIter, myThid )
    @63cdc0b pkg/gmredi/gmredi_residual_flow.F:6-144

    C     Add GM-bolus velocity to Eulerian velocity to get Residual Mean velocity.
    C     uFld   :: zonal      velocity (updated)
    C     vFld   :: meridional velocity (updated)
    C     wFld   :: vertical volume transport (updated)

    Returns (uFld, vFld, wFld). With GM_BOLUS_ADVEC (global_ocean.90x40x15) the bolus velocity is added only
    IF ( GM_AdvForm .AND. .NOT.GM_AdvSeparate .AND. .NOT.GM_InMomAsStress ) (:58-139); without it
    (tutorial_global_oce_optim) the routine is empty. In both M1 runs GM_AdvForm is .FALSE.: the fields are returned
    unchanged, as the Fortran leaves them. Not ported (raise): ALLOW_EDDYPSI (:49-51, :100-130).

    GOADK lane (M2, global_ocean.90x40x15/code_ad, GM_AdvForm = .TRUE.): the bolus velocity of :58-97, with
    `grid` = GRID.h (gravitySign, deepFacF, recip_drF, recip_hFacW, recip_hFacS, recip_deepFacC, dyG, dxG, recip_rA,
    recip_deepFacF; the macros _recip_hFacW/S are the arrays: no ALLOW_RECIP_HFACS_MACROS). The k loop writes level
    k from level k and kp1 of GM_PsiX/Y only (inputs), so its iterations are independent; it runs as a level scan
    in the Fortran order (KERNEL_GUIDE §4), each level vectorised over (i, j). `maskp1 = 1.` / `0.` are REAL*4 literals, exact.
    """
    if cfg.cpp.ALLOW_EDDYPSI:
        raise NotImplementedError("GMREDI_RESIDUAL_FLOW: ALLOW_EDDYPSI is not ported")
    if cfg.cpp.GM_BOLUS_ADVEC:                                      # :40
        if gm.GM_AdvForm and not gm.GM_AdvSeparate and not gm.GM_InMomAsStress:   # :58-59 (GOADK lane)
            sz = cfg.size
            sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
            GM_PsiX, GM_PsiY = gm.GM_PsiX, gm.GM_PsiY
            deepFacF, recip_drF, recip_deepFacC = grid.deepFacF, grid.recip_drF, grid.recip_deepFacC
            flipSign4LHCoord = -grid.gravitySign                    # :56
            # :61  DO k=1,Nr as a level scan (KERNEL_GUIDE §4): k = Nr-1, Nr static (MIN(k+1,Nr), k.GE.Nr)
            def level_k(k, c):
                uFld, vFld, wFld = c
                kp1 = min(k+1, Nr)  # MINMAX-INT: integer level index (no tie or NaN case)
                maskp1 = 1.                                         # :63  1.
                if k >= Nr:
                    maskp1 = 0.                                     # :64  0.
                j = loop_j(1-OLy, sNy+OLy)                          # :66-73
                i = loop_i(1-OLx, sNx+OLx)
                delPsi = (GM_PsiX[i, j, kp1]*deepFacF[kp1]*maskp1
                          - GM_PsiX[i, j, k]*deepFacF[k])
                uFld = uFld.at[i, j, k].set(uFld[i, j, k]
                                            + delPsi*recip_drF[k]*grid.recip_hFacW[i, j, k]
                                            * recip_deepFacC[k]*flipSign4LHCoord)
                delPsi = (GM_PsiY[i, j, kp1]*deepFacF[kp1]*maskp1   # :74-82
                          - GM_PsiY[i, j, k]*deepFacF[k])
                vFld = vFld.at[i, j, k].set(vFld[i, j, k]
                                            + delPsi*recip_drF[k]*grid.recip_hFacS[i, j, k]
                                            * recip_deepFacC[k]*flipSign4LHCoord)
                j = loop_j(1-OLy, sNy+OLy-1)                        # :83-95  deep-model: simplify to 1/FacF
                i = loop_i(1-OLx, sNx+OLx-1)
                delPsi = (grid.dyG[i+1, j]*GM_PsiX[i+1, j, k]
                          - grid.dyG[i, j]*GM_PsiX[i, j, k]
                          + grid.dxG[i, j+1]*GM_PsiY[i, j+1, k]
                          - grid.dxG[i, j]*GM_PsiY[i, j, k]
                          )
                wFld = wFld.at[i, j, k].set(wFld[i, j, k]
                                            + delPsi*grid.recip_rA[i, j]
                                            * grid.recip_deepFacF[k]*flipSign4LHCoord)
                return uFld, vFld, wFld

            from mitjax.ops.scan_k import scan_levels
            uFld, vFld, wFld = scan_levels(level_k, (uFld, vFld, wFld), 1, Nr, peel=(0, 2))
    return uFld, vFld, wFld
