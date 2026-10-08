"""pkg/mom_common/mom_v_botdrag_coeff.F: bottom-drag coefficient for V (MOM_V_BOTDRAG_COEFF)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import halfRL
from mitjax.ops.safe import safe_div, safe_sqrt

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_v_botdrag_coeff.F:1


def mom_v_botdrag_coeff(k, inp_KE, uFld, vFld, kappaRV, KE, cDrag, myIter, *, cfg, grid, params, ctrlf=None):
    """MOM_V_BOTDRAG_COEFF(bi, bj, k, inp_KE, uFld, vFld, kappaRV, KE, cDrag, myIter, myThid)
    @63cdc0b pkg/mom_common/mom_v_botdrag_coeff.F:6-235

    C Compute bottom-drag coefficient (Cd) for V component momentum,
    C   such that bottom stress: taux_{bot} = -Cd * V_{bot} * rUnit2mass ;
    C include linear and quadratic bottom drag and friction (no-slip BC) at bottom
    C  k              :: vertical level to process
    C  inp_KE         :: =T : KE is provided as input ; =F : to compute here
    C  uFld           :: velocity, zonal component
    C  vFld           :: velocity, meridional component
    C  kappaRV        :: vertical viscosity
    C  KE             :: Kinetic energy (input when inp_KE = T)
    C  myIter         :: current iteration number
    C  KE             :: Kinetic energy (output when inp_KE = F)
    C  cDrag          :: bottom drag coefficient

    Returns (KE, cDrag). Ported as MOM_U_BOTDRAG_COEFF (same branches, same notes), with the V-point stencils and
    loop bounds of this file.
    """
    # GOADK lane: the routine sees CTRL_OPTIONS.h #ifdef ALLOW_CTRL (mom_v_botdrag_coeff.F:2-4), so
    # ALLOW_BOTTOMDRAG_CONTROL is read in that view (in the MOM_COMMON_OPTIONS.h view it is never defined)
    bdragCtrl = bool(cfg.cpp.ALLOW_CTRL) and cfg.cpp.flag("ALLOW_BOTTOMDRAG_CONTROL", "CTRL_OPTIONS.h")
    if bdragCtrl and ctrlf is None:
        raise ValueError("MOM_V_BOTDRAG_COEFF: ALLOW_BOTTOMDRAG_CONTROL needs ctrlf (CTRL_FIELDS.h: bottomDragFld)")
    if cfg.cpp.flag("ALLOW_BOTTOMDRAG_ROUGHNESS", _OPT):
        raise NotImplementedError("MOM_V_BOTDRAG_COEFF: ALLOW_BOTTOMDRAG_ROUGHNESS is not ported")
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    hFacW, hFacS, recip_hFacS, recip_hFacC = grid.hFacW, grid.hFacS, grid.recip_hFacS, grid.recip_hFacC
    recip_drF, recip_drC, maskS = grid.recip_drF, grid.recip_drC, grid.maskS
    bottomDragLinear, bottomDragQuadratic = params.bottomDragLinear, params.bottomDragQuadratic

#-  No-slip BCs impose a drag at bottom
    viscFac = 0.                                                    # :67-68
    if params.no_slip_bottom:
        viscFac = 2.

    if params.usingZCoords:                                         # :72-78
        kBottom = Nr
        kDown = min(k+1, Nr)  # MINMAX-INT: integer (no tie or NaN case)
        kLowF = k+1
        dragFac = 1.
    else:                                                           # :79-85
        raise NotImplementedError("MOM_V_BOTDRAG_COEFF: usingZCoords = .FALSE. (mass2rUnit*rhoConst) is not ported")
    if k == kBottom:                                                # :86-90
        recDrC = recip_drF[k]
    else:
        recDrC = recip_drC[kLowF]

    if cfg.cpp.flag("ALLOW_AUTODIFF", _OPT):                        # :92-99
        j = loop_j(1-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx)
        cDrag = cDrag.at[i, j].set(0.)

#--   Linear bottom drag contribution to cDrag:
    j = loop_j(1-OLy+1, sNy+OLy)                                    # :102-110
    i = loop_i(1-OLx, sNx+OLx)
    if bdragCtrl:                                                   # (GOADK lane) #ifdef ALLOW_BOTTOMDRAG_CONTROL
        cDrag = cDrag.at[i, j].set(bottomDragLinear*dragFac
                                   + halfRL*(ctrlf.bottomDragFld[i, j-1]
                                             + ctrlf.bottomDragFld[i, j])*dragFac)
    else:
        cDrag = cDrag.at[i, j].set(bottomDragLinear*dragFac)

#--   Add friction at the bottom (no-slip BC) to cDrag:
    j = loop_j(1-OLy+1, sNy+OLy-1)
    i = loop_i(1-OLx, sNx+OLx-1)
    if params.no_slip_bottom and params.bottomVisc_pCell:           # :113-121
        cDrag = cDrag.at[i, j].set(cDrag[i, j]
                                   + kappaRV[i, j, kLowF]*recDrC*viscFac
                                   * recip_hFacS[i, j, k])
    elif params.no_slip_bottom:                                     # :122-130
        cDrag = cDrag.at[i, j].set(cDrag[i, j]
                                   + kappaRV[i, j, kLowF]*recDrC*viscFac)

#--   Add quadratic bottom drag to cDrag:
    sel = params.selectBotDragQuadr
    if sel == 0:                                                    # :133-160
        if not inp_KE:                                              # :134-145
            jK = loop_j(1-OLy, sNy+OLy-1)
            iK = loop_i(1-OLx, sNx+OLx-1)
            KE = KE.at[iK, jK].set(0.25*(
                (uFld[iK, jK]*uFld[iK, jK]*hFacW[iK, jK, k]
                 + uFld[iK+1, jK]*uFld[iK+1, jK]*hFacW[iK+1, jK, k])
                + (vFld[iK, jK]*vFld[iK, jK]*hFacS[iK, jK, k]
                   + vFld[iK, jK+1]*vFld[iK, jK+1]*hFacS[iK, jK+1, k])
                )*recip_hFacC[iK, jK, k])
#-    average grid-cell-center KE to get velocity norm @ V.pt
        s = KE[i, j]+KE[i, j-1]                                     # :147-160
        m = s > 0.
        cDrag = cDrag.at[i, j].set(jnp.where(
            m,
            cDrag[i, j]
            + (bottomDragQuadratic
               )*safe_sqrt(s, m)*dragFac,
            cDrag[i, j]))
    elif sel == 1:                                                  # :161-182
        vSq = (vFld[i, j]*vFld[i, j]
               + ((uFld[i, j-1]*uFld[i, j-1]*hFacW[i, j-1, k]
                   + uFld[i, j]*uFld[i, j]*hFacW[i, j, k])
                  + (uFld[i+1, j-1]*uFld[i+1, j-1]*hFacW[i+1, j-1, k]
                     + uFld[i+1, j]*uFld[i+1, j]*hFacW[i+1, j, k])
                  )*recip_hFacS[i, j, k]*0.25)
        m = vSq > 0.
        cDrag = cDrag.at[i, j].set(jnp.where(
            m,
            cDrag[i, j]
            + (bottomDragQuadratic
               )*safe_sqrt(vSq, m)*dragFac,
            cDrag[i, j]))
    elif sel == 2:                                                  # :183-210
        vSq = ((hFacW[i, j-1, k] + hFacW[i, j, k])
               + (hFacW[i+1, j-1, k] + hFacW[i+1, j, k]))
        m1 = vSq > 0.
        vSq = jnp.where(
            m1,
            vFld[i, j]*vFld[i, j]
            + safe_div(((uFld[i, j-1]*uFld[i, j-1]*hFacW[i, j-1, k]
                         + uFld[i, j]*uFld[i, j]*hFacW[i, j, k])
                        + (uFld[i+1, j-1]*uFld[i+1, j-1]*hFacW[i+1, j-1, k]
                           + uFld[i+1, j]*uFld[i+1, j]*hFacW[i+1, j, k])
                        ), vSq, m1),
            vFld[i, j]*vFld[i, j])
        m = vSq > 0.
        cDrag = cDrag.at[i, j].set(jnp.where(
            m,
            cDrag[i, j]
            + (bottomDragQuadratic
               )*safe_sqrt(vSq, m)*dragFac,
            cDrag[i, j]))
    elif sel != -1:                                                 # :211-212
        raise ValueError("MOM_V_BOTDRAG_COEFF: invalid selectBotDragQuadr value")

#--   Apply bottom mask (i.e., zero except at bottom):
    j = loop_j(1-OLy+1, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    if k == kBottom:                                                # :216-221
        cDrag = cDrag.at[i, j].set(cDrag[i, j]*maskS[i, j, k])
    else:                                                           # :222-229
        cDrag = cDrag.at[i, j].set(cDrag[i, j]*maskS[i, j, k]
                                   * (1. - maskS[i, j, kDown]))
    return KE, cDrag
