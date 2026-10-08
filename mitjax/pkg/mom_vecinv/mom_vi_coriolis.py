"""pkg/mom_vecinv/mom_vi_coriolis.F: the two horizontal components of the Coriolis acceleration (MOM_VI_CORIOLIS)."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX


def mom_vi_coriolis(k, uFld, vFld, hFacZ, r_hFacZ, uCoriolisTerm, vCoriolisTerm, *, cfg, grid, params):
    """MOM_VI_CORIOLIS(bi, bj, k, uFld, vFld, hFacZ, r_hFacZ, uCoriolisTerm, vCoriolisTerm, myThid)
    @63cdc0b pkg/mom_vecinv/mom_vi_coriolis.F:7-193

    C  Calculates the 2 horizontal components of Coriolis acceleration
    C   k              :: current vertical level
    C   uFld, vFld     :: local copy of horizontal velocity (u & v components)
    C   hFacZ          :: hFac thickness factor at corner location
    C   r_hFacZ        :: reciprocal hFac thickness factor at corner location
    C   uCoriolisTerm  :: Coriolis tendency for u-component momentum
    C   vCoriolisTerm  :: Coriolis tendency for v-component momentum

    Returns (uCoriolisTerm, vCoriolisTerm). selectCoriScheme (PARAMS.h INTEGER) is static; an invalid value raises
    with the Fortran STOP message. hFacZ and r_hFacZ are not read. epsil is `_RS` = `1. _d -9` (:56, a double);
    `0.25`, `0.5` are REAL*4 literals (exact), halfRL = 0.5 _d 0 (EEPARAMS.h:73). MAX winners from the oracle's site
    table. The point loops run on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dxG, dyG, recip_dxC, recip_dyC, fCoriG = grid.dxG, grid.dyG, grid.recip_dxC, grid.recip_dyC, grid.fCoriG
    hFacW, hFacS, recip_hFacW, recip_hFacS = grid.hFacW, grid.hFacS, grid.recip_hFacW, grid.recip_hFacS
    maskW, maskS = grid.maskW, grid.maskS
    halfRL = 0.5                                                    # EEPARAMS.h:73
    selectCoriScheme = params.selectCoriScheme

    epsil = 1.e-9                                                   # :56  epsil = 1. _d -9

    j = loop_j(1-OLy, sNy+OLy-1)
    i = loop_i(2-OLx, sNx+OLx)
    if selectCoriScheme == 0:                                       # :58-72
#- Simple average, no hFac :
        vBarXY = 0.25*(
            (vFld[i, j]*dxG[i, j]
             +vFld[i-1, j]*dxG[i-1, j])
            +(vFld[i, j+1]*dxG[i, j+1]
              +vFld[i-1, j+1]*dxG[i-1, j+1])
                       )
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            +0.5*(fCoriG[i, j]+fCoriG[i, j+1]
                  )*vBarXY*recip_dxC[i, j]*maskW[i, j, k])
    elif selectCoriScheme == 1:                                     # :73-89
#- Partial-cell generalization of the Wet-point average method :
#  (Jamart & Ozer, 1986, JGR 91 (C9), 10,621-10,631)
        vBarXY = ((
            (vFld[i, j]*dxG[i, j]*hFacS[i, j, k]
             +vFld[i-1, j]*dxG[i-1, j]*hFacS[i-1, j, k])
            +(vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k]
              +vFld[i-1, j+1]*dxG[i-1, j+1]*hFacS[i-1, j+1, k]))
            / MAX(epsil, (hFacS[i, j, k]+hFacS[i-1, j, k])            # :78-84
                  +(hFacS[i, j+1, k]+hFacS[i-1, j+1, k]), p="a"))
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            +0.5*(fCoriG[i, j]+fCoriG[i, j+1]
                  )*vBarXY*recip_dxC[i, j]*maskW[i, j, k])
    elif selectCoriScheme == 2:                                     # :90-104
#- Same as above, with hFac :
        vBarXY = 0.25*(
            (vFld[i, j]*dxG[i, j]*hFacS[i, j, k]
             +vFld[i-1, j]*dxG[i-1, j]*hFacS[i-1, j, k])
            +(vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k]
              +vFld[i-1, j+1]*dxG[i-1, j+1]*hFacS[i-1, j+1, k])
                       )
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            +0.5*(fCoriG[i, j]+fCoriG[i, j+1]
                  )*vBarXY*recip_dxC[i, j]*recip_hFacW[i, j, k])
    elif selectCoriScheme == 3:                                     # :105-120
#- Energy conserving discretization :
        vBarXm = halfRL*(
            vFld[i, j]*dxG[i, j]*hFacS[i, j, k]
            +vFld[i-1, j]*dxG[i-1, j]*hFacS[i-1, j, k])
        vBarXp = halfRL*(
            vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k]
            +vFld[i-1, j+1]*dxG[i-1, j+1]*hFacS[i-1, j+1, k])
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            +0.5
            *(vBarXm*fCoriG[i, j]
              +vBarXp*fCoriG[i, j+1]
              )*recip_dxC[i, j]*recip_hFacW[i, j, k])
    else:                                                           # :121-123
        raise ValueError("MOM_VI_CORIOLIS: invalid selectCoriScheme")

    j = loop_j(2-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx-1)
    if selectCoriScheme == 0:                                       # :125-139
        uBarXY = 0.25*(
            (uFld[i, j]*dyG[i, j]
             +uFld[i, j-1]*dyG[i, j-1])
            +(uFld[i+1, j]*dyG[i+1, j]
              +uFld[i+1, j-1]*dyG[i+1, j-1])
                       )
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            -0.5*(fCoriG[i, j]+fCoriG[i+1, j]
                  )*uBarXY*recip_dyC[i, j]*maskS[i, j, k])
    elif selectCoriScheme == 1:                                     # :140-156
        uBarXY = ((
            (uFld[i, j]*dyG[i, j]*hFacW[i, j, k]
             +uFld[i, j-1]*dyG[i, j-1]*hFacW[i, j-1, k])
            +(uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k]
              +uFld[i+1, j-1]*dyG[i+1, j-1]*hFacW[i+1, j-1, k]))
            / MAX(epsil, (hFacW[i, j, k]+hFacW[i, j-1, k])            # :145-151
                  +(hFacW[i+1, j, k]+hFacW[i+1, j-1, k]), p="a"))
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            -0.5*(fCoriG[i, j]+fCoriG[i+1, j]
                  )*uBarXY*recip_dyC[i, j]*maskS[i, j, k])
    elif selectCoriScheme == 2:                                     # :157-171
        uBarXY = 0.25*(
            (uFld[i, j]*dyG[i, j]*hFacW[i, j, k]
             +uFld[i, j-1]*dyG[i, j-1]*hFacW[i, j-1, k])
            +(uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k]
              +uFld[i+1, j-1]*dyG[i+1, j-1]*hFacW[i+1, j-1, k])
                       )
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            -0.5*(fCoriG[i, j]+fCoriG[i+1, j]
                  )*uBarXY*recip_dyC[i, j]*recip_hFacS[i, j, k])
    elif selectCoriScheme == 3:                                     # :172-187
        uBarYm = halfRL*(
            uFld[i, j]*dyG[i, j]*hFacW[i, j, k]
            +uFld[i, j-1]*dyG[i, j-1]*hFacW[i, j-1, k])
        uBarYp = halfRL*(
            uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k]
            +uFld[i+1, j-1]*dyG[i+1, j-1]*hFacW[i+1, j-1, k])
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            -0.5
            *(uBarYm*fCoriG[i, j]
              +uBarYp*fCoriG[i+1, j]
              )*recip_dyC[i, j]*recip_hFacS[i, j, k])
    else:                                                           # :188-190
        raise ValueError("MOM_VI_CORIOLIS: invalid selectCoriScheme")

    return uCoriolisTerm, vCoriolisTerm
