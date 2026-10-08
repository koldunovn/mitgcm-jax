"""pkg/mom_vecinv/mom_vi_hdissip.F: horizontal dissipation [del^2 - del^4] (u,v) from hDiv and vort3
(MOM_VI_HDISSIP)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_VECINV_OPTIONS.h"       # mom_vi_hdissip.F:1


def mom_vi_hdissip(k, hDiv, vort3, dStar, zStar, hFacZ, viscAh_Z, viscAh_D, viscA4_Z, viscA4_D,
                   harmonic, biharmonic, useVariableViscosity, uDissip, vDissip, *, cfg, grid, params):
    """MOM_VI_HDISSIP(bi, bj, k, hDiv, vort3, dStar, zStar, hFacZ, viscAh_Z, viscAh_D, viscA4_Z, viscA4_D,
                      harmonic, biharmonic, useVariableViscosity, uDissip, vDissip, myThid)
    @63cdc0b pkg/mom_vecinv/mom_vi_hdissip.F:3-238

    C     Calculate horizontal dissipation terms
    C     [del^2 - del^4] (u,v)

    Returns (uDissip, vDissip). Finished from lane C's Task 8 prototype (dev/prototype/style_farray.py) with the real
    configuration (`cfg.size`, `cfg.cpp.flag(..., MOM_VECINV_OPTIONS.h)`, `params.useDiagnostics`). `harmonic`,
    `biharmonic`, `useVariableViscosity` are static (arguments that select branches). The point loops (:50-72,
    :74-96, :127-176, :178-219, :221-228) run on the whole (i,j) range at once; each point reads only inputs or its
    own earlier values, so the result is the same. #ifdef MOM_VI_ORIGINAL_VISCA4 (:130-137, :154-162, :181-188,
    :199-207) is not ported (#undef in MOM_VECINV_OPTIONS.h of every M2 build; raises). The DIAGNOSTICS_FILL calls
    (:98-103, :229-234) only copy into pkg/diagnostics buffers (not ported): with useDiagnostics = .TRUE. this is a
    hard error. `0.` and `0. _d 0` are exact.
    """
    if cfg.cpp.flag("MOM_VI_ORIGINAL_VISCA4", _OPT):
        raise NotImplementedError("MOM_VI_HDISSIP: MOM_VI_ORIGINAL_VISCA4 is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    cosFacU, cosFacV = grid.cosFacU, grid.cosFacV
    recip_dxC, recip_dyC, recip_dxG, recip_dyG = grid.recip_dxC, grid.recip_dyC, grid.recip_dxG, grid.recip_dyG
    recip_hFacW, recip_hFacS = grid.recip_hFacW, grid.recip_hFacS
    maskW, maskS, recip_deepFacC = grid.maskW, grid.maskS, grid.recip_deepFacC
    viscAhD, viscAhZ, viscA4D, viscA4Z = params.viscAhD, params.viscAhZ, params.viscA4D, params.viscA4Z
    diagnostics = cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics

#     - Laplacian  terms
    j = loop_j(2-OLy, sNy+OLy-1)
    i = loop_i(2-OLx, sNx+OLx-1)
    if harmonic:                                                    # :45
        if useVariableViscosity:                                    # :49-72
            Dij = hDiv[i, j]*viscAh_D[i, j]
            Dim = hDiv[i, j-1]*viscAh_D[i, j-1]
            Dmj = hDiv[i-1, j]*viscAh_D[i-1, j]
            Zij = hFacZ[i, j]*vort3[i, j]*viscAh_Z[i, j]
            Zip = hFacZ[i, j+1]*vort3[i, j+1]*viscAh_Z[i, j+1]
            Zpj = hFacZ[i+1, j]*vort3[i+1, j]*viscAh_Z[i+1, j]

            uD2 = (
                       cosFacU[j]*(Dij-Dmj)*recip_dxC[i, j]
             -recip_hFacW[i, j, k]*(Zip-Zij)*recip_dyG[i, j])
            vD2 = (
              recip_hFacS[i, j, k]*(Zpj-Zij)*recip_dxG[i, j]
                                                   *cosFacV[j]
                                       +(Dij-Dim)*recip_dyC[i, j])

            uDissip = uDissip.at[i, j].set(uD2*maskW[i, j, k]*recip_deepFacC[k])
            vDissip = vDissip.at[i, j].set(vD2*maskS[i, j, k]*recip_deepFacC[k])
        else:                                                       # :73-96
            Dim = hDiv[i, j-1]
            Dij = hDiv[i, j]
            Dmj = hDiv[i-1, j]
            Zip = hFacZ[i, j+1]*vort3[i, j+1]
            Zij = hFacZ[i, j]*vort3[i, j]
            Zpj = hFacZ[i+1, j]*vort3[i+1, j]

            uD2 = (viscAhD*
                       cosFacU[j]*(Dij-Dmj)*recip_dxC[i, j]
                - viscAhZ*recip_hFacW[i, j, k]*
                                        (Zip-Zij)*recip_dyG[i, j])
            vD2 = (viscAhZ*recip_hFacS[i, j, k]*
                       cosFacV[j]*(Zpj-Zij)*recip_dxG[i, j]
                + viscAhD*              (Dij-Dim)*recip_dyC[i, j])

            uDissip = uDissip.at[i, j].set(uD2*maskW[i, j, k]*recip_deepFacC[k])
            vDissip = vDissip.at[i, j].set(vD2*maskS[i, j, k]*recip_deepFacC[k])
        if diagnostics:                                             # :98-103
            raise NotImplementedError("MOM_VI_HDISSIP: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")
    else:                                                           # :104-111
        uDissip = uDissip.at[i, j].set(0.)
        vDissip = vDissip.at[i, j].set(0.)

#     - Bi-harmonic terms
    if biharmonic:                                                  # :114
#--   initialize local arrays
        uD4 = uDissip.local("uD4")                                  # :41-42  _RL uD4(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
        vD4 = vDissip.local("vD4")
        jA = loop_j(1-OLy, sNy+OLy)                                 # :116-121
        iA = loop_i(1-OLx, sNx+OLx)
        uD4 = uD4.at[iA, jA].set(0.)
        vD4 = vD4.at[iA, jA].set(0.)

        if useVariableViscosity:                                    # :126-176
            Dim = dStar[i, j-1]
            Dij = dStar[i, j]
            Dmj = dStar[i-1, j]

            Zip = hFacZ[i, j+1]*zStar[i, j+1]
            Zij = hFacZ[i, j]*zStar[i, j]
            Zpj = hFacZ[i+1, j]*zStar[i+1, j]

            Dij = Dij*viscA4_D[i, j]
            Dim = Dim*viscA4_D[i, j-1]
            Dmj = Dmj*viscA4_D[i-1, j]
            Zij = Zij*viscA4_Z[i, j]
            Zip = Zip*viscA4_Z[i, j+1]
            Zpj = Zpj*viscA4_Z[i+1, j]

            uD4 = uD4.at[i, j].set(
                       cosFacU[j]*(Dij-Dmj)*recip_dxC[i, j]
             -recip_hFacW[i, j, k]*(Zip-Zij)*recip_dyG[i, j]
                       )
            vD4 = vD4.at[i, j].set(
              recip_hFacS[i, j, k]*(Zpj-Zij)*recip_dxG[i, j]
                                                   *cosFacV[j]
                                       +(Dij-Dim)*recip_dyC[i, j]
                       )
        else:                                                       # :177-220
            Dim = dStar[i, j-1]
            Dij = dStar[i, j]
            Dmj = dStar[i-1, j]

            Zip = hFacZ[i, j+1]*zStar[i, j+1]
            Zij = hFacZ[i, j]*zStar[i, j]
            Zpj = hFacZ[i+1, j]*zStar[i+1, j]

            uD4 = uD4.at[i, j].set(viscA4D*
                       cosFacU[j]*(Dij-Dmj)*recip_dxC[i, j]
                     - viscA4Z*recip_hFacW[i, j, k]*
                                        (Zip-Zij)*recip_dyG[i, j])
            vD4 = vD4.at[i, j].set(viscA4Z*recip_hFacS[i, j, k]*
                       cosFacV[j]*(Zpj-Zij)*recip_dxG[i, j]
                     + viscA4D*         (Dij-Dim)*recip_dyC[i, j])

        uD4 = uD4.at[i, j].set(-(uD4[i, j]*maskW[i, j, k]*recip_deepFacC[k]))   # :221-228
        vD4 = vD4.at[i, j].set(-(vD4[i, j]*maskS[i, j, k]*recip_deepFacC[k]))
        uDissip = uDissip.at[i, j].set(uDissip[i, j] + uD4[i, j])
        vDissip = vDissip.at[i, j].set(vDissip[i, j] + vD4[i, j])
        if diagnostics:                                             # :229-234
            raise NotImplementedError("MOM_VI_HDISSIP: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")

    return uDissip, vDissip
