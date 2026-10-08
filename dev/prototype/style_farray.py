"""Task 8 prototype, style A: Fortran-index arrays (mitjax.farray).

Three MITgcm master routines, translated literally (same statements, operation order and association), CPP branches
as static `cfg` flags. Conventions (plan Readability rules): same routine and argument names as the Fortran minus
bi, bj, myThid; the tile loop is implicit (every array carries all tiles); GRID.h / PARAMS.h variables are fields of
`grid` / `params` with their Fortran names; an output argument (O) is also an input, because points the Fortran does
not write keep their prior values; logical and integer arguments that select branches are static Python values.
"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j

# pkg/generic_advdiff/GAD.h:96-97   _RL oneSixth; PARAMETER(oneSixth=1.D0/6.D0)
oneSixth = 1.0 / 6.0


def mom_calc_ke(k, KEscheme, uFld, vFld, KE, *, cfg, grid):
    """MOM_CALC_KE(bi,bj,k,KEscheme, uFld, vFld, KE, myThid)   @63cdc0b pkg/mom_common/mom_calc_ke.F:7-141

    C !DESCRIPTION:
    C Calculates the Kinetic energy of horizontal flow
    C KE = \\frac{1}{2} \\left( h_w \\overline{u^2}^i + h_s \\overline{v^2}^j \\right)
    C !INPUT PARAMETERS:
    C  k                    :: vertical level
    C  KEscheme             :: spacial discretisation scheme for KE
    C  uFld                 :: zonal flow
    C  vFld                 :: meridional flow
    C !OUTPUT PARAMETERS:
    C  KE                   :: Kinetic energy

    `0.125`, `0.25` and `0.` are Fortran REAL*4 literals, exact in binary, so their float64 values are the same.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    rAw, rAs, recip_rA = grid.rAw, grid.rAs, grid.recip_rA
    hFacW, hFacS, recip_hFacC = grid.hFacW, grid.hFacS, grid.recip_hFacC

    if cfg.ALLOW_AUTODIFF:                                          # :53-59
        j = loop_j(1-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx)
        KE = KE.at[i, j].set(0.)

    j = loop_j(1-OLy, sNy+OLy-1)                                    # DO j=1-OLy,sNy+OLy-1 (:62, 77, ...)
    i = loop_i(1-OLx, sNx+OLx-1)
    if KEscheme == -1:                                              # :61-68
        KE = KE.at[i, j].set(0.125*(
                   (uFld[i, j]+uFld[i+1, j])**2
                  +(vFld[i, j]+vFld[i, j+1])**2))

    elif KEscheme == 0:                                             # :70-86
        KE = KE.at[i, j].set(0.25*(
                 (uFld[i, j]*uFld[i, j]
                  +uFld[i+1, j]*uFld[i+1, j])
               + (vFld[i, j]*vFld[i, j]
                  +vFld[i, j+1]*vFld[i, j+1])
                        ))

    elif KEscheme == 1:                                             # :88-99
        KE = KE.at[i, j].set(0.25*(
                 (uFld[i, j]*uFld[i, j]*rAw[i, j]
                  +uFld[i+1, j]*uFld[i+1, j]*rAw[i+1, j])
               + (vFld[i, j]*vFld[i, j]*rAs[i, j]
                  +vFld[i, j+1]*vFld[i, j+1]*rAs[i, j+1])
                        )*recip_rA[i, j])

    elif KEscheme == 2:                                             # :101-113
        KE = KE.at[i, j].set(0.25*(
                 (uFld[i, j]*uFld[i, j]*hFacW[i, j, k]
                  +uFld[i+1, j]*uFld[i+1, j]*hFacW[i+1, j, k])
               + (vFld[i, j]*vFld[i, j]*hFacS[i, j, k]
                  +vFld[i, j+1]*vFld[i, j+1]*hFacS[i, j+1, k])
                  )*recip_hFacC[i, j, k])

    elif KEscheme == 3:                                             # :115-134
        KE = KE.at[i, j].set(0.25*(
                 (
          uFld[i, j]*uFld[i, j]
              *hFacW[i, j, k]*rAw[i, j]
         +uFld[i+1, j]*uFld[i+1, j]
              *hFacW[i+1, j, k]*rAw[i+1, j]
                 )
               + (
          vFld[i, j]*vFld[i, j]
              *hFacS[i, j, k]*rAs[i, j]
         +vFld[i, j+1]*vFld[i, j+1]
              *hFacS[i, j+1, k]*rAs[i, j+1]
                 ))*recip_hFacC[i, j, k]
                         *recip_rA[i, j])

    else:                                                           # :136-137
        raise ValueError("S/R MOM_CALC_KE: We should never reach this point!")

    return KE


def gad_dst3_adv_x(k, calcCFL, deltaTloc, uTrans, uFld, maskLocW, tracer, uT, *, cfg, grid):
    """GAD_DST3_ADV_X(bi,bj,k, calcCFL, deltaTloc, uTrans, uFld, maskLocW, tracer, uT, myThid)
    @63cdc0b pkg/generic_advdiff/gad_dst3_adv_x.F:7-121

    C !DESCRIPTION:
    C  Calculates the area integrated zonal flux due to advection of a
    C  tracer using 3rd-order Direct Space and Time (DST-3) Advection Scheme
    C !INPUT PARAMETERS:
    C  k                 :: vertical level
    C  calcCFL           :: =T: calculate CFL number ; =F: take uFld as CFL.
    C  deltaTloc         :: local time-step (s)
    C  uTrans            :: zonal volume transport
    C  uFld              :: zonal flow / CFL number
    C  tracer            :: tracer field
    C !OUTPUT PARAMETERS:
    C  uT                :: zonal advective flux

    The Fortran spells the argument `tracer` also as `Tracer` (:112, :114; Fortran is case-insensitive). The point
    loop body (:78-114) works on scalars per (i,j); here every statement runs on the whole (i,j) range at once, which
    is the same computation because each point reads only inputs. `2.`, `1.`, `0.5`, `0.` are REAL*4 literals,
    exact in binary. #ifdef OLD_DST3_FORMULATION (:87-108) is not ported: not defined in the experiment's options.
    """
    if cfg.OLD_DST3_FORMULATION:
        raise NotImplementedError("GAD_DST3_ADV_X: OLD_DST3_FORMULATION is not ported")
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    recip_dxC, recip_deepFacC = grid.recip_dxC, grid.recip_deepFacC

    j = loop_j(1-OLy, sNy+OLy)                                      # :71-75
    uT = uT.at[1-OLx, j].set(0.)
    uT = uT.at[2-OLx, j].set(0.)
    uT = uT.at[sNx+OLx, j].set(0.)

    j = loop_j(1-OLy, sNy+OLy)                                      # :76-118
    i = loop_i(1-OLx+2, sNx+OLx-1)
    Rjp = (tracer[i+1, j]-tracer[i, j])*maskLocW[i+1, j]
    Rj = (tracer[i, j]-tracer[i-1, j])*maskLocW[i, j]
    Rjm = (tracer[i-1, j]-tracer[i-2, j])*maskLocW[i-1, j]

    uCFL = uFld[i, j]
    if calcCFL:
        uCFL = jnp.abs(uFld[i, j]*deltaTloc
                       *recip_dxC[i, j]*recip_deepFacC[k])
    d0 = (2.-uCFL)*(1.-uCFL)*oneSixth
    d1 = (1.-uCFL*uCFL)*oneSixth
    uT = uT.at[i, j].set(
          0.5*(uTrans[i, j]+jnp.abs(uTrans[i, j]))
             *(tracer[i-1, j] + (d0*Rj+d1*Rjm))
         +0.5*(uTrans[i, j]-jnp.abs(uTrans[i, j]))
             *(tracer[i, j] - (d0*Rj+d1*Rjp)))
    return uT


def mom_vi_hdissip(k, hDiv, vort3, dStar, zStar, hFacZ, viscAh_Z, viscAh_D, viscA4_Z, viscA4_D,
                   harmonic, biharmonic, useVariableViscosity, uDissip, vDissip, *, cfg, grid, params):
    """MOM_VI_HDISSIP(bi, bj, k, hDiv, vort3, dStar, zStar, hFacZ, viscAh_Z, viscAh_D, viscA4_Z, viscA4_D,
                      harmonic, biharmonic, useVariableViscosity, uDissip, vDissip, myThid)
    @63cdc0b pkg/mom_vecinv/mom_vi_hdissip.F:3-238

    C     Calculate horizontal dissipation terms
    C     [del^2 - del^4] (u,v)

    Returns (uDissip, vDissip). The point loops (:50-72, :74-96, :127-176, :178-219, :221-228) run on the whole
    (i,j) range at once; each point reads only inputs or its own earlier values, so the result is the same.
    #ifdef MOM_VI_ORIGINAL_VISCA4 (:130-137, :154-162, :181-188, :199-207) is not ported: not defined in the
    experiment's options. The DIAGNOSTICS_FILL calls (:98-103, :229-234) only copy into pkg/diagnostics buffers
    (not ported): with useDiagnostics = .TRUE. this is a hard error.
    """
    if cfg.MOM_VI_ORIGINAL_VISCA4:
        raise NotImplementedError("MOM_VI_HDISSIP: MOM_VI_ORIGINAL_VISCA4 is not ported")
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    cosFacU, cosFacV = grid.cosFacU, grid.cosFacV
    recip_dxC, recip_dyC, recip_dxG, recip_dyG = grid.recip_dxC, grid.recip_dyC, grid.recip_dxG, grid.recip_dyG
    recip_hFacW, recip_hFacS = grid.recip_hFacW, grid.recip_hFacS
    maskW, maskS, recip_deepFacC = grid.maskW, grid.maskS, grid.recip_deepFacC
    viscAhD, viscAhZ, viscA4D, viscA4Z = params.viscAhD, params.viscAhZ, params.viscA4D, params.viscA4Z

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
        if cfg.ALLOW_DIAGNOSTICS and cfg.useDiagnostics:            # :98-103
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
        if cfg.ALLOW_DIAGNOSTICS and cfg.useDiagnostics:            # :229-234
            raise NotImplementedError("MOM_VI_HDISSIP: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")

    return uDissip, vDissip
