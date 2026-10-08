"""Task 8 prototype, style B: plain slices of [tile, (k), j, i] arrays (no wrapper).

The same three routines as style_farray.py, statement for statement, written as plain jax.numpy slicing of the
storage arrays: axes [tile, j, i] for one level, [tile, k, j, i] for 3-D GRID.h fields, [tile, j] for cosFacU/V,
[k] for recip_deepFacC. Fortran index n of a dimension declared (1-OL:...) sits at storage index n-1+OL; `F(lo, hi,
OL)` is the storage slice of the Fortran loop DO n=lo,hi; a shifted index (i+1) is the shifted slice (ip1); a level
k is storage index k-1 (kk). Conventions otherwise as style_farray.py.
"""

import jax.numpy as jnp

# pkg/generic_advdiff/GAD.h:96-97   _RL oneSixth; PARAMETER(oneSixth=1.D0/6.D0)
oneSixth = 1.0 / 6.0


def F(lo, hi, OL):
    """Storage slice of the Fortran loop DO n=lo,hi over a dimension declared (1-OL:...)."""
    return slice(lo - 1 + OL, hi + OL)


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
    kk = k - 1

    if cfg.ALLOW_AUTODIFF:                                          # :53-59
        KE = KE.at[:, F(1-OLy, sNy+OLy, OLy), F(1-OLx, sNx+OLx, OLx)].set(0.)

    j, jp1 = F(1-OLy, sNy+OLy-1, OLy), F(2-OLy, sNy+OLy, OLy)       # DO j=1-OLy,sNy+OLy-1 ; j+1
    i, ip1 = F(1-OLx, sNx+OLx-1, OLx), F(2-OLx, sNx+OLx, OLx)       # DO i=1-OLx,sNx+OLx-1 ; i+1
    if KEscheme == -1:                                              # :61-68
        KE = KE.at[:, j, i].set(0.125*(
                   (uFld[:, j, i]+uFld[:, j, ip1])**2
                  +(vFld[:, j, i]+vFld[:, jp1, i])**2))

    elif KEscheme == 0:                                             # :70-86
        KE = KE.at[:, j, i].set(0.25*(
                 (uFld[:, j, i]*uFld[:, j, i]
                  +uFld[:, j, ip1]*uFld[:, j, ip1])
               + (vFld[:, j, i]*vFld[:, j, i]
                  +vFld[:, jp1, i]*vFld[:, jp1, i])
                        ))

    elif KEscheme == 1:                                             # :88-99
        KE = KE.at[:, j, i].set(0.25*(
                 (uFld[:, j, i]*uFld[:, j, i]*rAw[:, j, i]
                  +uFld[:, j, ip1]*uFld[:, j, ip1]*rAw[:, j, ip1])
               + (vFld[:, j, i]*vFld[:, j, i]*rAs[:, j, i]
                  +vFld[:, jp1, i]*vFld[:, jp1, i]*rAs[:, jp1, i])
                        )*recip_rA[:, j, i])

    elif KEscheme == 2:                                             # :101-113
        KE = KE.at[:, j, i].set(0.25*(
                 (uFld[:, j, i]*uFld[:, j, i]*hFacW[:, kk, j, i]
                  +uFld[:, j, ip1]*uFld[:, j, ip1]*hFacW[:, kk, j, ip1])
               + (vFld[:, j, i]*vFld[:, j, i]*hFacS[:, kk, j, i]
                  +vFld[:, jp1, i]*vFld[:, jp1, i]*hFacS[:, kk, jp1, i])
                  )*recip_hFacC[:, kk, j, i])

    elif KEscheme == 3:                                             # :115-134
        KE = KE.at[:, j, i].set(0.25*(
                 (
          uFld[:, j, i]*uFld[:, j, i]
              *hFacW[:, kk, j, i]*rAw[:, j, i]
         +uFld[:, j, ip1]*uFld[:, j, ip1]
              *hFacW[:, kk, j, ip1]*rAw[:, j, ip1]
                 )
               + (
          vFld[:, j, i]*vFld[:, j, i]
              *hFacS[:, kk, j, i]*rAs[:, j, i]
         +vFld[:, jp1, i]*vFld[:, jp1, i]
              *hFacS[:, kk, jp1, i]*rAs[:, jp1, i]
                 ))*recip_hFacC[:, kk, j, i]
                         *recip_rA[:, j, i])

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
    kk = k - 1

    j = F(1-OLy, sNy+OLy, OLy)                                      # :71-75  DO j=1-OLy,sNy+OLy
    uT = uT.at[:, j, 0].set(0.)                                     # uT(1-OLx,j)
    uT = uT.at[:, j, 1].set(0.)                                     # uT(2-OLx,j)
    uT = uT.at[:, j, sNx+2*OLx-1].set(0.)                           # uT(sNx+OLx,j)

    j = F(1-OLy, sNy+OLy, OLy)                                      # :76-118  DO j=1-OLy,sNy+OLy
    i = F(1-OLx+2, sNx+OLx-1, OLx)                                  #          DO i=1-OLx+2,sNx+OLx-1
    im2, im1, ip1 = F(1-OLx, sNx+OLx-3, OLx), F(2-OLx, sNx+OLx-2, OLx), F(4-OLx, sNx+OLx, OLx)
    Rjp = (tracer[:, j, ip1]-tracer[:, j, i])*maskLocW[:, j, ip1]
    Rj = (tracer[:, j, i]-tracer[:, j, im1])*maskLocW[:, j, i]
    Rjm = (tracer[:, j, im1]-tracer[:, j, im2])*maskLocW[:, j, im1]

    uCFL = uFld[:, j, i]
    if calcCFL:
        uCFL = jnp.abs(uFld[:, j, i]*deltaTloc
                       *recip_dxC[:, j, i]*recip_deepFacC[kk])
    d0 = (2.-uCFL)*(1.-uCFL)*oneSixth
    d1 = (1.-uCFL*uCFL)*oneSixth
    uT = uT.at[:, j, i].set(
          0.5*(uTrans[:, j, i]+jnp.abs(uTrans[:, j, i]))
             *(tracer[:, j, im1] + (d0*Rj+d1*Rjm))
         +0.5*(uTrans[:, j, i]-jnp.abs(uTrans[:, j, i]))
             *(tracer[:, j, i] - (d0*Rj+d1*Rjp)))
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
    kk = k - 1

#     - Laplacian  terms
    j, jm1, jp1 = F(2-OLy, sNy+OLy-1, OLy), F(1-OLy, sNy+OLy-2, OLy), F(3-OLy, sNy+OLy, OLy)
    i, im1, ip1 = F(2-OLx, sNx+OLx-1, OLx), F(1-OLx, sNx+OLx-2, OLx), F(3-OLx, sNx+OLx, OLx)
    if harmonic:                                                    # :45
        if useVariableViscosity:                                    # :49-72
            Dij = hDiv[:, j, i]*viscAh_D[:, j, i]
            Dim = hDiv[:, jm1, i]*viscAh_D[:, jm1, i]
            Dmj = hDiv[:, j, im1]*viscAh_D[:, j, im1]
            Zij = hFacZ[:, j, i]*vort3[:, j, i]*viscAh_Z[:, j, i]
            Zip = hFacZ[:, jp1, i]*vort3[:, jp1, i]*viscAh_Z[:, jp1, i]
            Zpj = hFacZ[:, j, ip1]*vort3[:, j, ip1]*viscAh_Z[:, j, ip1]

            uD2 = (
                       cosFacU[:, j, None]*(Dij-Dmj)*recip_dxC[:, j, i]
             -recip_hFacW[:, kk, j, i]*(Zip-Zij)*recip_dyG[:, j, i])
            vD2 = (
              recip_hFacS[:, kk, j, i]*(Zpj-Zij)*recip_dxG[:, j, i]
                                                   *cosFacV[:, j, None]
                                       +(Dij-Dim)*recip_dyC[:, j, i])

            uDissip = uDissip.at[:, j, i].set(uD2*maskW[:, kk, j, i]*recip_deepFacC[kk])
            vDissip = vDissip.at[:, j, i].set(vD2*maskS[:, kk, j, i]*recip_deepFacC[kk])
        else:                                                       # :73-96
            Dim = hDiv[:, jm1, i]
            Dij = hDiv[:, j, i]
            Dmj = hDiv[:, j, im1]
            Zip = hFacZ[:, jp1, i]*vort3[:, jp1, i]
            Zij = hFacZ[:, j, i]*vort3[:, j, i]
            Zpj = hFacZ[:, j, ip1]*vort3[:, j, ip1]

            uD2 = (viscAhD*
                       cosFacU[:, j, None]*(Dij-Dmj)*recip_dxC[:, j, i]
                - viscAhZ*recip_hFacW[:, kk, j, i]*
                                        (Zip-Zij)*recip_dyG[:, j, i])
            vD2 = (viscAhZ*recip_hFacS[:, kk, j, i]*
                       cosFacV[:, j, None]*(Zpj-Zij)*recip_dxG[:, j, i]
                + viscAhD*              (Dij-Dim)*recip_dyC[:, j, i])

            uDissip = uDissip.at[:, j, i].set(uD2*maskW[:, kk, j, i]*recip_deepFacC[kk])
            vDissip = vDissip.at[:, j, i].set(vD2*maskS[:, kk, j, i]*recip_deepFacC[kk])
        if cfg.ALLOW_DIAGNOSTICS and cfg.useDiagnostics:            # :98-103
            raise NotImplementedError("MOM_VI_HDISSIP: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")
    else:                                                           # :104-111
        uDissip = uDissip.at[:, j, i].set(0.)
        vDissip = vDissip.at[:, j, i].set(0.)

#     - Bi-harmonic terms
    if biharmonic:                                                  # :114
#--   initialize local arrays
        uD4 = jnp.full_like(uDissip, jnp.nan)                       # :41-42  _RL uD4(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
        vD4 = jnp.full_like(vDissip, jnp.nan)
        jA, iA = F(1-OLy, sNy+OLy, OLy), F(1-OLx, sNx+OLx, OLx)     # :116-121
        uD4 = uD4.at[:, jA, iA].set(0.)
        vD4 = vD4.at[:, jA, iA].set(0.)

        if useVariableViscosity:                                    # :126-176
            Dim = dStar[:, jm1, i]
            Dij = dStar[:, j, i]
            Dmj = dStar[:, j, im1]

            Zip = hFacZ[:, jp1, i]*zStar[:, jp1, i]
            Zij = hFacZ[:, j, i]*zStar[:, j, i]
            Zpj = hFacZ[:, j, ip1]*zStar[:, j, ip1]

            Dij = Dij*viscA4_D[:, j, i]
            Dim = Dim*viscA4_D[:, jm1, i]
            Dmj = Dmj*viscA4_D[:, j, im1]
            Zij = Zij*viscA4_Z[:, j, i]
            Zip = Zip*viscA4_Z[:, jp1, i]
            Zpj = Zpj*viscA4_Z[:, j, ip1]

            uD4 = uD4.at[:, j, i].set(
                       cosFacU[:, j, None]*(Dij-Dmj)*recip_dxC[:, j, i]
             -recip_hFacW[:, kk, j, i]*(Zip-Zij)*recip_dyG[:, j, i]
                       )
            vD4 = vD4.at[:, j, i].set(
              recip_hFacS[:, kk, j, i]*(Zpj-Zij)*recip_dxG[:, j, i]
                                                   *cosFacV[:, j, None]
                                       +(Dij-Dim)*recip_dyC[:, j, i]
                       )
        else:                                                       # :177-220
            Dim = dStar[:, jm1, i]
            Dij = dStar[:, j, i]
            Dmj = dStar[:, j, im1]

            Zip = hFacZ[:, jp1, i]*zStar[:, jp1, i]
            Zij = hFacZ[:, j, i]*zStar[:, j, i]
            Zpj = hFacZ[:, j, ip1]*zStar[:, j, ip1]

            uD4 = uD4.at[:, j, i].set(viscA4D*
                       cosFacU[:, j, None]*(Dij-Dmj)*recip_dxC[:, j, i]
                     - viscA4Z*recip_hFacW[:, kk, j, i]*
                                        (Zip-Zij)*recip_dyG[:, j, i])
            vD4 = vD4.at[:, j, i].set(viscA4Z*recip_hFacS[:, kk, j, i]*
                       cosFacV[:, j, None]*(Zpj-Zij)*recip_dxG[:, j, i]
                     + viscA4D*         (Dij-Dim)*recip_dyC[:, j, i])

        uD4 = uD4.at[:, j, i].set(-(uD4[:, j, i]*maskW[:, kk, j, i]*recip_deepFacC[kk]))   # :221-228
        vD4 = vD4.at[:, j, i].set(-(vD4[:, j, i]*maskS[:, kk, j, i]*recip_deepFacC[kk]))
        uDissip = uDissip.at[:, j, i].set(uDissip[:, j, i] + uD4[:, j, i])
        vDissip = vDissip.at[:, j, i].set(vDissip[:, j, i] + vD4[:, j, i])
        if cfg.ALLOW_DIAGNOSTICS and cfg.useDiagnostics:            # :229-234
            raise NotImplementedError("MOM_VI_HDISSIP: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")

    return uDissip, vDissip
