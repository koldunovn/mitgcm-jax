"""pkg/mom_common/mom_visc_qgl_stretch.F (MOM_VISC_QGL_STRETCH) and mom_visc_qgl_limit.F (MOM_VISC_QGL_LIMIT): the
vortex-stretching term of the QG Leith viscosity (M3 lane MLAdjust: input.QGLeith, input.QGLthGM)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import halfRL
from mitjax.ops.fortran_minmax import MAX, MIN

PI = 3.14159265358979323844         # PARAMS.h:15-16  PARAMETER ( PI = 3.14159265358979323844D0 )


def mom_visc_qgl_stretch(k, stretching, Nsquare, myTime, myIter, *, cfg, grid, params, state):
    """MOM_VISC_QGL_STRETCH( bi, bj, k, stretching, Nsquare, myTime, myIter, myThid )
    @63cdc0b pkg/mom_common/mom_visc_qgl_stretch.F:7-244

    C     | Calculate the vortex stretching term for QG Leith

    Returns (stretching, Nsquare) of level `k` (a Python int): both zeroed on every point (:78-90), then on every
    point (iMin..iMax = the whole tile, :98-101) the four arms of :105-236 selected per point by k against kSurfC and
    kLowC (`where`; an arm that cannot hold at this k, e.g. k-1 at k = 1, is not formed). DYNVARS.h rhoInSitu and
    sigmaRfield (ALLOW_LEITH_QG; written by DO_OCEANIC_PHYS) from `state`; GRID.h kSurfC, kLowC, fCori, maskC, recip_drF,
    recip_drC, rkSign, gravitySign from `grid`; PARAMS.h gravity, recip_rhoConst from `params`. Products left to
    right as written; QGL_epsil = 1. _d -12; MAX winners per site (MLAdjust table). The scalar Nsquarep1 is set in
    the arms that read it. Compiled under ALLOW_LEITH_QG (:57); the caller tests it."""
    sz = cfg.size
    Nr = sz.Nr
    g = grid
    jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    stretching = stretching.at[iA, jA].set(0.)                      # :87  0. _d 0
    Nsquare = Nsquare.at[iA, jA].set(0.)                            # :88
    QGL_epsil = 1e-12                                               # :92  1. _d -12
    ks, kl = g.kSurfC[iA, jA], g.kLowC[iA, jA]
    gfac = params.gravity*g.gravitySign*params.recip_rhoConst       # gravity*gravitySign*recip_rhoConst (left part)
    rho, sig = state.rhoInSitu, state.sigmaRfield
    fC = g.fCori[iA, jA]
    zero = jnp.zeros_like(fC)
    stretch, nsq = zero, zero
    if 1 < k < Nr:                                                  # :105-148  k > kSurfC and k < kLowC
        buoy = gfac*rho[iA, jA, k]                                  # :112-113
        buoy_m1 = gfac*rho[iA, jA, k-1]                             # :114-115
        buoy_p1 = gfac*rho[iA, jA, k+1]                             # :116-117
        buoy_1 = halfRL * (buoy + buoy_m1)                          # :121
        buoy_2 = halfRL * (buoy + buoy_p1)                          # :122
        Nsq = gfac * sig[iA, jA, k]                                 # :127-128
        Nsqp1 = gfac * sig[iA, jA, k+1]                             # :129-130
        kernel_1 = (fC/MAX(Nsq, QGL_epsil, p="b"))*buoy_1           # :132-134
        kernel_2 = (fC/MAX(Nsqp1, QGL_epsil, p="b"))*buoy_2         # :135-137
        Nsq_c = halfRL * (Nsq + Nsqp1)                              # :141
        st = (g.maskC[iA, jA, k]
              * g.recip_drF[k]*g.rkSign
              * (kernel_2-kernel_1))                                # :146-148
        c = (k > ks) & (k < kl)
        stretch = jnp.where(c, st, stretch)
        nsq = jnp.where(c, Nsq_c, nsq)
    if k < Nr:                                                      # :158-192  k == kSurfC and k < kLowC
        buoy = gfac*rho[iA, jA, k]                                  # :164-165
        buoy_p1 = gfac*rho[iA, jA, k+1]                             # :166-167
        Nsqp1 = gfac * sig[iA, jA, k+1]                             # :173-174
        kernel_1 = (fC/MAX(Nsqp1, QGL_epsil, p="b"))*buoy           # :176-178
        kernel_2 = (fC/MAX(Nsqp1, QGL_epsil, p="b"))*buoy_p1        # :179-181
        st = (g.maskC[iA, jA, k]
              * g.recip_drC[k+1]*g.rkSign
              * (kernel_2-kernel_1))                                # :190-192
        c = (k == ks) & (k < kl)
        stretch = jnp.where(c, st, stretch)
        nsq = jnp.where(c, Nsqp1, nsq)                              # :187
    if k > 1:                                                       # :195-228  k > kSurfC and k == kLowC
        buoy = gfac*rho[iA, jA, k]                                  # :201-202
        buoy_m1 = gfac*rho[iA, jA, k-1]                             # :203-204
        Nsq = gfac * sig[iA, jA, k]                                 # :210-211
        kernel_1 = (fC/MAX(Nsq, QGL_epsil, p="b"))*buoy_m1          # :213-215
        kernel_2 = (fC/MAX(Nsq, QGL_epsil, p="b"))*buoy             # :216-218
        st = (g.maskC[iA, jA, k]
              * g.recip_drC[k]*g.rkSign
              * (kernel_2-kernel_1))                                # :226-228
        c = (k > ks) & (k == kl)
        stretch = jnp.where(c, st, stretch)
        nsq = jnp.where(c, Nsq, nsq)
    # :151-155 (k == kSurfC == kLowC) and :230-234 (elsewhere): stretching = 0. _d 0, Nsquare keeps its 0.
    return stretching.at[iA, jA].set(stretch), Nsquare.at[iA, jA].set(nsq)


def mom_visc_qgl_limit(k, stretching, Nsquare, uFld, vFld, vort3, myTime, myIter, *, cfg, grid):
    """MOM_VISC_QGL_LIMIT( bi, bj, k, stretching, Nsquare, uFld, vFld, vort3, myTime, myIter, myThid )
    @63cdc0b pkg/mom_common/mom_visc_qgl_limit.F:7-138

    C     | Limit the vortex stretching term so that the QG Leith viscosity remains finite.

    Returns stretching, capped on DO j=2-OLy,sNy+OLy-1 / DO i=2-OLx,sNx+OLx-1 (:91-133; other points keep their
    value) with eqn. (56) of Bachman et al. (2017): U_scale_sq, Ro_g_sq, Fr_g_sq, vort3C (:93-122), stretching_hold
    = MIN(ABS(stretching), ABS(vort3C*Fr_g_sq/(Ro_g_sq + Fr_g_sq**2 + QGL_lim_epsil))), stretching = SIGN(...)
    (gfortran's SIGN: the magnitude with the sign bit of the second argument). `0.5` is a REAL*4 literal (exact);
    QGL_lim_epsil = 1. _d -24; MAX/MIN winners per site (MLAdjust table). GRID.h recip_rA, fCori, drF from `grid`.
    The point loop runs on the whole range at once (each point reads only inputs). The divisions are by MAX(.., eps)
    > 0 and by Ro_g_sq + Fr_g_sq**2 + eps > 0: finite on every lane."""
    sz = cfg.size
    g = grid
    QGL_lim_epsil = 1e-24                                           # :87  1. _d -24
    j = loop_j(2-sz.OLy, sz.sNy+sz.OLy-1)                           # :91
    i = loop_i(2-sz.OLx, sz.sNx+sz.OLx-1)                           # :92
    U_scale_sq = 0.5 * (
        (uFld[i, j]*uFld[i, j]
         + uFld[i+1, j]*uFld[i+1, j])
        + (vFld[i, j]*vFld[i, j]
           + vFld[i, j+1]*vFld[i, j+1]))                            # :93-97
    Ro_g_sq = U_scale_sq * g.recip_rA[i, j] / \
        MAX(QGL_lim_epsil, g.fCori[i, j]**2, p="a")                 # :101-102
    Fr_g_sq = U_scale_sq * PI * PI / \
        (MAX((Nsquare[i, j] * g.drF[k])**2,
             QGL_lim_epsil, p="b"))                                 # :108-110
    vort3C = halfRL*halfRL*(vort3[i, j] + vort3[i+1, j]
                            + vort3[i, j+1] + vort3[i+1, j+1])      # :121-122
    stretching_hold = MIN(jnp.abs(stretching[i, j]),
                          jnp.abs(vort3C * Fr_g_sq /
                                  (Ro_g_sq + Fr_g_sq**2
                                   + QGL_lim_epsil)), p="a")        # :124-128
    return stretching.at[i, j].set(jnp.copysign(stretching_hold, stretching[i, j]))   # :130
