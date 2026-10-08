"""GAD_OS7MP_ADV_Y: meridional advective flux, 7th order one-step with monotonicity-preserving limiter
(@63cdc0b pkg/generic_advdiff/gad_os7mp_adv_y.F; M3 Task 30, tutorial_reentrant_channel's tempAdvScheme = 7)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN


def gad_os7mp_adv_y(k, calcCFL, deltaTloc, vTrans, vFld, maskLocS, Q, vT, *, cfg, grid):
    """GAD_OS7MP_ADV_Y( bi,bj,k, calcCFL, deltaTloc, vTrans, vFld, maskLocS, Q, vT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_os7mp_adv_y.F:3-214

    C     | SUBROUTINE GAD_OS7MP_ADV_Y                               |
    C     | o Compute Meridional advective Flux of tracer Q using    |
    C     |   7th Order DST Sceheme with monotone preserving limiter |

    `calcCFL` is a static Python bool (GAD_ADVECTION passes .TRUE.); k is not used by the routine. GRID.h:
    recip_dyC. Returns vT: zero on the 4 + 3 rows next to the halo edges (:50-58), the limited flux on
    the rest (:59-211). The point loop body works on scalars per (i,j); here every statement runs on the whole range at
    once (each point reads only inputs): the three-way IF on the transport's sign (:66-111) selects the stencil with
    `where`s (zero outside both branches, as the ELSE), the flux IF (:113-208) is a final `where` (the ELSE writes 0;
    the branch not taken computes finite values there: Eps keeps the reciprocals and rp1h_cfl finite). SIGN(1,x) is
    `copysign` (gfortran's -fsign-zero: x = -0 gives -1). MAX/MIN carry gfortran's per-site winners (the table of the
    tutorial_reentrant_channel build); a Fortran statement with nested same-op calls of different winners (:180-181,
    :186-187, :198-199, :200-201) is written as one Python statement of named intermediate calls in gfortran's
    evaluation order. Literal doubles (`1. _d 0`, `0.5 _d 0`, ...) and Eps = `1. _d -20` (:48); the commented-out
    alternative formulas (:155-156, :162-163, :170, :188-193) are not ported."""
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    Eps = 1.e-20                                                    # :48  1. _d -20
    i = loop_i(1-OLx, sNx+OLx)                                      # :50-57
    vT = vT.at[i, 1-OLy].set(0.)                                    # :51  0. _d 0
    vT = vT.at[i, 2-OLy].set(0.)                                    # :52  0. _d 0
    vT = vT.at[i, 3-OLy].set(0.)                                    # :53  0. _d 0
    vT = vT.at[i, 4-OLy].set(0.)                                    # :54  0. _d 0
    vT = vT.at[i, sNy+OLy-2].set(0.)                                # :55  0. _d 0
    vT = vT.at[i, sNy+OLy-1].set(0.)                                # :56  0. _d 0
    vT = vT.at[i, sNy+OLy].set(0.)                                  # :57  0. _d 0
    j = loop_j(1-OLy+4, sNy+OLy-3)                                  # :59
    i = loop_i(1-OLx, sNx+OLx)                                      # :60
    Loc = vFld[i, j]                                                # :62  vLoc = vFld(i,j)
    cfl = Loc                                                       # :63
    if calcCFL:                                                     # :64
        cfl = jnp.abs(Loc*deltaTloc*grid.recip_dyC[i, j])
    pos = vTrans[i, j] > 0.                                         # :66
    neg = vTrans[i, j] < 0.                                         # :81

    def pick(a, b):                                                 # :66-111: (> 0) a, (< 0) b, else 0. _d 0
        return jnp.where(pos, a, jnp.where(neg, b, 0.))

    Qippp = pick(Q[i, j+2], Q[i, j-3])                              # :67 / :82
    Qipp = pick(Q[i, j+1], Q[i, j-2])                               # :68 / :83
    Qip = pick(Q[i, j], Q[i, j-1])                                  # :69 / :84
    Qi = pick(Q[i, j-1], Q[i, j])                                   # :70 / :85
    Qim = pick(Q[i, j-2], Q[i, j+1])                                # :71 / :86
    Qimm = pick(Q[i, j-3], Q[i, j+2])                               # :72 / :87
    Qimmm = pick(Q[i, j-4], Q[i, j+3])                              # :73 / :88
    MskIpp = pick(maskLocS[i, j+2], maskLocS[i, j-2])               # :75 / :90
    MskIp = pick(maskLocS[i, j+1], maskLocS[i, j-1])                # :76 / :91
    MskI = pick(maskLocS[i, j], maskLocS[i, j])                     # :77 / :92
    MskIm = pick(maskLocS[i, j-1], maskLocS[i, j+1])                # :78 / :93
    MskImm = pick(maskLocS[i, j-2], maskLocS[i, j+2])               # :79 / :94
    MskImmm = pick(maskLocS[i, j-3], maskLocS[i, j+3])              # :80 / :95

    nz = vTrans[i, j] != 0.                                         # :113  IF (vTrans(i,j).NE.0. _d 0)
    # :114-116  2nd order correction [i i-1]
    Fac = 1.                                                        # :115  1. _d 0
    DelP = (Qip-Qi)*MskI                                            # :116
    Phi = Fac * DelP                                                # :117
    Fac = Fac * (cfl + 1.)/3.                                       # :119
    DelM = (Qi-Qim)*MskIm                                           # :120
    Del2 = DelP - DelM                                              # :121
    Phi = Phi - Fac * Del2                                          # :122
    Fac = Fac * (cfl - 2.)/4.                                       # :124
    DelPP = (Qipp-Qip)*MskIp*MskI                                   # :125
    Del2P = DelPP - DelP                                            # :126
    Del3P = Del2P - Del2                                            # :127
    Phi = Phi + Fac * Del3P                                         # :128  (Del3p)
    Fac = Fac * (cfl - 3.)/5.                                       # :130
    DelMM = (Qim-Qimm)*MskImm*MskIm                                 # :131
    Del2M = DelM - DelMM                                            # :132
    Del3M = Del2 - Del2M                                            # :133
    Del4 = Del3P - Del3M                                            # :134
    Phi = Phi + Fac * Del4                                          # :135
    Fac = Fac * (cfl + 2.)/6.                                       # :137
    DelPPP = (Qippp-Qipp)*MskIpp*MskIp*MskI                         # :138
    Del2PP = DelPP - DelP                                           # :139
    Del3PP = Del2PP - Del2P                                         # :140
    Del4P = Del3PP - Del3P                                          # :141
    Del5P = Del4P - Del4                                            # :142
    Phi = Phi + Fac * Del5P                                         # :143
    Fac = Fac * (cfl + 2.)/7.                                       # :145
    DelMMM = (Qimm-Qimmm)*MskImmm*MskImm*MskIm                      # :146
    Del2MM = DelMM - DelMMM                                         # :147
    Del3MM = Del2M - Del2MM                                         # :148
    Del4M = Del3M - Del3MM                                          # :149
    Del5M = Del4 - Del4M                                            # :150
    Del6 = Del5P - Del5M                                            # :151
    Phi = Phi - Fac * Del6                                          # :152

    DelIp = (Qip - Qi) * MskI                                       # :154
    recip_DelIp = jnp.copysign(1., DelIp)/MAX(jnp.abs(DelIp), Eps, p="b")      # :158
    Phi = Phi*recip_DelIp                                           # :159
    DelI = (Qi - Qim) * MskIm                                       # :161
    recip_DelI = jnp.copysign(1., DelI)/MAX(jnp.abs(DelI), Eps, p="b")         # :165
    rp1h = DelI*recip_DelIp                                         # :166
    rp1h_cfl = rp1h/(cfl+Eps)                                       # :167

    d2 = Del2                                                       # :173
    d2p1 = Del2P                                                    # :174
    d2m1 = Del2M                                                    # :175
    A = 4.*d2 - d2p1                                                # :176
    B = 4.*d2p1 - d2                                                # :177
    C = d2                                                          # :178
    D = d2p1                                                        # :179
    # :180-181  dp1h = max(min(min(A,B),min(C,D)),0) + min(max(max(A,B),max(C,D)),0): the calls in gfortran's
    # evaluation order (one statement, no nested same-op call; the table's winners in this order)
    (mnAB := MIN(A, B, p="b"), mnCD := MIN(C, D, p="b"), mn := MIN(mnAB, mnCD, p="b"),    # :180-181
     lo := MAX(mn, 0., p="b"), mxAB := MAX(A, B, p="b"), mxCD := MAX(C, D, p="b"),
     mx := MAX(mxAB, mxCD, p="b"), hi := MIN(mx, 0., p="a"))
    dp1h = lo + hi                                                  # :180-181
    A = 4.*d2m1 - d2                                                # :182
    B = 4.*d2 - d2m1                                                # :183
    C = d2m1                                                        # :184
    D = d2                                                          # :185
    (mnAB := MIN(A, B, p="b"), mnCD := MIN(C, D, p="b"), mn := MIN(mnAB, mnCD, p="b"),    # :186-187
     lo := MAX(mn, 0., p="b"), mxAB := MAX(A, B, p="b"), mxCD := MAX(C, D, p="b"),
     mx := MAX(mxAB, mxCD, p="b"), hi := MIN(mx, 0., p="a"))
    dm1h = lo + hi                                                  # :186-187

    PhiMD = 1./(1.-cfl)*(DelIp-dp1h)*recip_DelIp                    # :195
    PhiLC = rp1h_cfl*(1.+dm1h*recip_DelI)                           # :196

    (m1 := MIN(0., PhiMD, p="b"), m2 := MIN(0., 2.*rp1h_cfl, p="a"), m3 := MIN(m2, PhiLC, p="b"),
     PhiMin := MAX(m1, m3, p="a"))                                  # :198-199
    (x1 := MAX(2./(1.-cfl), PhiMD, p="b"), x2 := MAX(0., 2.*rp1h_cfl, p="a"), x3 := MAX(x2, PhiLC, p="b"),
     PhiMax := MIN(x1, x3, p="a"))                                  # :200-201
    Phi = MAX(PhiMin, MIN(Phi, PhiMax, p="b"), p="b")               # :202

    Psi = Phi * 0.5 * (1. - cfl)                                    # :204  0.5 _d 0
    vT = vT.at[i, j].set(jnp.where(nz, vTrans[i, j]*(Qi + Psi*DelIp), 0.))   # :205-207
    return vT
