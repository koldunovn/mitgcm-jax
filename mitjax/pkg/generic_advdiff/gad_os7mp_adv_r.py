"""GAD_OS7MP_ADV_R: vertical advective flux, 7th order one-step with monotonicity-preserving limiter
(@63cdc0b pkg/generic_advdiff/gad_os7mp_adv_r.F; M3 Task 30, tutorial_reentrant_channel's tempVertAdvScheme = 7)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN


def _dk(a, b):
    """a - b of two level indices: Python ints (a peeled level), or KIdx of one level scan (every clamp of :48-54
    inactive there, all of them k + constant: the difference is the static int a.lo - b.lo at every level)."""
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    d = a.lo - b.lo
    if a.hi - b.hi != d:
        raise TypeError(f"GAD_OS7MP_ADV_R: level offset {a} - {b} is not static")
    return d


def gad_os7mp_adv_r(k, deltaTloc, wTrans, wFld, Q, wT, *, cfg, grid):
    """GAD_OS7MP_ADV_R( bi,bj,k,deltaTloc, wTrans, wFld, Q, wT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_os7mp_adv_r.F:3-210

    C     | SUBROUTINE GAD_OS7MP_ADV_R                               |
    C     | o Compute Vertical advective Flux of tracer Q using      |
    C     |   7th Order DST Sceheme with monotone preserving limiter |

    k: the level of GAD_ADVECTION's scanned vertical loop (a Python int on the peeled levels 1..5 and Nr-3..Nr, a KIdx
    on the others, where every clamp of :48-54 is inactive: km4..kp3 are then k + constant, `_dk`); deltaTloc: REAL
    (traced);
    wTrans, wFld, wT: (1-OLx:sNx+OLx,1-OLy:sNy+OLy); Q: the same with k=(1,Nr). Returns wT, written on every point
    (:56-206). GRID.h: maskC, recip_drC. As GAD_OS7MP_ADV_X (gad_os7mp_adv_x.py): the three-way IF on wTrans' sign
    (:62-107, upwind below for wTrans < 0) is `where`s, the flux IF (:109-204) a final `where` (the ELSE writes 0),
    SIGN is `copysign`, MAX/MIN carry gfortran's per-site winners, statements with nested same-op calls of different
    winners are one Python statement of named calls in gfortran's evaluation order; `float(kp2-kp1)` etc. (:71-91) are
    REAL*4 conversions of small integers, exact."""
    sNx, sNy, OLx, OLy, Nr = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy, cfg.Nr
    maskC = grid.maskC
    Eps = 1.e-20                                                    # :46  1. _d -20
    km4 = max(1, k-4)                                               # :48; MINMAX-INT: integer (no tie or NaN case)
    km3 = max(1, k-3)                                               # :49; MINMAX-INT: integer (no tie or NaN case)
    km2 = max(1, k-2)                                               # :50; MINMAX-INT: integer (no tie or NaN case)
    km1 = max(1, k-1)                                               # :51; MINMAX-INT: integer (no tie or NaN case)
    kp1 = min(Nr, k+1)                                              # :52; MINMAX-INT: integer (no tie or NaN case)
    kp2 = min(Nr, k+2)                                              # :53; MINMAX-INT: integer (no tie or NaN case)
    kp3 = min(Nr, k+3)                                              # :54; MINMAX-INT: integer (no tie or NaN case)
    j = loop_j(1-OLy, sNy+OLy)                                      # :56
    i = loop_i(1-OLx, sNx+OLx)                                      # :57
    Loc = wFld[i, j]                                                # :59  wLoc = wFld(i,j)
    cfl = jnp.abs(Loc*deltaTloc*grid.recip_drC[k])                  # :60
    neg = wTrans[i, j] < 0.                                         # :62
    pos = wTrans[i, j] > 0.                                         # :77

    def pick(a, b):                                                 # :62-107: (< 0) a, (> 0) b, else 0. _d 0
        return jnp.where(neg, a, jnp.where(pos, b, 0.))

    Qippp = pick(Q[i, j, kp2], Q[i, j, km3])                        # :63 / :78
    Qipp = pick(Q[i, j, kp1], Q[i, j, km2])                         # :64 / :79
    Qip = pick(Q[i, j, k], Q[i, j, km1])                            # :65 / :80
    Qi = pick(Q[i, j, km1], Q[i, j, k])                             # :66 / :81
    Qim = pick(Q[i, j, km2], Q[i, j, kp1])                          # :67 / :82
    Qimm = pick(Q[i, j, km3], Q[i, j, kp2])                         # :68 / :83
    Qimmm = pick(Q[i, j, km4], Q[i, j, kp3])                        # :69 / :84
    MskIpp = pick(maskC[i, j, kp2]*float(_dk(kp2, kp1)), maskC[i, j, km2]*float(_dk(km2, km3)))     # :71 / :86
    MskIp = pick(maskC[i, j, kp1]*float(_dk(kp1, k)), maskC[i, j, km1]*float(_dk(km1, km2)))        # :72 / :87
    MskI = pick(maskC[i, j, k]*float(_dk(k, km1)), maskC[i, j, k]*float(_dk(k, km1)))               # :73 / :88
    MskIm = pick(maskC[i, j, km1]*float(_dk(km1, km2)), maskC[i, j, kp1]*float(_dk(kp1, k)))        # :74 / :89
    MskImm = pick(maskC[i, j, km2]*float(_dk(km2, km3)), maskC[i, j, kp2]*float(_dk(kp2, kp1)))     # :75 / :90
    MskImmm = pick(maskC[i, j, km3]*float(_dk(km3, km4)), maskC[i, j, kp3]*float(_dk(kp3, kp2)))    # :76 / :91

    nz = wTrans[i, j] != 0.                                         # :109  IF (wTrans(i,j).NE.0. _d 0)
    # :110-112  2nd order correction [i i-1]
    Fac = 1.                                                        # :111  1. _d 0
    DelP = (Qip-Qi)*MskI                                            # :112
    Phi = Fac * DelP                                                # :113
    Fac = Fac * (cfl + 1.)/3.                                       # :115
    DelM = (Qi-Qim)*MskIm                                           # :116
    Del2 = DelP - DelM                                              # :117
    Phi = Phi - Fac * Del2                                          # :118
    Fac = Fac * (cfl - 2.)/4.                                       # :120
    DelPP = (Qipp-Qip)*MskIp*MskI                                   # :121
    Del2P = DelPP - DelP                                            # :122
    Del3P = Del2P - Del2                                            # :123
    Phi = Phi + Fac * Del3P                                         # :124  (Del3p)
    Fac = Fac * (cfl - 3.)/5.                                       # :126
    DelMM = (Qim-Qimm)*MskImm*MskIm                                 # :127
    Del2M = DelM - DelMM                                            # :128
    Del3M = Del2 - Del2M                                            # :129
    Del4 = Del3P - Del3M                                            # :130
    Phi = Phi + Fac * Del4                                          # :131
    Fac = Fac * (cfl + 2.)/6.                                       # :133
    DelPPP = (Qippp-Qipp)*MskIpp*MskIp*MskI                         # :134
    Del2PP = DelPP - DelP                                           # :135
    Del3PP = Del2PP - Del2P                                         # :136
    Del4P = Del3PP - Del3P                                          # :137
    Del5P = Del4P - Del4                                            # :138
    Phi = Phi + Fac * Del5P                                         # :139
    Fac = Fac * (cfl + 2.)/7.                                       # :141
    DelMMM = (Qimm-Qimmm)*MskImmm*MskImm*MskIm                      # :142
    Del2MM = DelMM - DelMMM                                         # :143
    Del3MM = Del2M - Del2MM                                         # :144
    Del4M = Del3M - Del3MM                                          # :145
    Del5M = Del4 - Del4M                                            # :146
    Del6 = Del5P - Del5M                                            # :147
    Phi = Phi - Fac * Del6                                          # :148

    DelIp = (Qip - Qi) * MskI                                       # :150
    recip_DelIp = jnp.copysign(1., DelIp)/MAX(jnp.abs(DelIp), Eps, p="b")      # :154
    Phi = Phi*recip_DelIp                                           # :155
    DelI = (Qi - Qim) * MskIm                                       # :157
    recip_DelI = jnp.copysign(1., DelI)/MAX(jnp.abs(DelI), Eps, p="b")         # :161
    rp1h = DelI*recip_DelIp                                         # :162
    rp1h_cfl = rp1h/(cfl+Eps)                                       # :163

    d2 = Del2                                                       # :169
    d2p1 = Del2P                                                    # :170
    d2m1 = Del2M                                                    # :171
    A = 4.*d2 - d2p1                                                # :172
    B = 4.*d2p1 - d2                                                # :173
    C = d2                                                          # :174
    D = d2p1                                                        # :175
    # :176-177  dp1h = max(min(min(A,B),min(C,D)),0) + min(max(max(A,B),max(C,D)),0): the calls in gfortran's
    # evaluation order (one statement, no nested same-op call; the table's winners in this order)
    (mnAB := MIN(A, B, p="b"), mnCD := MIN(C, D, p="b"), mn := MIN(mnAB, mnCD, p="b"),    # :176-177
     lo := MAX(mn, 0., p="b"), mxAB := MAX(A, B, p="b"), mxCD := MAX(C, D, p="b"),
     mx := MAX(mxAB, mxCD, p="b"), hi := MIN(mx, 0., p="a"))
    dp1h = lo + hi                                                  # :176-177
    A = 4.*d2m1 - d2                                                # :178
    B = 4.*d2 - d2m1                                                # :179
    C = d2m1                                                        # :180
    D = d2                                                          # :181
    (mnAB := MIN(A, B, p="b"), mnCD := MIN(C, D, p="b"), mn := MIN(mnAB, mnCD, p="b"),    # :182-183
     lo := MAX(mn, 0., p="b"), mxAB := MAX(A, B, p="b"), mxCD := MAX(C, D, p="b"),
     mx := MAX(mxAB, mxCD, p="b"), hi := MIN(mx, 0., p="a"))
    dm1h = lo + hi                                                  # :182-183

    PhiMD = 1./(1.-cfl)*(DelIp-dp1h)*recip_DelIp                    # :191
    PhiLC = rp1h_cfl*(1.+dm1h*recip_DelI)                           # :192

    (m1 := MIN(0., PhiMD, p="b"), m2 := MIN(0., 2.*rp1h_cfl, p="a"), m3 := MIN(m2, PhiLC, p="b"),
     PhiMin := MAX(m1, m3, p="a"))                                  # :194-195
    (x1 := MAX(2./(1.-cfl), PhiMD, p="b"), x2 := MAX(0., 2.*rp1h_cfl, p="a"), x3 := MAX(x2, PhiLC, p="b"),
     PhiMax := MIN(x1, x3, p="a"))                                  # :196-197
    Phi = MAX(PhiMin, MIN(Phi, PhiMax, p="b"), p="b")               # :198

    Psi = Phi * 0.5 * (1. - cfl)                                    # :200  0.5 _d 0
    wT = wT.at[i, j].set(jnp.where(nz, wTrans[i, j]*(Qi + Psi*DelIp), 0.))   # :201-203
    return wT
