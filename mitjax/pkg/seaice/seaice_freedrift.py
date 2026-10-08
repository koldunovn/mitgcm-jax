"""SEAICE_FREEDRIFT: pkg/seaice/seaice_freedrift.F @63cdc0b (lane M4LAB session 3: lab_sea's LSR_mixIniGuess = 0, the
free-drift solution as the initial guess of SEAICE_LSR)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import _rewrap
from mitjax.farray import loop_i, loop_j
from mitjax.ops.safe import safe_sqrt
from mitjax.pkg.seaice.seaice_params_h import HALF, ZERO
from mitjax.pkg.seaice.seaice_read_pickup import _exch_uv_xy_rl


def _atan2(y, x, mask):
    """ATAN2(y, x) on `mask` lanes, 0 elsewhere; the guard (y, x) = (0, 1) before the operation keeps the derivative
    finite where the Fortran does not call it (atan2(0, 0) has a 0/0 derivative). XLA's atan2 is glibc's bit for bit
    (mitjax/ops/libm.py MEASURED["atan2"] = 0)."""
    return jnp.where(mask, jnp.arctan2(jnp.where(mask, y, 0.0), jnp.where(mask, x, 1.0)), 0.0)


def seaice_freedrift(myTime, myIter, sf, *, cfg, sp, op, grid, state, ex):
    """SEAICE_FREEDRIFT( myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_freedrift.F:4-177

    C     | o Solve ice approximate momentum equation analytically

    `sf` SEAICE.h (FORCEX0, FORCEY0, HEFF, SIMaskU, SIMaskV, uice_fd, vice_fd), `state` DYNVARS.h (uVel, vVel),
    `grid` GRID.h (fCori, yC), `op.rhoConst`, `ex` the exchanger. Returns sf with uice_fd, vice_fd.
    Compiled with SEAICE_CGRID and SEAICE_ALLOW_FREEDRIFT (:31; else the routine is empty: raises). The tile loops are
    the leading axis; the (i,j) loops are independent per point (vectorised). The pointwise IFs are wheres with both
    arms finite and the guards before the operation: :89-95 (tmpscal1 > ZERO: SQRT, ATAN2), :99-103, :117-121 (YC <
    ZERO), :109-113 (tmpscal3 > ZERO: the two SQRTs; the outer argument HALF*(SQRT(tmpscal4)-tmpscal2) is >= 0 there, since
    sqrt(fl(x*x)) = |x| in IEEE double and tmpscal4 >= tmpscal2**2, so the extra guard `> 0` returns the same +0. at
    0), :127-131 (tmpscal4 > ZERO: ATAN2). SIN, COS, ATAN2: XLA's = glibc's (MEASURED)."""
    if not (cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h")
            and cfg.cpp.flag("SEAICE_ALLOW_FREEDRIFT", "SEAICE_OPTIONS.h")):       # :31
        raise NotImplementedError("SEAICE_FREEDRIFT: compiled empty without SEAICE_CGRID / SEAICE_ALLOW_FREEDRIFT")
    del myTime, myIter
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    if op.usingPCoords:                                                        # :44-48
        kSrf = sz.Nr
    else:
        kSrf = 1
    sf = dict(sf)
    jA, iA = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)
    uice_fd = sf["uice_fd"].at[iA, jA].set(0.)                                # :57
    vice_fd = sf["vice_fd"].at[iA, jA].set(0.)                                # :58
    uice_cntr = sf["uice_fd"].local("uice_cntr").at[iA, jA].set(0.)           # :59
    vice_cntr = sf["vice_fd"].local("vice_cntr").at[iA, jA].set(0.)           # :60
    j, i = loop_j(1, sNy), loop_i(1, sNx)                                     # :68-69
    FORCEX0, FORCEY0, HEFF = sf["FORCEX0"], sf["FORCEY0"], sf["HEFF"]
    uVel, vVel, YC = state.uVel, state.vVel, grid.yC
    taux_onIce_cntr = HALF*(                                                   # :74-75
        FORCEX0[i, j]+FORCEX0[i+1, j])
    tauy_onIce_cntr = HALF*(                                                   # :76-77
        FORCEY0[i, j]+FORCEY0[i, j+1])
    mIceCor = sp.SEAICE_rhoIce*HEFF[i, j]*grid.fCori[i, j]                     # :79
    uvel_cntr = HALF*(uVel[i, j, kSrf]+uVel[i+1, j, kSrf])                     # :81
    vvel_cntr = HALF*(vVel[i, j, kSrf]+vVel[i, j+1, kSrf])                     # :82
    rhs_x = -taux_onIce_cntr - mIceCor*vvel_cntr                               # :84
    rhs_y = -tauy_onIce_cntr + mIceCor*uvel_cntr                               # :85
    tmpscal1 = rhs_x*rhs_x + rhs_y*rhs_y                                       # :88
    pos1 = tmpscal1 > ZERO                                                     # :89
    rhs_n = safe_sqrt(rhs_x*rhs_x + rhs_y*rhs_y, pos1)                         # :90 (else :93 0. _d 0)
    rhs_a = _atan2(rhs_y, rhs_x, pos1)                                         # :91 (else :94 0. _d 0)
    south = YC[i, j] < ZERO                                                    # :99
    tmpscal1 = jnp.where(south, 1.0 / op.rhoConst / sp.SEAICE_waterDrag_south,    # :100
                         1.0 / op.rhoConst / sp.SEAICE_waterDrag)                # :102
    tmpscal2 = tmpscal1*tmpscal1*mIceCor*mIceCor                               # :105
    tmpscal3 = tmpscal1*tmpscal1*rhs_n*rhs_n                                   # :106
    tmpscal4 = tmpscal2*tmpscal2+4.0*tmpscal3                                  # :108
    pos3 = tmpscal3 > ZERO                                                     # :109
    arg = HALF*(safe_sqrt(tmpscal4, pos3)-tmpscal2)
    sol_n = safe_sqrt(arg, pos3 & (arg > ZERO))                                # :110 (else :112 0. _d 0)
    tmpscal1 = jnp.where(south, sp.SEAICE_waterDrag_south*op.rhoConst,        # :117-121
                         sp.SEAICE_waterDrag*op.rhoConst)
    tmpscal2 = tmpscal1*sol_n*sol_n                                            # :123
    tmpscal3 = mIceCor*sol_n                                                   # :124
    tmpscal4 = tmpscal2*tmpscal2 + tmpscal3*tmpscal3                           # :126
    pos4 = tmpscal4 > ZERO                                                     # :127
    sol_a = jnp.where(pos4, rhs_a-_atan2(tmpscal3, tmpscal2, pos4), 0.)        # :128 (else :130 0. _d 0)
    uice_cntr = uice_cntr.at[i, j].set(uvel_cntr-sol_n*jnp.cos(sol_a))         # :135
    vice_cntr = vice_cntr.at[i, j].set(vvel_cntr-sol_n*jnp.sin(sol_a))         # :136
    u, v = ex.EXCH_UV_AGRID_3D_RL(uice_cntr.data, vice_cntr.data, True)      # :146 (one level)
    uice_cntr, vice_cntr = _rewrap(u, uice_cntr), _rewrap(v, vice_cntr)
    uice_fd = uice_fd.at[i, j].set(HALF*(                                      # :152-153
        uice_cntr[i-1, j]+uice_cntr[i, j]))
    vice_fd = vice_fd.at[i, j].set(HALF*(                                      # :154-155
        vice_cntr[i, j-1]+vice_cntr[i, j]))
    uice_fd, vice_fd = _exch_uv_xy_rl(uice_fd, vice_fd, True, ex=ex)           # :161
    sf["uice_fd"] = uice_fd.at[iA, jA].set(uice_fd[iA, jA]*sf["SIMaskU"][iA, jA])   # :168
    sf["vice_fd"] = vice_fd.at[iA, jA].set(vice_fd[iA, jA]*sf["SIMaskV"][iA, jA])   # :169
    return sf
