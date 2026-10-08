"""INI_NH_VARS: model/src/ini_nh_vars.F @63cdc0b (lane B, plan Task 25: ALLOW_NONHYDROSTATIC builds, cs32x15)."""

from mitjax.farray import loop_i, loop_j, loops_kji


def ini_nh_vars(state, *, cfg):
    """INI_NH_VARS( myThid )   @63cdc0b model/src/ini_nh_vars.F:6-91

    C     | SUBROUTINE INI_NH_VARS
    C     | o Initialise to zero all NH_VARS.h arrays

    Under ALLOW_NONHYDROSTATIC (the `IF (nonHydrostatic)` is commented out, :40): dPhiNH = 0. _d 0 (:47),
    phi_nh = gW = gwNm1 = 0. _d 0 (:53-54, :59, AB2) on every point of every level. ALLOW_QHYD_STAGGER_TS
    (QHydGwNm, :69-88) is not carried (mitjax/model/state.py) and raises. Pointwise."""
    if cfg.cpp.ALLOW_QHYD_STAGGER_TS:
        raise NotImplementedError("INI_NH_VARS: QHydGwNm (ALLOW_QHYD_STAGGER_TS, :69-88) is not ported")
    if not cfg.cpp.ALLOW_NONHYDROSTATIC:
        return state
    sz = cfg.size
    j2, i2 = loop_j(1-sz.OLy, sz.sNy+sz.OLy), loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    return state.replace(dPhiNH=state.dPhiNH.at[i2, j2].set(0.),                   # :47
                         phi_nh=state.phi_nh.at[i, j, k].set(0.),                   # :53
                         gW=state.gW.at[i, j, k].set(0.),                           # :54
                         gwNm1=state.gwNm1.at[i, j, k].set(0.))                     # :59
