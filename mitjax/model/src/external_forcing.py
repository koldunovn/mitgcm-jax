"""EXTERNAL_FORCING_U, EXTERNAL_FORCING_V: model/src/external_forcing.F @63cdc0b, the USE_OLD_EXTERNAL_FORCING arm
(adjustment.cs-32x32x1 defines it; lane B, plan Task 25). Without the option the routines are empty and
APPLY_FORCING_U/V do the work.

They add the forcing of level kLev to the DYNVARS.h tendency gU / gV of one tile (here every tile): the surface stress
in level kSurface. APPLY_FORCING_U/V (apply_forcing.F:67-92 / :257-282) call them on the State's gU(:,:,k) and take
the difference back out (mitjax/model/src/apply_forcing.py). The package hooks (AIM, ATM_PHYS, FIZHI, EDDYPSI,
RBCS, OBCS, MYPACKAGE) raise when compiled and switched on (the hooks of apply_forcing.py, the same lists).
"""

from mitjax.farray import loop_i, loop_j


def _ksurface(fp, cfg):
    """kSurface (external_forcing.F:59-65 / :199-205)."""
    if fp.fluidIsAir:
        return 0
    if fp.usingPCoords:
        return cfg.size.Nr
    return 1


def external_forcing_u(gU, iMin, iMax, jMin, jMax, kLev, myTime, *, cfg, grid, fp, ff):
    """EXTERNAL_FORCING_U( iMin,iMax, jMin,jMax, bi,bj, kLev, myTime, myThid )   @63cdc0b
    model/src/external_forcing.F:15-150 (USE_OLD_EXTERNAL_FORCING: :50-147). gU: the State's gU (all levels);
    returns it with level kLev updated."""
    from mitjax.model.src.apply_forcing import _UV_HOOKS, _hooks
    sz = cfg.size
    kSurface = _ksurface(fp, cfg)                                               # :59-65
    _hooks(cfg, "EXTERNAL_FORCING_U", _UV_HOOKS[:3])                            # :68-88
    if kLev == kSurface:                                                        # :90-99
        j = loop_j(0, sz.sNy+1)
        i = loop_i(1, sz.sNx+1)
        gU = gU.at[i, j, kLev].set(
            gU[i, j, kLev]
            + fp.foFacMom*ff.surfaceForcingU[i, j]
            * grid.recip_drF[kLev]*grid.recip_hFacW[i, j, kLev])
    elif kSurface == -1:                                                        # :100-110
        raise NotImplementedError("EXTERNAL_FORCING_U: kSurface = -1 (kSurfW) branch is not ported")
    _hooks(cfg, "EXTERNAL_FORCING_U", _UV_HOOKS[3:])                            # :112-145
    return gU


def external_forcing_v(gV, iMin, iMax, jMin, jMax, kLev, myTime, *, cfg, grid, fp, ff):
    """EXTERNAL_FORCING_V( iMin,iMax, jMin,jMax, bi,bj, kLev, myTime, myThid )   @63cdc0b
    model/src/external_forcing.F:155-289 (USE_OLD_EXTERNAL_FORCING: :190-286)."""
    from mitjax.model.src.apply_forcing import _UV_HOOKS, _hooks
    sz = cfg.size
    kSurface = _ksurface(fp, cfg)                                               # :199-205
    _hooks(cfg, "EXTERNAL_FORCING_V", _UV_HOOKS[:3])                            # :208-228
    if kLev == kSurface:                                                        # :230-239
        j = loop_j(1, sz.sNy+1)
        i = loop_i(0, sz.sNx+1)
        gV = gV.at[i, j, kLev].set(
            gV[i, j, kLev]
            + fp.foFacMom*ff.surfaceForcingV[i, j]
            * grid.recip_drF[kLev]*grid.recip_hFacS[i, j, kLev])
    elif kSurface == -1:                                                        # :240-250
        raise NotImplementedError("EXTERNAL_FORCING_V: kSurface = -1 (kSurfS) branch is not ported")
    _hooks(cfg, "EXTERNAL_FORCING_V", _UV_HOOKS[3:])                            # :252-284
    return gV
