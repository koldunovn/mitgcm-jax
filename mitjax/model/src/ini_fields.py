"""INI_FIELDS: model/src/ini_fields.F @63cdc0b."""

from mitjax.model.src.ini_pressure import ini_pressure
from mitjax.model.src.read_pickup import read_pickup


def _routines(cfg):
    """INI_VEL, INI_THETA, INI_SALT, INI_PSURF of THIS build: the experiment's own code/<name>.F when its code
    directory holds one (genmake2 takes the first directory holding a file, the experiment's code directory before
    model/src: tools/genmake2:2945-2961; advect_xy), else model/src."""
    import importlib
    import importlib.util
    import re

    from mitjax.config.params import code_path
    from mitjax.model.src import ini_psurf, ini_salt, ini_theta, ini_vel
    mods = {"ini_vel": ini_vel, "ini_theta": ini_theta, "ini_salt": ini_salt, "ini_psurf": ini_psurf}
    out = {}
    for name, mod in mods.items():
        if (code_path(cfg) / f"{name}.F").exists():
            # the experiment's own routine: mitjax/verification/<experiment>/code/<name>.py (advect_xy, advect_cs)
            # lane B: an experiment name that is not a Python identifier (solid-body.cs-32x32x1) maps to its
            # identifier form, every other character replaced by "_" (mitjax/verification/solid_body_cs_32x32x1)
            pkg = re.sub(r"\W", "_", cfg.experiment)
            modname = f"mitjax.verification.{pkg}.code.{name}"
            if importlib.util.find_spec(f"mitjax.verification.{pkg}") is None or \
                    importlib.util.find_spec(modname) is None:
                raise NotImplementedError(f"INI_FIELDS: {cfg.experiment}'s own {name}.F is not ported")
            out[name] = getattr(importlib.import_module(modname), name)
        else:
            out[name] = getattr(mod, name)
    return out


def ini_fields(state, *, cfg, grid, params, ex, rw, dyn=None, eos=None, host=None):
    """INI_FIELDS( myThid )   @63cdc0b model/src/ini_fields.F:9-56

    C     | SUBROUTINE INI_FIELDS
    C     | o Set model initial conditions.

    :27-36 cold start (startTime .EQ. baseTime .AND. nIter0 .EQ. 0 .AND. pickupSuff .EQ. ' '): INI_VEL, INI_THETA,
    INI_SALT, INI_PSURF, INI_PRESSURE (INI_EP under INCLUDE_EP_FORCING_CODE raises); :37-39 otherwise (.NOT.useOffLine
    .OR. nonlinFreeSurf.GT.0) READ_PICKUP( nIter0 ). :41-45 INI_NH_FIELDS (ALLOW_NONHYDROSTATIC, not in M1) and
    :47-53 UPDATE_ETAWS (selectSigmaCoord != 0) raise."""
    tp, ip = params.time, params.init
    if tp.startTime == tp.baseTime and tp.nIter0 == 0 and str(ip.pickupSuff).strip() == "":     # :27-28
        r = _routines(cfg)
        state = r["ini_vel"](state, cfg=cfg, grid=grid, params=params, ex=ex, rw=rw)            # :29
        state = r["ini_theta"](state, cfg=cfg, grid=grid, params=params, ex=ex, rw=rw)          # :30
        state = r["ini_salt"](state, cfg=cfg, grid=grid, params=params, ex=ex, rw=rw)           # :31
        state = r["ini_psurf"](state, cfg=cfg, grid=grid, params=params, ex=ex, rw=rw)          # :32
        state = ini_pressure(state, cfg=cfg, params=params, grid=grid, dyn=dyn, eos=eos)        # :33
        if cfg.cpp.INCLUDE_EP_FORCING_CODE:                                                     # :34-36
            raise NotImplementedError("INI_FIELDS: INI_EP (INCLUDE_EP_FORCING_CODE) is not ported")
    elif not ip.useOffLine or ip.nonlinFreeSurf > 0:                                            # :37-39
        state = read_pickup(state, tp.nIter0, cfg=cfg, params=params, ex=ex, rw=rw, dyn=dyn, host=host)   # GO lane
    if cfg.cpp.ALLOW_NONHYDROSTATIC and ip.nonHydrostatic:                                      # :41-45 (lane B)
        raise NotImplementedError("INI_FIELDS: INI_NH_FIELDS (nonHydrostatic) is not ported")
    if cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_SIGMA_CODE and ip.selectSigmaCoord != 0:   # :47-53
        raise NotImplementedError("INI_FIELDS: UPDATE_ETAWS (selectSigmaCoord) is not ported")
    return state
