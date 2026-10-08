"""GMREDI_WRITE_PICKUP: pkg/gmredi/gmredi_write_pickup.F @63cdc0b (host side)."""


def gmredi_write_pickup(permPickup, suff, myTime, myIter, *, exp, cfg):
    """GMREDI_WRITE_PICKUP( permPickup, suff, myTime, myIter, myThid )   @63cdc0b
    pkg/gmredi/gmredi_write_pickup.F:7-221

    C     Writes current state of passive tracers to a pickup file

    The whole body sits under `#if ( defined GM_BATES_K3D || defined GM_GEOM_VARIABLE_K )` (:33) and writes only
    when GM_useBatesK3d .OR. GM_useGEOM (:49; both default .FALSE., gmredi_readparms.F:160, :172): a build without
    those options (global_ocean.90x40x15, tutorial_global_oce_optim) writes nothing. Raise: the K3D / GEOM pickup
    (:49-216). Returns the file names written (none). GO lane."""
    opt = "GMREDI_OPTIONS.h"
    if cfg.cpp.flag("GM_BATES_K3D", opt) or cfg.cpp.flag("GM_GEOM_VARIABLE_K", opt):     # :33
        from mitjax.params_io import RunParams
        rp = RunParams(exp.run)
        on = [k for k in ("GM_useBatesK3d", "GM_useGEOM")
              if rp.has("data.gmredi", "GM_PARM01", k) and rp.get("data.gmredi", "GM_PARM01", k)]
        if on:                                                                  # :49
            raise NotImplementedError(f"GMREDI_WRITE_PICKUP: the {on} pickup (:49-216) is not ported")
    return []
