"""LOAD_FIELDS_DRIVER: model/src/load_fields_driver.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.model.src.external_fields_load import external_fields_load


def load_fields_driver(myTime, myIter, ff, *, cfg, fp, state, rw, ex, stdout=None, iloop=None, pre=None,
                       ctrl=None, rbcs=None, exf=None):
    """LOAD_FIELDS_DRIVER( myTime, myIter, myThid )   @63cdc0b model/src/load_fields_driver.F:19-272

    C     | SUBROUTINE LOAD_FIELDS_DRIVER
    C     | o Load external forcing fields from file

    `ff` FFIELDS.h (ini_ffields.FFields), `fp` forcing parameters (ini_parms_forcing), `state` the model State
    (theta). Returns the new FFields. Concrete myTime/myIter (EXTERNAL_FIELDS_LOAD reads files on the host).

    Ported: gcmSST = theta(ks) on every point (:148-152, :170-183: the ALLOW_AUTODIFF `ELSE` or the plain
    `ELSEIF ( fluidIsWater )`), EXTERNAL_FIELDS_LOAD when fluidIsWater (:229-236). Raises: eosType 'TEOS10'
    (CONVERT_CT2PT, :153-168), ALLOW_ADDFLUID / ALLOW_FRICTION_HEATING (:109-143, no M1 build), useCTRL
    (CTRL_MAP_GENTIM2D :186-193: pkg/ctrl is another lane's; raises until it exists), and the packages of
    :195-258 (BULK_FORCE, EXF, CheapAML, GCHEM, RBCS, AIM) when a build switches one on. DIAGNOSTICS_FILL of GCM_SST
    (:260-265) is output only (diagnostics off/on runs identical, job27827363-diagoff) and not ported.

    R5 arm (plan Task 16): `pre` (external_fields_load.ForcingPreload) with the traced step number `iloop`: the
    periodic forcing of a traced time loop (EXTERNAL_FIELDS_LOAD's host part preloaded, external_fields_load.
    external_fields_load_preloaded). `ctrl` (dict: genarr, effective, gentim2d, clock): CTRL_MAP_GENTIM2D
    (:186-193, useCTRL); then (ff, genarr) is returned.
    M3 Task 30: `rbcs` (dict: state = RBCS_FIELDS.h, pre = rbcs_preload's records, params): RBCS_FIELDS_LOAD
    (:247-251, useRBCS); then (result, new RBCS state) is returned, `result` as above.
    Lane M4COL (M4 step 3): `exf` (dict: f = EXF_FIELDS.h, pre = the EXF record preload, exfp = EXF_PARAM.h,
    exf_fp, cal, tp, grid, params, probe, mon): EXF_GETFORCING (:208-217, useEXF) of the traced step `iloop`
    (mitjax/pkg/exf/exf_preload.exf_getforcing_preloaded); then (result, new EXF_FIELDS.h) is returned, `result`
    as above. Lane M4ADCOL: with both (useCTRL and useEXF) the gentim2d controls of CTRL_MAP_GENTIM2D (:186-193, before
    EXF_GETFORCING) go to EXF_GETFFIELDS' useCTRL block; the result is ((ff, genarr), new EXF_FIELDS.h)."""
    sz = cfg.size
    if cfg.cpp.ALLOW_FRICTION_HEATING:                                          # :129-143
        raise NotImplementedError("LOAD_FIELDS_DRIVER: ALLOW_FRICTION_HEATING is not ported")
    # lane B (Task 25, cs32x15): ALLOW_ADDFLUID's block (:109-126) acts only with selectAddFluid /= 0
    if cfg.cpp.ALLOW_ADDFLUID and fp.selectAddFluid != 0:                       # :114
        raise NotImplementedError("LOAD_FIELDS_DRIVER: selectAddFluid /= 0 (addMass reset, :114-125) is not ported")
    if fp.usingPCoords:                                                         # :148-152
        ks = sz.Nr
    else:
        ks = 1
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    if fp.fluidIsWater and fp.eosType == "TEOS10":                                 # :153-168
        raise NotImplementedError("LOAD_FIELDS_DRIVER: eosType='TEOS10' (CONVERT_CT2PT) is not ported")
    elif cfg.cpp.ALLOW_AUTODIFF or fp.fluidIsWater:                             # :169-173
        ff = ff.replace(gcmSST=ff.gcmSST.at[i, j].set(state.theta[i, j, ks]))  # :178
    genarr = None
    if cfg.cpp.ALLOW_CTRL and _use(cfg, "useCTRL"):                             # :186-193 (R5 arm)
        if ctrl is None:
            raise NotImplementedError("LOAD_FIELDS_DRIVER: CTRL_MAP_GENTIM2D (useCTRL) needs the CTRL_GENARR.h "
                                      "state `ctrl`")
        from mitjax.pkg.ctrl.ctrl_map_gentim2d import ctrl_map_gentim2d
        genarr = ctrl_map_gentim2d(myTime, myIter, cfg=cfg, gentim2d=ctrl["gentim2d"], maskC=ctrl["maskC"],
                                   genarr=ctrl["genarr"], effective=ctrl["effective"], clock=ctrl["clock"])
    if cfg.cpp.ALLOW_BULK_FORCE and _use(cfg, "useBulkForce"):                 # :195-206
        raise NotImplementedError("LOAD_FIELDS_DRIVER: useBulkForce is not ported")
    exf_new = None
    if cfg.cpp.ALLOW_EXF and _use(cfg, "useEXF"):                               # :208-217 (lane M4COL)
        if exf is None or iloop is None:
            raise ValueError("LOAD_FIELDS_DRIVER: useEXF needs the EXF state and preload `exf` and the step `iloop`")
        from mitjax.pkg.exf.exf_preload import exf_getforcing_preloaded
        exf_new, ff = exf_getforcing_preloaded(iloop, myTime, myIter, exf["f"], ff, pre=exf["pre"], cfg=cfg,
                                               exf=exf["exfp"], cal=exf["cal"], grid=exf["grid"],
                                               params=exf["params"], fp=exf["exf_fp"], state=state, tp=exf["tp"],
                                               ex=ex, probe=exf.get("probe"), mon=exf.get("mon"),
                                               ctrl=None if genarr is None else dict(       # lane M4ADCOL
                                                   xx_gentim2d=genarr["xx_gentim2d"],
                                                   files={g.iarr: g.xx_gentim2d_file for g in ctrl["gentim2d"]}))
                                                                               # :214
    if cfg.cpp.ALLOW_CHEAPAML and _use(cfg, "useCheapAML"):                   # :219-227
        raise NotImplementedError("LOAD_FIELDS_DRIVER: useCheapAML is not ported")
    if fp.fluidIsWater:                                                         # :229-236
        if pre is not None:                                                     # R5 arm: traced time loop
            from mitjax.model.src.external_fields_load import external_fields_load_preloaded
            ff = external_fields_load_preloaded(iloop, ff, cfg=cfg, fp=fp, pre=pre)
        else:
            ff = external_fields_load(myTime, myIter, ff, cfg=cfg, fp=fp, rw=rw, ex=ex, stdout=stdout)
    if cfg.cpp.ALLOW_GCHEM and _use(cfg, "useGCHEM"):                         # :238-245
        raise NotImplementedError("LOAD_FIELDS_DRIVER: useGCHEM is not ported")
    rbcs_new = None
    if cfg.cpp.ALLOW_RBCS and _use(cfg, "useRBCS"):                             # :247-251 (M3 Task 30)
        if rbcs is None:
            raise ValueError("LOAD_FIELDS_DRIVER: useRBCS needs the RBCS state `rbcs`")
        from mitjax.pkg.rbcs.rbcs_fields_load import rbcs_fields_load
        rbcs_new = rbcs_fields_load(myTime, myIter, cfg=cfg, params=rbcs["params"], rbcs=rbcs["state"],
                                    pre=rbcs["pre"], ex=ex)
    if cfg.cpp.ALLOW_AIM and _use(cfg, "useAIM"):                               # :253-258
        raise NotImplementedError("LOAD_FIELDS_DRIVER: useAIM is not ported")
    out = (ff, genarr) if ctrl is not None else ff
    if rbcs is not None:
        out = (out, rbcs_new)
    if exf is not None:                                                         # lane M4COL
        out = (out, exf_new)
    return out


def _use(cfg, name):
    """A package switch as PACKAGES_BOOT leaves it; .FALSE. if the build has none."""
    return cfg.use_flag(name) if any(k.lower() == name.lower() for k, _ in cfg.use) else False

