"""COST_FINAL   @63cdc0b pkg/cost/cost_final.F:11-260"""

from mitjax.eesupp.global_sum import global_sum_tile
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.cost.cost_hflux import cost_hflux
from mitjax.pkg.cost.cost_temp import cost_temp


def cost_final(cost, *, cfg, params, maskC, wtheta=None, thetalev=None, whfluxm=None, xx_gentim2d=None, ex=None,
               atl=None, test=None, pkg_final=None):
    """COST_FINAL( myThid )

    c     | o Sum of all cost function contributions.

    Branches of tutorial_global_oce_optim/code_ad: ALLOW_COST_HFLUXM and ALLOW_COST_TEMP (the experiment's own
    COST_HFLUX, COST_TEMP, :133-139, in that order); ALLOW_CTRL with useCTRL: CTRL_COST_FINAL (:101-103) adds nothing
    when useCtrlCostContribution = .FALSE. (ctrl_cost_final.F:66; set in R5's STDOUT) and is not ported otherwise;
    no profiles/obsfit/ecco/obcs/thsice/seaice/shelfice/streamice, ALLOW_COST_TEST/ATLANTIC_HEAT/STATE_FINAL/VECTOR
    undefined, no ALLOW_DIC_COST. `params`: the traced floats mult_temp_tut, mult_hflux_tut (data.cost).
    tile_fc(bi,bj) = tile_fc + mult_temp_tut*objf_temp_tut + mult_hflux_tut*objf_hflux_tut (:189-204, TEMP before
    HFLUXM as the #ifdef blocks are ordered), loc_fc (:158, :206: a print value only), GLOBAL_SUM_TILE_RL (:215),
    fc = fc + glob_fc (:217), fc = fc + glofc (:220). Returns (cost, loc_fc); `cost_final_lines` prints.
    GOADK lane (global_ocean.90x40x15/code_ad): ALLOW_COST_ATLANTIC_HEAT alone: COST_ATLANTIC_HEAT (:129-131) with
    `atl` = dict(grid (dxG, maskS, maskC, drF), params (HeatCapacity_Cp, rhoConst), myXGlobalLo, myYGlobalLo),
    tile_fc + mult_atl*objf_atl (:196-198; `params["mult_atl"]`), the objf_atl line (:173-177); the TEMP / HFLUXM
    calls only under their own flags. PTRACERS lane: ALLOW_COST_TRACER (tutorial_tracer_adjsens/code_ad): +
    mult_tracer*objf_tracer (:193-195, after the ALLOW_COST_TEST term, before ATL), its line (:168-172);
    `params["mult_tracer"]`.
    Lane M4ADCOL (1D_ocean_ice_column/code_ad): `pkg_final(cost) -> cost` runs the package finals of :86-122 in
    their order, ECCO_COST_FINAL (:97, useECCO), CTRL_COST_FINAL (:102, useCTRL), SEAICE_COST_FINAL (:113, useSEAICE)
    (Model.cost_final builds it; each adds its terms to tile_fc); a build with ALLOW_ECCO needs it."""
    for flag in ("ALLOW_COST_STATE_FINAL", "ALLOW_COST_VECTOR"):
        if getattr(cfg.cpp, flag):
            raise NotImplementedError(f"COST_FINAL: {flag} not ported")
    if cfg.cpp.ALLOW_ECCO and pkg_final is None:
        raise NotImplementedError("COST_FINAL: ALLOW_ECCO needs the package finals (`pkg_final`)")
    if pkg_final is not None:                                             # :86-122 (lane M4ADCOL)
        cost = pkg_final(cost)
    if cfg.cpp.ALLOW_COST_TEST:                                           # :124-127 (GO lane)
        from mitjax.pkg.cost.cost_test import cost_test
        if test is None:
            raise ValueError("COST_FINAL: COST_TEST needs `test` (theta, thetaLev)")
        cost = cost_test(cost, cfg=cfg, theta=test["theta"], thetaLev=test["thetaLev"], maskC=maskC)
    if cfg.cpp.ALLOW_COST_ATLANTIC_HEAT:                                  # :129-131 (GOADK lane)
        from mitjax.pkg.cost.cost_atlantic_heat import cost_atlantic_heat
        cost.objf_atl = cost_atlantic_heat(cfg=cfg, grid=atl["grid"], params=atl["params"], cost=cost,
                                           myXGlobalLo=atl["myXGlobalLo"], myYGlobalLo=atl["myYGlobalLo"])
    if cfg.cpp.ALLOW_COST_HFLUXM:                                         # :133-135
        cost = cost_hflux(cost, cfg=cfg, maskC=maskC, whfluxm=whfluxm, xx_gentim2d=xx_gentim2d)
    if cfg.cpp.ALLOW_COST_TEMP:                                           # :137-139
        cost = cost_temp(cost, cfg=cfg, maskC=maskC, wtheta=wtheta, thetalev=thetalev)
    tile_fc = cost.tile_fc                                                # :189-204 (per tile, independent)
    if cfg.cpp.ALLOW_COST_TEST:                                           # :190-192 (GO lane)
        tile_fc = tile_fc + params["mult_test"]*cost.objf_test
    if cfg.cpp.ALLOW_COST_TRACER:                                         # :193-195 (PTRACERS lane)
        tile_fc = tile_fc + params["mult_tracer"]*cost.objf_tracer
    if cfg.cpp.ALLOW_COST_ATLANTIC_HEAT:                                  # :196-198 (GOADK lane)
        tile_fc = tile_fc + params["mult_atl"]*cost.objf_atl
    if cfg.cpp.ALLOW_COST_TEMP:
        tile_fc = tile_fc + params["mult_temp_tut"]*cost.objf_temp_tut
    if cfg.cpp.ALLOW_COST_HFLUXM:
        tile_fc = tile_fc + params["mult_hflux_tut"]*cost.objf_hflux_tut
    cost.tile_fc = tile_fc
    loc_fc = global_sum_tile(tile_fc)                                     # :158, :206 0. + tile 1 + tile 2 + ...
    glob_fc = global_sum_tile(tile_fc) if ex is None else ex.global_sum_tile(tile_fc)       # :215
    cost.fc = cost.fc + glob_fc                                           # :217
    cost.fc = cost.fc + cost.glofc                                        # :220
    return cost, loc_fc


def cost_final_lines(cost, loc_fc, *, cfg, params, early_fc, files=False, rundir=None):
    """The STDOUT records COST_FINAL writes (host side): `  early fc = ` (:154-155, '(A,1PE22.14)'), the per-tile
    lines (:161-187, '(3A,1PE22.14,1X,1PE9.2)', written with WRITE(ioUnit,...) i.e. without the PID.TID prefix),
    `  local fc = ` (:211-212), ` global fc = ` (:244). Values as host floats."""
    sz = cfg.size
    out = [fortran_write("(A,1PE22.14)", "  early fc = ", float(early_fc))]
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            t = bi - 1 + (bj - 1) * sz.nSx
            tile15c = fortran_write("(2(A,I3),A)", "(bi,bj=", bi, ",", bj, ")")      # :161
            if cfg.cpp.ALLOW_COST_TEST:                                                      # :163-167 (GO lane)
                out.append(fortran_write("(3A,1PE22.14,1X,1PE9.2)", " --> objf_test     ", tile15c, " =",
                                         float(cost.objf_test[t]), float(params["mult_test"])))
            if cfg.cpp.ALLOW_COST_TRACER:                                                    # :168-172 (PTRACERS)
                out.append(fortran_write("(3A,1PE22.14,1X,1PE9.2)", " --> objf_tracer   ", tile15c, " =",
                                         float(cost.objf_tracer[t]), float(params["mult_tracer"])))
            if cfg.cpp.ALLOW_COST_ATLANTIC_HEAT:                                         # :173-177 (GOADK lane)
                out.append(fortran_write("(3A,1PE22.14,1X,1PE9.2)", " --> objf_atl      ", tile15c, " =",
                                         float(cost.objf_atl[t]), float(params["mult_atl"])))
            if cfg.cpp.ALLOW_COST_TEMP:
                out.append(fortran_write("(3A,1PE22.14,1X,1PE9.2)", " --> objf_temp_tut ", tile15c, " =",
                                         float(cost.objf_temp_tut[t]), float(params["mult_temp_tut"])))
            if cfg.cpp.ALLOW_COST_HFLUXM:
                out.append(fortran_write("(3A,1PE22.14,1X,1PE9.2)", " --> objf_hflux_tut", tile15c, " =",
                                         float(cost.objf_hflux_tut[t]), float(params["mult_hflux_tut"])))
    out.append(fortran_write("(A,1PE22.14)", "  local fc = ", float(loc_fc)))
    if files:                       # lane M4ADCOL: ifc .NE. -1 (costWriteCostFunction, :81-84, :228-242)
        from mitjax.pkg.cost.cost_copy_file import cost_copy_file
        cfname = "costfunction.0000"                                          # :229 (optimcycle 0)
        out.append(f"Writing global cost function info to {cfname}")          # :230-232
        body = [fortran_write("(A,1PE22.14)", "fc =", float(cost.fc))]        # :234
        copied, msgs = cost_copy_file(rundir, 0)                              # :237
        out += msgs
        (rundir / cfname).write_text("\n".join(body + copied) + "\n")
    out.append(fortran_write("(A,1PE22.14)", " global fc = ", float(cost.fc)))                 # :244
    return out

