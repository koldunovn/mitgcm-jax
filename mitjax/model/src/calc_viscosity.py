"""CALC_VISCOSITY: model/src/calc_viscosity.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def calc_viscosity(iMin, iMax, jMin, jMax, kappaRU, kappaRV, *, cfg, params, grid=None, mix=None):
    """CALC_VISCOSITY( bi,bj, iMin,iMax,jMin,jMax, kappaRU, kappaRV, myThid )
    @63cdc0b model/src/calc_viscosity.F:6-398

    C     *==========================================================*
    C     | SUBROUTINE CALC_VISCOSITY
    C     | o Calculate net vertical viscosity
    C     *==========================================================*
    C     kappaRU :: Total vertical viscosity for zonal flow.
    C     kappaRV :: Total vertical viscosity for meridional flow.

    kappaRU, kappaRV: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1). Returns (kappaRU, kappaRV), every point of every
    level set to viscArNr(MIN(k,Nr)) (:66-74; the k iterations are independent, one Python statement per level as in
    the Fortran loop). The partial-cell hacks interViscAr_pCell (:141-168) and pCellMix_select > 0 (:170-394)
    (compiled unless EXCLUDE_PCELL_MIX_CODE) raise. Static values from `params` (as INI_PARMS leaves them).

    vermix lane (M3 Task 30): the KPP hook (:76-82, useKPP: KPP_CALC_VISC of every level k <= Nr; `mix`: the mixing
    packages' state, "kppf" the KPP.h fields, `grid`: maskW, maskS). With a hook on, the DO k = 1,Nr+1 loop is a per-level caller: levels
    1..Nr run as a level scan (mitjax/ops/scan_k.scan_levels, KERNEL_GUIDE §4), k = Nr+1 (no hook call) static.
    The PP81/MY82/GGL90 hooks (:84-90, :102-117; `mix` "pp81", "my82", "ggl") and their k = Nr+1 copy (:121-130)
    likewise (gated in-model with their variants); KL10 (:93-99) raises.
    """
    sz = cfg.size
    Nr = sz.Nr
    if cfg.cpp.flag("ALLOW_KL10") and cfg.use_flag("useKL10"):           # :93-99
        raise NotImplementedError("CALC_VISCOSITY: useKL10 (its CALC_VISC hook) is not ported")
    on = {u: bool(cfg.cpp.flag(a) and cfg.use_flag(u)) for a, u in (
        ("ALLOW_KPP", "useKPP"), ("ALLOW_PP81", "usePP81"), ("ALLOW_MY82", "useMY82"), ("ALLOW_GGL90", "useGGL90"))}
    useKPP = on["useKPP"]
    hooks = any(on.values())
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)

    def level_k(k, c):                                                  # :66  DO k = 1,Nr+1
        kappaRU, kappaRV = c
        ki = min(k, Nr)                                                 # :67; MINMAX-INT: integer (no tie or NaN case)
        kappaRU = kappaRU.at[i, j, k].set(params.viscArNr[ki])          # :71
        kappaRV = kappaRV.at[i, j, k].set(params.viscArNr[ki])          # :72
        if useKPP and k <= Nr:                                          # :76-82
            from mitjax.pkg.kpp.kpp_calc_visc import kpp_calc_visc
            kappaRU, kappaRV = kpp_calc_visc(iMin, iMax, jMin, jMax, k, kappaRU, kappaRV, cfg=cfg, grid=grid,
                                             params=params, kppf=mix["kppf"])
        if on["usePP81"] and k <= Nr:                                   # :84-90
            from mitjax.pkg.pp81.pp81_calc_visc import pp81_calc_visc
            kappaRU, kappaRV = pp81_calc_visc(iMin, iMax, jMin, jMax, k, kappaRU, kappaRV, grid=grid,
                                              params=params, pp=mix["pp81"])
        if on["useMY82"] and k <= Nr:                                   # :102-108
            from mitjax.pkg.my82.my82_calc_visc import my82_calc_visc
            kappaRU, kappaRV = my82_calc_visc(iMin, iMax, jMin, jMax, k, kappaRU, kappaRV, grid=grid,
                                              params=params, my=mix["my82"])
        if on["useGGL90"] and k <= Nr:                                  # :111-117
            from mitjax.pkg.ggl90.ggl90_calc_visc import ggl90_calc_visc
            kappaRU, kappaRV = ggl90_calc_visc(iMin, iMax, jMin, jMax, k, kappaRU, kappaRV, params=params,
                                               ggl=mix["ggl"])
        if k == Nr+1 and (on["usePP81"] or on["useMY82"] or on["useGGL90"]):   # :121-130
            kappaRU = kappaRU.at[i, j, k].set(kappaRU[i, j, ki])
            kappaRV = kappaRV.at[i, j, k].set(kappaRV[i, j, ki])
        return kappaRU, kappaRV

    c = (kappaRU, kappaRV)
    if hooks:
        from mitjax.ops.scan_k import scan_levels
        c = scan_levels(level_k, c, 1, Nr)                              # k = 1..Nr (a level scan)
        c = level_k(Nr+1, c)                                            # k = Nr+1
    else:
        for k in range(1, Nr+2):
            c = level_k(k, c)
        # no hook on: :121-130 does nothing
    kappaRU, kappaRV = c
    if not cfg.cpp.EXCLUDE_PCELL_MIX_CODE:                              # :135-395
        if params.interViscAr_pCell:
            raise NotImplementedError("CALC_VISCOSITY: interViscAr_pCell (:141-168) is not ported")
        if params.pCellMix_select > 0:
            raise NotImplementedError("CALC_VISCOSITY: pCellMix_select > 0 (:170-394) is not ported")
    return kappaRU, kappaRV
