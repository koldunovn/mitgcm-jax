"""SEAICE_INIT_FIXED: pkg/seaice/seaice_init_fixed.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import UNSET_RL
from mitjax.ops.fortran_minmax_host import MIN
from mitjax.pkg.seaice.seaice_params_h import MCPHEE_TAPER_FAC, STANTON_NUMBER, USTAR_BASE


def seaice_init_fixed(sf, *, cfg, sp, grid, params, op):
    """SEAICE_INIT_FIXED( myThid )   @63cdc0b pkg/seaice/seaice_init_fixed.F:7-490

    C     | o Initialization of sea ice model.

    `sf` SEAICE.h / SEAICE_GRID.h (dict, seaice_h.seaice_fields), `sp` SeaiceParams, `params` PARAMS.h
    (usingPCoords), `op` OceanParams. Host side (eager, once). Returns (sp, sf): sp with SEAICE_mcPheePiston set if
    it was unset (:77-92), sf with HEFFM (:244), UVM (:252-259), the metric coefficients k1/k2At* (:264-277) and KGEO
    (:378-399).
    Ported for the B-grid build and (lane M4OFF) the C-grid build (SIMaskU/V :245-248, k1AtZ/k2AtZ :272-275, the
    reset :362-374; SEAICE_ALLOW_JFNK's solver counters :187-195 are not carried), ALLOW_SITRACER's tracer
    specifications :94-144 (lane M4LAB session 4, `_sitracer_specs`), without SEAICE_ITD / SEAICE_ALLOW_SIDEDRAG / ALLOW_SHELFICE (each
    raises when compiled); SEAICEselectMetricTerms > 0 (:278-358): the spherical-polar arm (:279-294, lane M4LAB) and the
    curvilinear arm (:295-356, lane M4CS32ICE: finite differences of the cube grid, each loop on its own range; the 1D
    column runs a Cartesian grid, :945 of SEAICE_READPARMS sets 0). Not ported: SEAICE_MNC_INIT (:66-68, output), SEAICE_SUMMARY (:234, print-out; its
    values are checked against the oracle's STDOUT by the gate), SEAICEmomStartBDF (:74-75, read only with
    SEAICEuseBDF2, which this port does not carry)."""
    for o in ("SEAICE_ITD", "SEAICE_ALLOW_SIDEDRAG"):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h"):
            raise NotImplementedError(f"SEAICE_INIT_FIXED: {o} is not ported")
    if cfg.cpp.flag("ALLOW_SHELFICE") and cfg.use_flag("useShelfIce"):
        raise NotImplementedError("SEAICE_INIT_FIXED: ALLOW_SHELFICE (:465-485) is not ported")
    sz = cfg.size
    OLx, OLy, sNx, sNy, Nr = sz.OLx, sz.OLy, sz.sNx, sz.sNy, sz.Nr
    if params.usingPCoords:                                                    # :59-63
        kSrf = Nr
    else:
        kSrf = 1
    # :77-92 Set mcPheePiston coeff (if still unset)
    dzSurf = float(grid.drF.data[kSrf-1])                                      # :78
    if params.usingPCoords:                                                    # :79-80
        dzSurf = dzSurf * float(op.recip_rhoConst) * float(op.recip_gravity)
    if float(sp.SEAICE_mcPheePiston) == UNSET_RL:                              # :81
        if float(sp.SEAICE_availHeatFrac) != UNSET_RL:                         # :82-84
            mcPheePiston = float(sp.SEAICE_availHeatFrac) * dzSurf/float(sp.SEAICE_deltaTtherm)
        else:                                                                  # :86-89
            mcPheePiston = MCPHEE_TAPER_FAC * STANTON_NUMBER * USTAR_BASE
            mcPheePiston = MIN(mcPheePiston, dzSurf/float(sp.SEAICE_deltaTtherm), p="a")   # :88-89
        sp = sp.replace(SEAICE_mcPheePiston=mcPheePiston)
    if cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h"):                    # :94-144 (lane M4LAB session 4)
        sp = _sitracer_specs(sp, cfg=cfg)
    sf = dict(sf)
    # :239-403 grid info (vectorised over the tile loops)
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    cgrid = cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h")
    bgrid = cfg.cpp.flag("SEAICE_BGRID_DYNAMICS", "SEAICE_OPTIONS.h")
    sf["HEFFM"] = sf["HEFFM"].at[i, j].set(grid.maskC[i, j, kSrf])            # :244
    if cgrid:                                                                  # :245-248
        sf["SIMaskU"] = sf["SIMaskU"].at[i, j].set(grid.maskW[i, j, kSrf])
        sf["SIMaskV"] = sf["SIMaskV"].at[i, j].set(grid.maskS[i, j, kSrf])
    if bgrid:                                                                  # :251-260
        j = loop_j(1-OLy+1, sNy+OLy)                                           # :252-259
        i = loop_i(1-OLx+1, sNx+OLx)
        UVM = sf["UVM"].at[i, j].set(0.0)                                      # :254
        HEFFM = sf["HEFFM"]
        mask_uice = (HEFFM[i, j] + HEFFM[i-1, j-1]                             # :255-256
                     + HEFFM[i, j-1] + HEFFM[i-1, j])
        sf["UVM"] = UVM.at[i, j].set(jnp.where(mask_uice > 3.5, 1.0, UVM[i, j]))   # :257
    if not (cgrid or bgrid):          # lane M4ADCOL (code_ad of the column): :263-359, :361-374, :376-400 not compiled
        return sp, sf
    j = loop_j(1-OLy, sNy+OLy)                                                 # :264-277
    i = loop_i(1-OLx, sNx+OLx)
    for n in ("k1AtC", "k2AtC", "k1AtU", "k1AtV", "k2AtU", "k2AtV"):           # :266-271
        sf[n] = sf[n].at[i, j].set(0.0)
    if cgrid:                                                                  # :272-275
        sf["k1AtZ"] = sf["k1AtZ"].at[i, j].set(0.0)
        sf["k2AtZ"] = sf["k2AtZ"].at[i, j].set(0.0)
    if sp.SEAICEselectMetricTerms > 0:                                         # :278
        if params.usingSphericalPolarGrid:                                     # :279-294 (lane M4LAB, lab_sea)
            # tan(phi) of the grid (GRID.h tanPhiAtU/V), not tan(YC/YG): C and U points, Z and V points share phi
            sf["k2AtC"] = sf["k2AtC"].at[i, j].set(-grid.tanPhiAtU[i, j]*params.recip_rSphere)   # :288
            sf["k2AtU"] = sf["k2AtU"].at[i, j].set(-grid.tanPhiAtU[i, j]*params.recip_rSphere)   # :289
            sf["k2AtV"] = sf["k2AtV"].at[i, j].set(-grid.tanPhiAtV[i, j]*params.recip_rSphere)   # :290
            if cgrid:                                                          # :291-293
                sf["k2AtZ"] = sf["k2AtZ"].at[i, j].set(-grid.tanPhiAtV[i, j]*params.recip_rSphere)
        elif params.usingCurvilinearGrid:                                      # :295-356 (lane M4CS32ICE, cs32)
            # compute metric term coefficients from finite difference approximation; each loop has its own range
            # (the points outside it keep the 0. of :266-275)
            j, i = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx-1)            # :299-304
            sf["k1AtC"] = sf["k1AtC"].at[i, j].set(grid.recip_dyF[i, j]
                                                   * (grid.dyG[i+1, j] - grid.dyG[i, j])
                                                   * grid.recip_dxF[i, j])
            j, i = loop_j(1-OLy, sNy+OLy-1), loop_i(1-OLx, sNx+OLx)            # :306-311
            sf["k2AtC"] = sf["k2AtC"].at[i, j].set(grid.recip_dxF[i, j]
                                                   * (grid.dxG[i, j+1] - grid.dxG[i, j])
                                                   * grid.recip_dyF[i, j])
            j, i = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx+1, sNx+OLx)            # :313-318
            sf["k1AtU"] = sf["k1AtU"].at[i, j].set(grid.recip_dyG[i, j]
                                                   * (grid.dyF[i, j] - grid.dyF[i-1, j])
                                                   * grid.recip_dxC[i, j])
            j, i = loop_j(1-OLy, sNy+OLy-1), loop_i(1-OLx, sNx+OLx)            # :320-325
            sf["k2AtU"] = sf["k2AtU"].at[i, j].set(grid.recip_dxC[i, j]
                                                   * (grid.dxV[i, j+1] - grid.dxV[i, j])
                                                   * grid.recip_dyG[i, j])
            j, i = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx-1)            # :327-332
            sf["k1AtV"] = sf["k1AtV"].at[i, j].set(grid.recip_dyC[i, j]
                                                   * (grid.dyU[i+1, j] - grid.dyU[i, j])
                                                   * grid.recip_dxG[i, j])
            j, i = loop_j(1-OLy+1, sNy+OLy), loop_i(1-OLx, sNx+OLx)            # :334-339
            sf["k2AtV"] = sf["k2AtV"].at[i, j].set(grid.recip_dxG[i, j]
                                                   * (grid.dxF[i, j] - grid.dxF[i, j-1])
                                                   * grid.recip_dyC[i, j])
            if cgrid:                                                          # :341-355
                j, i = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx+1, sNx+OLx)        # :342-347
                sf["k1AtZ"] = sf["k1AtZ"].at[i, j].set(grid.recip_dyU[i, j]
                                                       * (grid.dyC[i, j] - grid.dyC[i-1, j])
                                                       * grid.recip_dxV[i, j])
                j, i = loop_j(1-OLy+1, sNy+OLy), loop_i(1-OLx, sNx+OLx)        # :349-354
                sf["k2AtZ"] = sf["k2AtZ"].at[i, j].set(grid.recip_dxV[i, j]
                                                       * (grid.dxC[i, j] - grid.dxC[i, j-1])
                                                       * grid.recip_dyU[i, j])
            j, i = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)              # the full range of :362-374
    if cgrid and sp.SEAICEselectMetricTerms < 2:                               # :362-374
        for n in ("k1AtU", "k2AtU", "k1AtV", "k2AtV"):
            sf[n] = sf[n].at[i, j].set(0.0)
    if not bgrid:                                                              # KGEO: SEAICE_BGRID_DYNAMICS only
        return sp, sf
    # :377-399 Choose a proxy level for geostrophic velocity
    sf["KGEO"] = sf["KGEO"].at[i, j].set(0)                                    # :380
    if cfg.cpp.flag("SEAICE_BICE_STRESS", "SEAICE_OPTIONS.h"):                 # :385-386
        sf["KGEO"] = sf["KGEO"].at[i, j].set(1)                                # :386
    else:
        raise NotImplementedError("SEAICE_INIT_FIXED: KGEO without SEAICE_BICE_STRESS (:387-397) is not ported")
    return sp, sf


def _sitracer_specs(sp, *, cfg):
    """seaice_init_fixed.F:94-144 (#ifdef ALLOW_SITRACER, lane M4LAB session 4): the SItracer specifications of the
    basic tracers, per tracer in use (DO iTracer = 1, SItrNumInUse): 'one' (:98-104) and 'age' (:106-112) set their
    exchange constants; 'salinity' (:114-121, with SEAICE_salinityTracer resetting SEAICE_salt0/saltFrac), 'ridge'
    (:123-130) and SEAICE_GREASE's 'grease' (:132-143) raise (SEAICE_TRACER_PHYS's arms of those names are not
    ported). Returns sp with the REAL arrays replaced."""
    from mitjax.pkg.seaice.seaice_params_h import ONE, ZERO
    if cfg.cpp.flag("SEAICE_GREASE", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_INIT_FIXED: SEAICE_GREASE (:132-143) is not ported")
    a = {n: list(getattr(sp, n)) for n in ("SItrFromOcean0", "SItrFromFlood0", "SItrExpand0", "SItrFromOceanFrac",
                                           "SItrFromFloodFrac")}
    for iTracer in range(1, sp.SItrNumInUse + 1):                              # :96
        n, name = iTracer - 1, sp.SItrName[iTracer - 1]
        if name == "one":                                                      # :98-104
            a["SItrFromOcean0"][n] = ONE                                       # :99
            a["SItrFromFlood0"][n] = ONE                                       # :100
            a["SItrExpand0"][n] = ONE                                          # :101
            a["SItrFromOceanFrac"][n] = ZERO                                   # :102
            a["SItrFromFloodFrac"][n] = ZERO                                   # :103
        if name == "age":                                                      # :106-112
            a["SItrFromOcean0"][n] = ZERO                                      # :107
            a["SItrFromFlood0"][n] = ZERO                                      # :108
            a["SItrExpand0"][n] = ZERO                                         # :109
            a["SItrFromOceanFrac"][n] = ZERO                                   # :110
            a["SItrFromFloodFrac"][n] = ZERO                                   # :111
        if name in ("salinity", "ridge"):                                      # :114-121, :123-130
            raise NotImplementedError(f"SEAICE_INIT_FIXED: SItrName = '{name}' (and its SEAICE_TRACER_PHYS arm) is "
                                      "not ported")
    return sp.replace(**{k: tuple(float(x) for x in v) for k, v in a.items()})
