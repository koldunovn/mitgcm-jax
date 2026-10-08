"""SEAICE_DYNSOLVER: pkg/seaice/seaice_dynsolver.F @63cdc0b (lane M4OFF: the C-grid driver without dynamics)."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ad.freedrift_switch import freedrift_in_ad, lsr_with_freedrift_in_ad
from mitjax.ad.modes import lsr_derivative
from mitjax.ad.seaice_lsr_rule import seaice_lsr_forward_only
from mitjax.pkg.seaice.seaice_lsr import seaice_lsr
from mitjax.pkg.seaice.seaice_calc_ice_strength import seaice_calc_ice_strength
from mitjax.pkg.seaice.seaice_lsr import different_multiple_traced
from mitjax.ops.libm import glibc_exp
from mitjax.pkg.seaice.seaice_params_h import HALF, ONE
from mitjax.pkg.seaice.seaice_get_dynforcing import seaice_get_dynforcing
from mitjax.pkg.seaice.seaice_ocean_stress import seaice_ocean_stress


def seaice_dynsolver(myTime, myIter, sf, ff, exf, *, cfg, sp, op, exfp, grid, state, ex, probe=None):
    """SEAICE_DYNSOLVER( myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_dynsolver.F:9-743

    C     | o Ice dynamics using LSR solver
    C     |   Zhang and Hibler,   JGR, 102, 8691-8702, 1997

    `sf` SEAICE.h (dict), `ff` FFields (fu, fv), `exf` EXF_FIELDS.h. Returns (sf, ff).
    Ported for SEAICEuseDYNAMICS = .FALSE. (offline_exf_seaice/input.thermo): the locals TAUX/TAUY = 0 on every point
    (:92-120), SEAICE_GET_DYNFORCING (:130-133), the IF of :136-372 not taken, SEAICE_OCEAN_STRESS
    (SEAICEupdateOceanStress, :376-386). SEAICEuseDYNAMICS (the momentum solvers) raises; the ALLOW_AUTODIFF
    re-initialisations (:96-112, :210-221) are ported (lane M4ADLAB session 2); SEAICE_ALLOW_CLIPVELS with
    SEAICE_clipVelocities clips
    uIce / vIce to +-0.40 on every point after the ocean stress (:388-410; lane M4CS32ICE: until then the port
    skipped it silently); the diagnostics (:412-733) are output only. `probe(stage, values)` (static, optional) sees Y01_get_dynforcing (TAUX,
    TAUY) and Y09_ocean_stress (ff) where the dump build writes them."""
    # lane M4ADCOL: a build with neither SEAICE_CGRID nor SEAICE_BGRID_DYNAMICS (1D_ocean_ice_column/code_ad) calls
    # this routine too (seaice_model.F:185-187); without SEAICE_CGRID the dynamics block :135-373 and the ALLOW_AUTODIFF
    # re-initialisations :96-112 (#if ALLOW_AUTODIFF && SEAICE_CGRID) are not compiled
    cgrid = cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    dims = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    if cgrid and cfg.cpp.flag("ALLOW_AUTODIFF"):                               # :96-112 (lane M4ADLAB session 2)
        # the re-initialisation on every point (1-OLx..sNx+OLx, 1-OLy..sNy+OLy, :94-95); strDivX / strDivY
        # (:107-110, ALLOW_DIAGNOSTICS) are locals of the diagnostics (:412-733, output only): not formed
        sf = dict(sf)
        jA, iA = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)
        HEFF, AREA = sf["HEFF"], sf["AREA"]
        PRESS0 = sf["PRESS0"].at[iA, jA].set(sp.SEAICE_strength*HEFF[iA, jA]   # :99-100
                                             * glibc_exp(-sp.SEAICE_cStar*(ONE-AREA[iA, jA])))
        sf["SEAICE_zMax"] = sf["SEAICE_zMax"].at[iA, jA].set(sp.SEAICE_zetaMaxFac*PRESS0[iA, jA])   # :101
        sf["SEAICE_zMin"] = sf["SEAICE_zMin"].at[iA, jA].set(sp.SEAICE_zetaMin)   # :102
        sf["PRESS0"] = PRESS0.at[iA, jA].set(PRESS0[iA, jA]*sf["HEFFM"][iA, jA])   # :103
        if cfg.cpp.flag("SEAICE_ALLOW_FREEDRIFT", "SEAICE_OPTIONS.h"):         # :104-107
            sf["uice_fd"] = sf["uice_fd"].at[iA, jA].set(0.)                   # :105
            sf["vice_fd"] = sf["vice_fd"].at[iA, jA].set(0.)                   # :106
    T = sf["AREA"].data.shape[0]                    # the tiles of this process (a block of them under shard_map)
    TAUX = FArray(jnp.zeros((T, sNy+2*OLy, sNx+2*OLx)), "TAUX", **dims)        # :115 (every point, :92-120)
    TAUY = FArray(jnp.zeros((T, sNy+2*OLy, sNx+2*OLx)), "TAUY", **dims)        # :116
    TAUX, TAUY = seaice_get_dynforcing(sf["UICE"], sf["VICE"], sf["AREA"], sf["SIMaskU"], sf["SIMaskV"],   # :130-133
                                       TAUX, TAUY, myTime, myIter, cfg=cfg, sp=sp, exfp=exfp, exf=exf, grid=grid,
                                       op=op, ff=ff)
    if probe is not None:
        probe("Y01_get_dynforcing", dict(TAUX=TAUX, TAUY=TAUY), ff)
    lsr_out = None
    if cgrid and sp.SEAICEuseDYNAMICS:                         # :135 #ifdef SEAICE_CGRID, :136-372 (lane M4OFF s3)
        doDyn = different_multiple_traced(sp.SEAICE_deltaTdyn, myTime, sp.SEAICE_deltaTtherm)   # :137
        sf_dyn = _dynamics_forcing(TAUX, TAUY, myTime, myIter, sf, ff, cfg=cfg, sp=sp, op=op, grid=grid,
                                   state=state)                                # :145-302
        if probe is not None:
            probe("Y02_ice_strength", sf_dyn, ff)
        if (cfg.cpp.flag("SEAICE_ALLOW_FREEDRIFT", "SEAICE_OPTIONS.h")        # :304-323 (lane M4LAB session 3)
                and (sp.SEAICEuseFREEDRIFT or sp.SEAICEuseEVP or sp.LSR_mixIniGuess == 0)):
            from mitjax.pkg.seaice.seaice_freedrift import seaice_freedrift
            sf_dyn = seaice_freedrift(myTime, myIter, sf_dyn, cfg=cfg, sp=sp, op=op, grid=grid, state=state,
                                      ex=ex)                                   # :307
            if sp.SEAICEuseFREEDRIFT:                                          # :309-322 (raises in READPARMS)
                raise NotImplementedError("SEAICE_DYNSOLVER: SEAICEuseFREEDRIFT (:309-322) is not ported")
            if probe is not None:
                probe("Y03_freedrift", sf_dyn, ff)
        if probe is not None:
            probe("Y04_solver_inputs", sf_dyn, ff)                             # :325-330: no OBCS
        if sp.SEAICEuseLSR:                                                    # :343-346
            # plan decision 14 (lane M4ADLAB session 4): with SEAICE_LSR_ADJOINT_ITER the LSR's derivative is the
            # transpose of its executed sweeps (mitjax/ad/lsr_sweeps.py); every other build keeps the guard
            # (lane M4ADCS32ICE session 2: or by the explicit option sp.mjx_lsr_derivative = "sweeps",
            # mitjax/ad/modes.lsr_derivative)
            lsr = seaice_lsr if lsr_derivative(cfg, sp) == "sweeps" else seaice_lsr_forward_only
            if freedrift_in_ad(sp):
                # lane M4ADCS32ICE session 2: SEAICEuseFREEDRIFTswitchInAd (autodiff_inadmode_set_ad.F:69-72), the
                # backward-only switch of mitjax/ad/freedrift_switch.py: the value is SEAICE_LSR's, the reverse
                # derivative that of the AD-mode block :308-321 (uIce = uIce_fd, ..., no SEAICE_LSR)
                sf_dyn, lsr_out = lsr_with_freedrift_in_ad(lsr, myTime, myIter, sf_dyn, cfg=cfg, sp=sp, op=op,
                                                           grid=grid, state=state, ex=ex)
            else:
                sf_dyn, lsr_out = lsr(myTime, myIter, sf_dyn, cfg=cfg, sp=sp, op=op, grid=grid, state=state, ex=ex)
        if probe is not None:
            probe("Y06_lsr", sf_dyn, ff)
        sf = {n: _sel(doDyn, sf_dyn[n], v) if n in sf_dyn and hasattr(v, "data") else v for n, v in sf.items()}
        lsr_out = {n: jnp.where(doDyn, v, jnp.zeros_like(v)) for n, v in lsr_out.items()}
    if sp.SEAICEupdateOceanStress:                                             # :376-386
        ff = seaice_ocean_stress(TAUX, TAUY, myTime, myIter, sf, ff, cfg=cfg, sp=sp, op=op, grid=grid, state=state,
                                 ex=ex)                                        # :384-385
    if probe is not None:
        probe("Y09_ocean_stress", {}, ff)
    sf = clip_velocities(sf, cfg=cfg, sp=sp)                                  # :388-410 (lane M4CS32ICE)
    if lsr_out is not None:
        sf = {**sf, "_lsr_out": lsr_out}
    return sf, ff


def clip_velocities(sf, *, cfg, sp):
    """seaice_dynsolver.F:388-410 (lane M4CS32ICE): with SEAICE_ALLOW_CLIPVELS compiled, SEAICEuseDYNAMICS and
    SEAICE_clipVelocities, uIce / vIce = MAX(MIN(., 0.40), -0.40) on every point (REAL*8 literals `0.40 _d +00`);
    else sf unchanged. Returns sf."""
    if not (cfg.cpp.flag("SEAICE_ALLOW_CLIPVELS", "SEAICE_OPTIONS.h")          # :388
            and sp.SEAICEuseDYNAMICS and sp.SEAICE_clipVelocities):            # :389
        return sf
    sz = cfg.size
    ja = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                       # :397-400
    ia = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    sf = dict(sf)
    sf["UICE"] = sf["UICE"].at[ia, ja].set(                                    # :401-402
        MAX(MIN(sf["UICE"][ia, ja], 0.40, p="b"), -0.40, p="a"))
    sf["VICE"] = sf["VICE"].at[ia, ja].set(                                    # :403-404
        MAX(MIN(sf["VICE"][ia, ja], 0.40, p="b"), -0.40, p="a"))
    return sf


def _sel(flag, a, b):
    return type(a)(jnp.where(flag, a.data, b.data), b.name, tiled=b.tiled, _dims=b.dims)


def _dynamics_forcing(TAUX, TAUY, myTime, myIter, sf, ff, *, cfg, sp, op, grid, state):
    """SEAICE_DYNSOLVER :145-302: the ice mass per area (:145-173), SEAICE_maskRHS (:176-204, raises in
    SEAICE_READPARMS), the surface geopotential phiSurf with the atmospheric and sea-ice loading
    (ATMOSPHERIC_LOADING, :223-265), FORCEX0/FORCEY0 from the wind stress (SEAICEscaleSurfStress, :266-282) and the
    tilt (SEAICEuseTilt, :284-296), then SEAICE_CALC_ICE_STRENGTH (:298). Returns sf. The loops are independent per
    point (the bi,bj loop and the tile-local phiSurf: vectorised)."""
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    rhoIce, rhoSnow = sp.SEAICE_rhoIce, sp.SEAICE_rhoSnow
    HEFF, HSNOW, AREA = sf["HEFF"], sf["HSNOW"], sf["AREA"]
    sf = dict(sf)
    j, i = loop_j(1-OLy+1, sNy+OLy), loop_i(1-OLx+1, sNx+OLx)                  # :148-149
    massC = sf["seaiceMassC"].at[i, j].set(rhoIce*HEFF[i, j])                 # :150
    massU = sf["seaiceMassU"].at[i, j].set(rhoIce*HALF*(                      # :151-152
        HEFF[i, j] + HEFF[i-1, j]))
    massV = sf["seaiceMassV"].at[i, j].set(rhoIce*HALF*(                      # :153-154
        HEFF[i, j] + HEFF[i, j-1]))
    if sp.SEAICEaddSnowMass:                                                   # :157-171
        massC = massC.at[i, j].set(massC[i, j]                                 # :160-161
                                   + rhoSnow*HSNOW[i, j])
        massU = massU.at[i, j].set(massU[i, j]                                 # :162-164
                                   + rhoSnow*HALF*(
                                       HSNOW[i, j] + HSNOW[i-1, j]))
        massV = massV.at[i, j].set(massV[i, j]                                 # :166-168
                                   + rhoSnow*HALF*(
                                       HSNOW[i, j] + HSNOW[i, j-1]))
    sf.update(seaiceMassC=massC, seaiceMassU=massU, seaiceMassV=massV)
    if sp.SEAICE_maskRHS:                                                      # :176-204
        # (#ifndef ALLOW_AUTODIFF, :176/:205: an AD build ignores SEAICE_maskRHS; refused in either build here)
        raise NotImplementedError("SEAICE_DYNSOLVER: SEAICE_maskRHS (:176-204) is not ported")
    if cfg.cpp.flag("ALLOW_AUTODIFF") and cfg.cpp.flag("SEAICE_ALLOW_EVP", "SEAICE_OPTIONS.h"):   # :210-221
        jA, iA = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)                # :213-214 (lane M4ADLAB session 2)
        sf["stressDivergenceX"] = sf["stressDivergenceX"].at[iA, jA].set(0.)   # :215
        sf["stressDivergenceY"] = sf["stressDivergenceY"].at[iA, jA].set(0.)   # :216
    jA, iA = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)                    # :228-229, :234-235
    if op.usingPCoords:                                                        # :227-232
        raise NotImplementedError("SEAICE_DYNSOLVER: usingPCoords (phiSurf = phiHydLow) is not ported")
    phiSurf = AREA.local("phiSurf").at[iA, jA].set(grid.Bo_surf[iA, jA]*state.etaN[iA, jA])   # :236
    if not cfg.cpp.flag("ATMOSPHERIC_LOADING"):                                # :240-265
        raise NotImplementedError("SEAICE_DYNSOLVER: a build without ATMOSPHERIC_LOADING is not ported")
    usingZCoords = not op.usingPCoords
    if usingZCoords:                                                           # :243
        if op.useRealFreshWaterFlux:                                           # :244-252
            phiSurf = phiSurf.at[iA, jA].set(phiSurf[iA, jA]
                                             + (ff.pLoad[iA, jA]
                                                + ff.sIceLoad[iA, jA]*op.gravity*op.sIceLoadFac
                                                )*op.recip_rhoConst)
        else:                                                                  # :253-259
            phiSurf = phiSurf.at[iA, jA].set(phiSurf[iA, jA]
                                             + ff.pLoad[iA, jA]*op.recip_rhoConst)
    if sp.SEAICEscaleSurfStress:                                               # :267-275
        FORCEX0 = sf["FORCEX0"].at[i, j].set(TAUX[i, j]
                                             * 0.5*(AREA[i, j]+AREA[i-1, j]))
        FORCEY0 = sf["FORCEY0"].at[i, j].set(TAUY[i, j]
                                             * 0.5*(AREA[i, j]+AREA[i, j-1]))
    else:                                                                      # :276-282
        FORCEX0 = sf["FORCEX0"].at[i, j].set(TAUX[i, j])
        FORCEY0 = sf["FORCEY0"].at[i, j].set(TAUY[i, j])
    if sp.SEAICEuseTilt:                                                       # :284-296
        FORCEX0 = FORCEX0.at[i, j].set(FORCEX0[i, j]                           # :288-290
                                       - massU[i, j]*grid.recip_dxC[i, j]
                                       * (phiSurf[i, j]-phiSurf[i-1, j]))
        FORCEY0 = FORCEY0.at[i, j].set(FORCEY0[i, j]                           # :291-293
                                       - massV[i, j]*grid.recip_dyC[i, j]
                                       * (phiSurf[i, j]-phiSurf[i, j-1]))
    sf.update(FORCEX0=FORCEX0, FORCEY0=FORCEY0)
    return seaice_calc_ice_strength(myTime, myIter, sf, cfg=cfg, sp=sp)        # :298
