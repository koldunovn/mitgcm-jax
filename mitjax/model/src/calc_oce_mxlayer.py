"""CALC_OCE_MXLAYER: model/src/calc_oce_mxlayer.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.src.find_alpha import find_alpha
from mitjax.model.src.find_rho import find_rho_2d
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.safe import safe_div
from mitjax.ops.scan_k import level, scan_k


def calc_oce_mxlayer(rhoSurf, sigmaR, myTime, myIter, *, cfg, grid, params, eos, state, gmredi=None,
                     diagnostics_is_on=None):
    """CALC_OCE_MXLAYER( rhoSurf, sigmaR, bi, bj, myTime, myIter, myThid )
    @63cdc0b model/src/calc_oce_mxlayer.F:3-259

    C     *==========================================================*
    C     | S/R CALC_OCE_MXLAYER
    C     | o Diagnose the Oceanic surface Mixed-Layer
    C     | Note: output "hMixLayer" is in "r" unit, i.e., in Pa
    C     |       when using P-coordinate.
    C     *==========================================================*
    C     rhoSurf   :: Surface density anomaly
    C     sigmaR    :: Vertical gradient of potential density

    rhoSurf: FArray (i, j); sigmaR: FArray (i, j, Nr). Returns hMixLayer (DYNVARS.h, `state`); unchanged when the
    mixed layer is not diagnosed.

    calcMixLayerDepth (:72-84): under ALLOW_GMREDI with useGMRedi and not useKPP, from `gmredi` (GMREDI.h values
    GM_useSubMeso, GM_taper_scheme, GM_useBatesK3d; required then); under ALLOW_DIAGNOSTICS with useDiagnostics,
    `diagnostics_is_on('MXLDEPTH')` (a static callable standing for pkg/diagnostics' DIAGNOSTICS_IS_ON, which is not
    ported; required then). Ported: z coordinates (:91-96), method 1 (hMixCriteria < 0, :102-157; the k loop
    carries rhoKm1, rhoMxL and hMixLayer from level to level: scan_k in the Fortran order). Method 2 (:159-218),
    hMixSmooth > 0 (:223-246) and p coordinates raise. hMixCriteria and hMixSmooth select branches, so they are read
    through `params.static_float(name)` (the concrete host value; the traced field is used in arithmetic). The
    DIAGNOSTICS_FILL (:248-253) changes no output.
    """
    sz = cfg.size
    Nr = sz.Nr
    calcMixLayerDepth = False                                           # :72
    if cfg.cpp.ALLOW_GMREDI:                                            # :73-78
        if cfg.use_flag("useGMRedi") and not cfg.use_flag("useKPP"):
            if gmredi is None:
                raise ValueError("CALC_OCE_MXLAYER: useGMRedi is on: the GMREDI.h values (gmredi=) are needed")
            calcMixLayerDepth = (gmredi.GM_useSubMeso or gmredi.GM_taper_scheme.rstrip() == "fm07"
                                 or gmredi.GM_useBatesK3d)
    if cfg.cpp.ALLOW_DIAGNOSTICS:                                       # :79-83
        if params.useDiagnostics and not calcMixLayerDepth:
            if diagnostics_is_on is None:
                raise ValueError("CALC_OCE_MXLAYER: useDiagnostics is on: diagnostics_is_on= is needed")
            calcMixLayerDepth = bool(diagnostics_is_on("MXLDEPTH"))
    hMixLayer = state.hMixLayer
    if not calcMixLayerDepth:                                           # :84
        return hMixLayer

    if params.usingPCoords:                                             # :86-90
        raise NotImplementedError("CALC_OCE_MXLAYER: pressure coordinates are not ported")
    kTop, kSrf, kDir, deltaK = 1, 1, 1, 0                               # :92-95
    method = 0                                                          # :98-100
    if params.static_float("hMixCriteria") < 0.:
        method = 1
    if params.static_float("hMixCriteria") > 1.:
        method = 2

    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    if method == 1:                                                     # :102-157
        rhoBigNb = params.rhoConst*1.0e10                               # :112  rhoConst*1. _d 10
        rhoMxL = rhoSurf.local("rhoMxL")
        rhoKm1 = rhoSurf.local("rhoKm1")
        rhoLoc = rhoSurf.local("rhoLoc")
        rhoMxL = find_alpha(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, kSrf, kSrf,     # :113-115
                            rhoMxL, params=params, eos=eos, cfg=cfg, grid=grid, state=state)
        rhoKm1 = rhoKm1.at[i, j].set(rhoSurf[i, j])                     # :119
        rhoMxL = rhoMxL.at[i, j].set(rhoSurf[i, j]                      # :120-121
                                     + MAX(rhoMxL[i, j]*params.hMixCriteria, params.dRhoSmall, p="b"))
        hMixLayer = hMixLayer.at[i, j].set(grid.Ro_surf[i, j] - grid.R_low[i, j])         # :122
        kU = 2 + deltaK*(Nr-3)                                          # :126
        kL = Nr - deltaK*(Nr-1)                                         # :127

        def iteration(carry, x):                                        # :128-157  one level k
            rhoKm1, rhoMxL, hMix = carry
            rhoLoc_k = find_rho_2d(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, kSrf,  # :130-134
                                   x["theta"], x["salt"], rhoLoc, None,
                                   cfg=cfg, grid=grid, params=params, eos=eos, state=state)
            # :139  kIn = k.LE.klowC(i,j,bi,bj).AND.k.GE.kSurfC(i,j,bi,bj)
            kIn = (x["k"] <= grid.kLowC[i, j]) & (x["k"] >= grid.kSurfC[i, j])
            found = kIn & (rhoLoc_k[i, j] >= rhoMxL[i, j])              # :140
            up = rhoLoc_k[i, j] > rhoKm1[i, j]                          # :141
            tmpFac = jnp.where(up, safe_div(rhoMxL[i, j] - rhoKm1[i, j],                 # :142-143
                                            rhoLoc_k[i, j] - rhoKm1[i, j], up), 0.0)   # :145  0.
            hNew = (-grid.gravitySign*(grid.rF[kTop]-x["rC_kmkDir"])   # :149-150
                    + tmpFac*x["drC_kpdeltaK"])
            hMix = hMix.at[i, j].set(jnp.where(found, hNew, hMix[i, j]))
            rhoMxL = rhoMxL.at[i, j].set(jnp.where(found, rhoBigNb, rhoMxL[i, j]))      # :151
            rhoKm1 = rhoKm1.at[i, j].set(jnp.where(found, rhoKm1[i, j], rhoLoc_k[i, j]))  # :153
            return (rhoKm1, rhoMxL, hMix), None

        (_, _, hMixLayer), _ = scan_k(
            iteration, (rhoKm1, rhoMxL, hMixLayer), range(kU, kL+1, kDir),
            lambda k: {"theta": level(state.theta, k), "salt": level(state.salt, k), "k": jnp.int32(k),
                       "rC_kmkDir": grid.rC[k-kDir], "drC_kpdeltaK": grid.drC[k+deltaK]})
    elif method == 2:                                                   # :159-218
        raise NotImplementedError("CALC_OCE_MXLAYER: method 2 (hMixCriteria > 1) is not ported")
    else:                                                               # :219-221
        raise ValueError("S/R CALC_OCE_MXLAYER: invalid method")

    if params.static_float("hMixSmooth") > 0.:                                   # :223-246
        raise NotImplementedError("CALC_OCE_MXLAYER: hMixSmooth > 0 is not ported")
    return hMixLayer
