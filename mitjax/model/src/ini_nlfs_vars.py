"""INI_NLFS_VARS: model/src/ini_nlfs_vars.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_XY_RL
from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.safe import safe_div


def ini_nlfs_vars(state, *, cfg, grid, params, ex):
    """INI_NLFS_VARS( myThid )   @63cdc0b model/src/ini_nlfs_vars.F:7-190

    C     | SUBROUTINE INI_NLFS_VARS
    C     | o Initialise variables for Non-Linear Free-Surface
    C     |   formulations (formerly INI_SURF_DR & INI_R_STAR)

    :55-61 etaHnm1, dEtaHdt, PmEpR = 0. on every point. Under NONLIN_FRSURF: :65-75 hFac_surf*, hFac_surfNm1* = 0.,
    Rmin_surf = Ro_surf; :78-94 rStarFac*, pStarFacK, rStarFacNm1*, rStarExp* = 1., rStarDh*Dt = 0.; :148-180
    Rmin_surf on the interior from kSurfC/W/S, R_low, rF, drF and hFacInf (hFacInfMOM = hFacInf, :148), then
    EXCH_XY_RL( Rmin_surf ) (:180). The literals `0.` and `1.` are REAL*4 constants, exact.
    Not carried: :97-104 etaHw, etaHs, dEtaWdt, dEtaSdt (SURFACE.h /SIGMA_CHANGE/, read only with selectSigmaCoord
    != 0, which INI_PARMS refuses; mitjax/model/state.py). The ALLOW_AUTODIFF branch :106-138 (GOADK lane,
    global_ocean.90x40x15/code_ad: NONLIN_FRSURF with ALLOW_AUTODIFF): hFacC/W/S = h0FacC/W/S at every point, and
    recip_hFacC/W/S = oneRS / hFac where the mask is nonzero (the division guarded, the other points keep their
    value); the k loops write level k from level k only (k-vectorised)."""
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    out = {n: getattr(state, n).at[i, j].set(0.) for n in ("etaHnm1", "dEtaHdt", "PmEpR")}     # :55-61
    if cfg.cpp.NONLIN_FRSURF:
        for n in ("hFac_surfC", "hFac_surfW", "hFac_surfS", "hFac_surfNm1C", "hFac_surfNm1W", "hFac_surfNm1S"):
            out[n] = getattr(state, n).at[i, j].set(0.)                                    # :67-72
        Rmin_surf = state.Rmin_surf.at[i, j].set(grid.Ro_surf[i, j])                        # :73
        for n in ("rStarFacC", "rStarFacW", "rStarFacS", "pStarFacK", "rStarFacNm1C", "rStarFacNm1W",
                  "rStarFacNm1S", "rStarExpC", "rStarExpW", "rStarExpS"):
            out[n] = getattr(state, n).at[i, j].set(1.)                                    # :80-89
        for n in ("rStarDhCDt", "rStarDhWDt", "rStarDhSDt"):
            out[n] = getattr(state, n).at[i, j].set(0.)                                    # :90-92
        if cfg.cpp.ALLOW_AUTODIFF:                                  # :106-138 (GOADK lane: global_ocean code_ad)
            if cfg.cpp.USE_MASK_AND_NO_IF:                          # :118-124 (PTRACERS lane: the other arm raises)
                raise NotImplementedError("INI_NLFS_VARS: USE_MASK_AND_NO_IF (:118-124) is not ported")
            k3, j3, i3 = loops_kji((1, sz.Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
            for c in ("C", "W", "S"):                                                       # :108-116
                out["hFac" + c] = getattr(state, "hFac" + c).at[i3, j3, k3].set(
                    getattr(grid, "h0Fac" + c)[i3, j3, k3])
            for c in ("C", "W", "S"):                               # :117-137 (USE_MASK_AND_NO_IF undefined)
                h = out["hFac" + c][i3, j3, k3]
                wet = getattr(grid, "mask" + c)[i3, j3, k3] != 0.            # maskX(i,j,k,bi,bj).NE.zeroRS
                r = getattr(state, "recip_hFac" + c)
                out["recip_hFac" + c] = r.at[i3, j3, k3].set(
                    jnp.where(wet, safe_div(1., h, wet), r[i3, j3, k3]))     # oneRS / _hFacX(i,j,k,bi,bj)
        # :148-178
        hFacInf = params.init.hFacInf
        hFacInfMOM = hFacInf                                                                # :148
        Nr = sz.Nr
        j = loop_j(1, sNy)
        i = loop_i(1, sNx)
        ks = grid.kSurfC[i, j]                                                              # :156
        wet = ks <= Nr                                                                      # :157
        kc = jnp.clip(ks, 1, Nr)    # storage gather at a valid level (dry: unused)  MINMAX-RAW: index clamp, not a Fortran MAX/MIN
        rF_ks1 = grid.rF.data[kc]                                # rF(ks+1): storage index (ks+1) - 1
        drF_ks = grid.drF.data[kc - 1]                           # drF(ks)
        Rmin_tmp = rF_ks1                                                                   # :158
        Rmin_tmp = jnp.where(ks == grid.kSurfW[i, j], MAX(Rmin_tmp, grid.R_low[i-1, j], p="b"), Rmin_tmp)  # :159-160
        Rmin_tmp = jnp.where(ks == grid.kSurfW[i+1, j], MAX(Rmin_tmp, grid.R_low[i+1, j], p="b"), Rmin_tmp)  # :161-162
        Rmin_tmp = jnp.where(ks == grid.kSurfS[i, j], MAX(Rmin_tmp, grid.R_low[i, j-1], p="b"), Rmin_tmp)  # :163-164
        Rmin_tmp = jnp.where(ks == grid.kSurfS[i, j+1], MAX(Rmin_tmp, grid.R_low[i, j+1], p="b"), Rmin_tmp)  # :165-166
        new = MAX(MAX(rF_ks1, grid.R_low[i, j], p="a") + hFacInf*drF_ks,                      # :168-171
                  Rmin_tmp + hFacInfMOM*drF_ks, p="a")
        Rmin_surf = Rmin_surf.at[i, j].set(jnp.where(wet, new, Rmin_surf[i, j]))
        out["Rmin_surf"] = EXCH_XY_RL(Rmin_surf, ex=ex)                                     # :180
    return state.replace(**out)
