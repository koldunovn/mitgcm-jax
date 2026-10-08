"""DWNSLP_CALC_FLOW: pkg/down_slope/dwnslp_calc_flow.F @63cdc0b (lane M4ADLAB session 3)."""

import jax.numpy as jnp

from mitjax.pkg.down_slope.dwnslp_init_varia import site_arrays, with_sites


def _flat(fld):
    """A [T, (Nr,) ny, nx] field as [T, (Nr,) xySize]: Fortran's (xySize, ...) view of the tile with overlap,
    ij = 1 + (i+OLx-1) + (j+OLy-1)*xSize at flat index ij-1."""
    d = fld.data
    return d.reshape(d.shape[:-2] + (d.shape[-2]*d.shape[-1],))


def dwnslp_calc_flow(rho3d, dws, *, cfg, params, usingPCoords):
    """DWNSLP_CALC_FLOW( bi, bj, kBottom, rho3d, myTime, myIter, myThid )
    @63cdc0b pkg/down_slope/dwnslp_calc_flow.F:6-171, with kBottom = kLowC (do_oceanic_phys.F:1055-1059; p
    coordinates raise)

    C     | o Detect active site of Down-Sloping flow and compute the flow rate

    `rho3d` rhoInSitu (FArray [T, Nr, ny, nx]), `dws` the down-slope state (pk["dwnslp"]: the DWNSLP_INIT_FIXED tables
    NbSite, ijDeep, shVsD, Gamma, kshelf = kBottom(ijd+ijr), kdeepMax = kBottom(ijd), and DWNSLP_VARS.h deepK,
    Transp; DWNSLP_rec_mu). Returns dws with deepK and Transp. :91-131 for n = 1..NbSite: deepK = 0; where
    `rho3d(ijs,kshelf+1) .GT. rho3d(ijd,kshelf+1) .AND. dRhoH .GT. 0` (dRhoH = rho3d(ijs,kshelf) - rho3d(ijd,kshelf),
    :101-102), deepK = the last k of kshelf+1..kBottom(ijd) with rho3d(ijs,k) > rho3d(ijd,k), else kshelf (:108-111),
    and Transp = Gamma*rec_mu*gravity*dRhoH*recip_rhoConst (:118-119); elsewhere Transp keeps its value (it is read
    only where deepK /= 0). The sites are independent: vectorised over n (padding entries n > NbSite keep their
    zeros). The diagnostics (ALLOW_DIAGNOSTICS with useDiagnostics) and the log (DWNSLP_ioUnit > 0, closed after
    DWNSLP_INIT_FIXED below debLevD: DWNSLP_READPARMS refuses debLevD) write no model variable."""
    if usingPCoords:                                                           # :72-74 downward = -1
        raise NotImplementedError("DWNSLP_CALC_FLOW: p coordinates are not ported")
    Nr = cfg.size.Nr
    rho = _flat(rho3d)                                                         # [T, Nr, xySize]
    dws0, dws = dws, site_arrays(dws)
    T, NS = dws["ijDeep"].shape
    tix = jnp.arange(T)[:, None]
    ijd = dws["ijDeep"] - 1                                                    # :97  (0-based flat index)
    ijs = ijd + dws["shVsD"]                                                   # :98
    kshelf = dws["kshelf"]                                                     # :99  kBottom(ijs)
    rho_s = jnp.moveaxis(rho[tix, :, ijs], -1, 1)                              # [T, Nr, NS] rho3d(ijs, k)
    rho_d = jnp.moveaxis(rho[tix, :, ijd], -1, 1)                              # [T, Nr, NS] rho3d(ijd, k)

    def lev(a, kk):                                                            # a(.., k) at a per-site level k
        return jnp.take_along_axis(a, (kk - 1)[:, None, :], axis=1)[:, 0]
    dRhoH = lev(rho_s, kshelf) - lev(rho_d, kshelf)                            # :101-102
    active = jnp.arange(NS)[None, :] < dws["NbSite"][:, None]                  # :91  n = 1..NbSite
    cond = active & (lev(rho_s, kshelf + 1) > lev(rho_d, kshelf + 1)) & (dRhoH > 0.)   # :104-105
    ks = jnp.arange(1, Nr + 1)[None, :, None]                                  # [1, Nr, 1]
    denser = ((ks >= (kshelf + 1)[:, None, :]) & (ks <= dws["kdeepMax"][:, None, :])   # :109-111
              & (rho_s > rho_d))
    kdeep = jnp.max(jnp.where(denser, ks, kshelf[:, None, :]), axis=1)     # :108-111  MINMAX-RAW: the last k (INTEGER)
    deepK = jnp.where(cond, kdeep, 0).astype(dws["deepK"].dtype)              # :92, :112
    Transp = jnp.where(cond, dws["Gamma"]                                      # :118-119
                       * dws["DWNSLP_rec_mu"]*params.gravity*dRhoH*params.recip_rhoConst, dws["Transp"])
    return with_sites(dws0, deepK=deepK, Transp=Transp)
