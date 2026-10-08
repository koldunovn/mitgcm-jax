"""DWNSLP_INIT_VARIA: pkg/down_slope/dwnslp_init_varia.F @63cdc0b (lane M4ADLAB session 3)."""

import jax.numpy as jnp

from mitjax.farray import FArray


def dwnslp_init_varia(fixed, dp):
    """DWNSLP_INIT_VARIA( myThid )   @63cdc0b pkg/down_slope/dwnslp_init_varia.F:6-55

    C     | o Initialise Down-Sloping variables

    `fixed` DWNSLP_INIT_FIXED's tables, `dp` DwnslpParams. Returns the down-slope state the time loop carries
    (pk["dwnslp"]): DWNSLP_VARS.h deepK = 0 and Transp = 0. _d 0 (:44-49, DO n=1,DWNSLP_size: here the NS entries
    the kernels read), with the fixed site tables (NbSite, ijDeep, shVsD, Gamma, kshelf, kdeepMax) and DWNSLP_rec_mu
    (per tile, so every leaf has the tile axis first). Session 4: each leaf is a tiled FArray over a per-tile site
    axis (declared i = 1..n), so that a sharded run (P = N) carries each device's own tiles; the kernels read the
    arrays through `site_arrays`."""
    T, NS = fixed["ijDeep"].shape
    out = {n: jnp.asarray(fixed[n]) for n in ("NbSite", "ijDeep", "shVsD", "Gamma", "kshelf", "kdeepMax")}
    out["NbSite"] = out["NbSite"].reshape(T, 1)
    out["deepK"] = jnp.zeros((T, NS), jnp.int32)                              # :46
    out["Transp"] = jnp.zeros((T, NS), jnp.float64)                           # :47  0. _d 0
    out["DWNSLP_rec_mu"] = jnp.full((T, 1), dp.DWNSLP_rec_mu, jnp.float64)
    return {n: FArray(v, n, i=(1, v.shape[1])) for n, v in out.items()}


def site_arrays(dws):
    """The down-slope state as the kernels read it: {name: [tile, n] array} (NbSite [tile])."""
    out = {n: (v.data if isinstance(v, FArray) else v) for n, v in dws.items()}
    if out["NbSite"].ndim == 2:
        out["NbSite"] = out["NbSite"][:, 0]
    return out


def with_sites(dws, **new):
    """`dws` with the named leaves replaced by the [tile, n] arrays `new` (the FArray wrapping kept)."""
    out = dict(dws)
    for n, v in new.items():
        old = dws[n]
        out[n] = FArray(v, old.name, tiled=old.tiled, _dims=old.dims) if isinstance(old, FArray) else v
    return out
