"""DWNSLP_APPLY: pkg/down_slope/dwnslp_apply.F @63cdc0b (lane M4ADLAB session 3)."""

import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.pkg.down_slope.dwnslp_init_varia import site_arrays


def dwnslp_apply(tracer, gTracer, recip_hFac, recip_rA, recip_drF, dws, *, cfg, usingZCoords):
    """DWNSLP_APPLY( trIdentity, bi, bj, kBottom, tracer, gTracer, recip_hFac, recip_rA_arg, recip_drF, deltaTLev,
    myTime, myIter, myThid )   @63cdc0b pkg/down_slope/dwnslp_apply.F:6-201, with kBottom = kLowC
    (temp_integrate.F:458-463, salt_integrate.F:456-461; p coordinates raise)

    C     | o Apply the tendency due to Down-Sloping flow (on tracer field "tracer")

    `tracer`, `gTracer`, `recip_hFac` FArrays [T, Nr, ny, nx], `recip_rA` [T, ny, nx], `recip_drF` [Nr], `dws` the
    down-slope state after DWNSLP_CALC_FLOW. Returns gTracer. onOffFlag (:90-95) is .TRUE.: DWNSLP_READPARMS refuses
    temp_/salt_useDWNSLP different from tempStepping / saltStepping. :137-168 for n = 1..NbSite with deepK /= 0, in
    order n (sites share columns: the additions to gTracer stay sequential, a Python loop over the padded site
    index, each site's update a `where` on its activity): dTrac(k) = tracer(ijd,k+1)-tracer(ijd,k) for k =
    kshelf..kDeep-1 and dTrac(kDeep) = tracer(ijs,kshelf)-tracer(ijd,kDeep) (:147-150, upward = -1); gTracer(ijd,k)
    += Transp*dTrac(k)*recip_drF(k)*recip_hFac(ijd,k)*recip_rA(ijd) for k = kshelf..kDeep (:152-158), then
    gTracer(ijs,kshelf) += Transp*(tracer(ijd,kshelf)-tracer(ijs,kshelf))*recip_drF(kshelf)*recip_hFac(ijs,kshelf)
    *recip_rA(ijs) (:160-165). The diagnostics and the log write no model variable (not carried)."""
    if not usingZCoords:                                                       # :99-100 upward = 1
        raise NotImplementedError("DWNSLP_APPLY: p coordinates are not ported")
    Nr = cfg.size.Nr
    shp = gTracer.data.shape
    flat = lambda a: a.reshape(a.shape[:-2] + (a.shape[-2]*a.shape[-1],))   # noqa: E731  (xySize, ...) view
    tr, g, rh = flat(tracer.data), flat(gTracer.data), flat(recip_hFac.data)  # [T, Nr, xySize]
    rA = flat(recip_rA.data)                                                   # [T, xySize]
    rdrF = jnp.asarray(recip_drF.data if hasattr(recip_drF, "data") else recip_drF)   # [Nr]
    dws = site_arrays(dws)
    T, NS = dws["ijDeep"].shape
    tix = jnp.arange(T)
    ks = jnp.arange(1, Nr + 1)[None, :]                                        # [1, Nr]
    for n in range(NS):                                                        # :137  DO n=1,DWNSLP_NbSite
        active = (n < dws["NbSite"]) & (dws["deepK"][:, n] != 0)               # :138
        ijd = dws["ijDeep"][:, n] - 1                                          # :141
        ijs = ijd + dws["shVsD"][:, n]                                         # :142
        kshelf = dws["kshelf"][:, n][:, None]                                  # :144  kBottom(ijs)
        kDeep = dws["deepK"][:, n][:, None]                                    # :145
        Transp = dws["Transp"][:, n][:, None]
        tr_d, tr_s = tr[tix, :, ijd], tr[tix, :, ijs]                          # [T, Nr]
        tr_d_up = jnp.concatenate([tr_d[:, 1:], tr_d[:, -1:]], axis=1)        # tracer(ijd,k-upward) = (ijd,k+1)
        tr_s_shelf = jnp.take_along_axis(tr_s, jnp.maximum(kshelf, 1) - 1, axis=1)   # MINMAX-RAW: index guard
        tr_d_deep = jnp.take_along_axis(tr_d, jnp.maximum(kDeep, 1) - 1, axis=1)     # MINMAX-RAW: index guard
        dTrac = jnp.where(ks < kDeep, tr_d_up - tr_d,                          # :147-149
                          jnp.where(ks == kDeep, tr_s_shelf - tr_d_deep, 0.))  # :150
        gTrLoc = (Transp                                                       # :153-156
                  * dTrac
                  * rdrF[None, :]*rh[tix, :, ijd]
                  * rA[tix, ijd][:, None])
        in_k = active[:, None] & (ks >= kshelf) & (ks <= kDeep)                # :152  k = kshelf..kDeep
        g_d = g[tix, :, ijd]
        g = g.at[tix, :, ijd].set(jnp.where(in_k, g_d + gTrLoc, g_d))          # :157
        ksh = jnp.maximum(kshelf, 1) - 1                                       # MINMAX-RAW: index guard
        tr_d_shelf = jnp.take_along_axis(tr_d, ksh, axis=1)
        gTrLoc0 = (Transp                                                      # :161-164
                   * (tr_d_shelf - tr_s_shelf)
                   * jnp.take(rdrF, ksh)*jnp.take_along_axis(rh[tix, :, ijs], ksh, axis=1)
                   * rA[tix, ijs][:, None])
        g_s = g[tix, :, ijs]
        new_s = jnp.where(active[:, None] & (ks == kshelf), g_s + gTrLoc0, g_s)   # :165
        g = g.at[tix, :, ijs].set(new_s)
    return FArray(g.reshape(shp), gTracer.name, tiled=gTracer.tiled, _dims=gTracer.dims)
