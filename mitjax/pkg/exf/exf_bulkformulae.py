"""EXF_BULKFORMULAE: pkg/exf/exf_bulkformulae.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.libm import glibc_exp, powi
from mitjax.ops.safe import div, safe_sqrt
from mitjax.pkg.exf.exf_param_h import exf_half, exf_one, exf_two, karman, niter_bulk

PI = 3.14159265358979323844  # PARAMS.h:16  PARAMETER ( PI    = 3.14159265358979323844D0   )
halfRL = 0.5                 # eesupp/inc/EEPARAMS.h:73  PARAMETER ( ..., halfRL = 0.5 _d 0 )


def exf_bulkformulae(exf_Tsf, myTime, myIter, f, *, cfg, exf, params, state):
    """EXF_BULKFORMULAE( exf_Tsf, myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_bulkformulae.F:7-539

    C     Calculate bulk formula fluxes over open ocean (Large and Pond, JPO, 1981/82)

    `f` EXF_FIELDS.h (dict), `state` (uVel, vVel: useRelativeWind). Returns the new dict.
    Ported: ALLOW_BULKFORMULAE + ALLOW_ATM_TEMP with or without ALLOW_BULK_LARGEYEAGER04 (lane M4CS32ICE: the
    Large&Yeager04 arms :376-378, :389-392, :405-408, :420-424, :448-450, and solve4Stress = wspeedfile /= ' '
    with useAtmWind = .FALSE., :233-234), with or without ALLOW_DRAG_LARGEYEAGER09 and ALLOW_AUTODIFF (lane
    M4ADCOL; `wsm**6` is libgcc's __powidf2, powi), without EXF_CALC_ATMRHO (raises), EXF_READ_EVAP undefined;
    lane M4ADCS32ICE: solve4Stress = .FALSE. (useAtmWind = .FALSE. without ALLOW_BULK_LARGEYEAGER04, :236;
    global_ocean.cs32x15/code_ad): rdn = 0, ustar and tau from EXF_WIND's wStress (:324-333), psimh = 0 and no
    stress update in the iteration (:402-403, :415). With useAtmWind = .FALSE.
    the stress is not touched (:501-521).
    The point loops are independent per (i,j): vectorised. `IF ( atemp .NE. 0. )` (:283, :357, :481) is a pointwise
    `where`; on atemp = 0 lanes the discarded arm is computed with the divisions by t0 (= 0 there) and ustar**2
    guarded (`div` on a substituted 1.), so its derivative is finite (measured: unguarded, d/d(aqh) is NaN there). The locals tstar, qstar, ustar, tau, rdn, rd, delq, deltap start at
    0. (the Fortran leaves them unset; their values are read only on atemp /= 0 lanes, where they are written first),
    so that discarded lanes stay finite. SIGN(a,b) is gfortran's copysign (SIGN(0.5,-0.) = -0.5). EXP is libm's
    (glibc_exp); LOG, ATAN, SQRT are XLA's (bitwise glibc, mitjax/ops/libm.py MEASURED). REAL*4 literals `16.`
    exact; `.125 _d 0`, `0.5 _d 0` double."""
    for o in ("EXF_CALC_ATMRHO", "EXF_READ_EVAP"):
        if cfg.cpp.flag(o, "EXF_OPTIONS.h"):
            raise NotImplementedError(f"EXF_BULKFORMULAE: {o} is not ported")
    # lane M4ADCOL (1D_ocean_ice_column/code_ad): ALLOW_DRAG_LARGEYEAGER09 (:310-317, :435-442) and ALLOW_AUTODIFF
    # (deltap = delq = 0. _d 0 at every point before the IF, :277-280; the other code_ad-only lines are tamc.h and
    # the tape keys, :154, :225, :268, :274, :361-362, :484: no value)
    ly09 = cfg.cpp.flag("ALLOW_DRAG_LARGEYEAGER09", "EXF_OPTIONS.h")
    if not (cfg.cpp.flag("ALLOW_BULKFORMULAE", "EXF_OPTIONS.h") and cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h")):
        return f                                                               # :171-172
    ly04 = cfg.cpp.flag("ALLOW_BULK_LARGEYEAGER04", "EXF_OPTIONS.h")           # lane M4CS32ICE (cs32x15)
    if exf.useAtmWind:                                                         # :230-238
        solve4Stress = True
    elif ly04:                                                                 # :233-234
        solve4Stress = exf.wspeedfile.strip() != ""
    else:                                                                      # :236
        solve4Stress = False
    sz = cfg.size
    f = dict(f)
    if params.usingPCoords:                                                    # :240-246
        ks = sz.Nr
    else:
        ks = 1
    zwln = jnp.log(exf.hu/exf.zref)                                            # :249
    ztln = jnp.log(exf.ht/exf.zref)                                            # :250
    czol = exf.hu*karman*exf.gravity_mks                                       # :251
    recip_rhoConstFresh = 1.0/params.rhoConstFresh                             # :253
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    T = f["atemp"].data.shape[0]

    def local(name):
        return FArray(jnp.zeros((T, sz.sNy, sz.sNx)), name, i=(1, sz.sNx), j=(1, sz.sNy))
    tstar, qstar, ustar, tau = local("tstar"), local("qstar"), local("ustar"), local("tau")
    rdn, rd, delq, deltap = local("rdn"), local("rd"), local("delq"), local("deltap")

    if cfg.cpp.flag("ALLOW_AUTODIFF"):                                         # :277-280
        deltap = deltap.at[i, j].set(0.)                                       # :278  0. _d 0
        delq = delq.at[i, j].set(0.)                                           # :279  0. _d 0
    atemp = f["atemp"][i, j]
    aqh = f["aqh"][i, j]
    on = atemp != 0.                                                           # :283
    # :283-343 (atemp /= 0)
    Tsf = exf_Tsf[i, j]                                                        # :285
    tmpbulk = exf.cvapor_fac*glibc_exp(-exf.cvapor_exp/Tsf)                    # :286
    ssq = exf.saltsat*tmpbulk/exf.atmrho                                       # :293
    dp = atemp + exf.gamma_blk*exf.ht - Tsf                                    # :295
    dq = aqh - ssq                                                             # :296
    if exf.noNegativeEvap:                                                     # :298
        dq = MIN(0., dq, p="b")                                                # :298
    stable = exf_half + jnp.copysign(exf_half, dp)                             # :302
    wsm = f["sh"][i, j]                                                        # :309
    if ly09:                                                                   # :310-317 Large and Yeager (2009)
        tmpbulk = (exf.cdrag_1/wsm + exf.cdrag_2 + exf.cdrag_3*wsm             # :312-313
                   + exf.cdrag_8 * powi(wsm, 6))
        tmpbulk = exf.exf_scal_BulkCdn * (                                     # :314-317
            (halfRL - jnp.copysign(halfRL, wsm - exf.umax))*tmpbulk
            + (halfRL + jnp.copysign(halfRL, wsm - exf.umax))*exf.cdragMax)
    else:
        tmpbulk = exf.exf_scal_BulkCdn*(exf.cdrag_1/wsm + exf.cdrag_2 + exf.cdrag_3*wsm)   # :319-320
    if solve4Stress:                                                           # :304-323
        rdn_on = jnp.sqrt(tmpbulk)                                             # :322
        ustar_on = rdn_on*wsm                                                  # :323
        tau_on = None
    else:                                                                      # :324-333 (lane M4ADCS32ICE)
        rdn_on = 0.                                                            # :325  0. _d 0
        windStress = f["wStress"][i, j]                                        # :326
        # lane M4ADCS32ICE session 2: the two roots are guarded by EXF_WIND's own AD guard of the same root,
        # `IF ( wStress .LE. 0. ) ustar = 0.` ("check for zero wStress to please AD tools", exf_wind.F:194-199):
        # mask wStress > 0, fill 0. = SQRT(+0.) (EXF_WIND writes wStress = 0. _d 0 or SQRT(usSq) >= +0., :171,
        # :176), so the value is the Fortran's on every lane; the derivative at wStress = 0 is 0 instead of
        # 1/(2 SQRT(0)) = inf, which on the atemp = 0 lanes (the arm computed and discarded) gave 0*inf = NaN.
        # (On an atemp /= 0 lane wStress = 0 is singular in the Fortran itself: huol divides by ustar**2 = 0,
        # :373-375.)
        wpos = windStress > 0.
        ustar_on = safe_sqrt(windStress/exf.atmrho, wpos)                      # :331
        tau_on = safe_sqrt(windStress*exf.atmrho, wpos)                        # :333
    rhn = (exf_one-stable)*exf.cstanton_1 + stable*exf.cstanton_2              # :338
    ren = exf.cdalton                                                          # :339
    deltap = deltap.at[i, j].set(jnp.where(on, dp, deltap[i, j]))
    delq = delq.at[i, j].set(jnp.where(on, dq, delq[i, j]))
    rdn = rdn.at[i, j].set(jnp.where(on, rdn_on, 0.))                         # :350
    ustar = ustar.at[i, j].set(jnp.where(on, ustar_on, 0.))                   # :348
    tstar = tstar.at[i, j].set(jnp.where(on, rhn*deltap[i, j], 0.))           # :341, :346
    qstar = qstar.at[i, j].set(jnp.where(on, ren*delq[i, j], 0.))             # :342, :347
    tau = tau.at[i, j].set(jnp.where(on, tau[i, j] if tau_on is None else tau_on, 0.))   # :333, :349
    for it in range(1, niter_bulk + 1):                                        # :354-478
        t0 = atemp*(exf_one + exf.humid_fac*aqh)                               # :371-372
        us2 = ustar[i, j]*ustar[i, j]
        huol = div((div(tstar[i, j], jnp.where(on, t0, 1.0))                  # :373-375
                    + qstar[i, j]/(exf_one/exf.humid_fac+aqh))*czol,
                   jnp.where(on, us2, 1.0))
        if ly04:                                                               # :376-378 Large&Yeager04
            tmpbulk = MIN(jnp.abs(huol), 10.0, p="a")                          # :377
            huol = jnp.copysign(tmpbulk, huol)                                 # :378
        else:
            huol = MAX(huol, exf.zolmin, p="a")                                # :381
        htol = huol*exf.ht/exf.hu                                              # :383
        hqol = huol*exf.hq/exf.hu                                              # :384 (not used afterwards)
        del hqol
        stable = exf_half + jnp.copysign(exf_half, huol)                       # :385
        if solve4Stress:                                                       # :388-404
            if ly04:
                xsq = jnp.sqrt(jnp.abs(exf_one - huol*16.0))                   # :391
            else:
                xsq = MAX(jnp.sqrt(jnp.abs(exf_one - 16.*huol)), exf_one, p="a")   # :394
            x = jnp.sqrt(xsq)                                                  # :396
            psimh = (-exf.psim_fac*huol*stable                                 # :397-401
                     + (exf_one-stable)
                     * (jnp.log((exf_one + exf_two*x + xsq)*(exf_one+xsq)*.125)
                        - exf_two*jnp.arctan(x) + exf_half*PI))
        else:
            psimh = 0.
        if ly04:
            xsq = jnp.sqrt(jnp.abs(exf_one - htol*16.0))                       # :407
        else:
            xsq = MAX(jnp.sqrt(jnp.abs(exf_one - 16.*htol)), exf_one, p="a")   # :410
        psixh = (-exf.psim_fac*htol*stable + (exf_one-stable)                  # :412-413
                 * (exf_two*jnp.log(exf_half*(exf_one+xsq))))
        if solve4Stress:                                                       # :415-460
            if ly04:
                dzTmp = (zwln-psimh)/karman                                    # :422
                usn = f["wspeed"][i, j]/(exf_one + rdn[i, j]*dzTmp)            # :423
            else:
                usn = f["sh"][i, j]/(exf_one - rdn[i, j]/karman*psimh)         # :430
            usm = MAX(usn, exf.umin, p="a")                                    # :432
            if ly09:                                                           # :435-442 Large and Yeager (2009)
                tmpbulk = (exf.cdrag_1/usm + exf.cdrag_2 + exf.cdrag_3*usm     # :437-438
                           + exf.cdrag_8 * powi(usm, 6))
                tmpbulk = exf.exf_scal_BulkCdn * (                             # :439-442
                    (halfRL - jnp.copysign(halfRL, usm - exf.umax))*tmpbulk
                    + (halfRL + jnp.copysign(halfRL, usm - exf.umax))*exf.cdragMax)
            else:
                tmpbulk = exf.exf_scal_BulkCdn*(exf.cdrag_1/usm + exf.cdrag_2 + exf.cdrag_3*usm)   # :444-445
            rdn_n = jnp.sqrt(tmpbulk)                                          # :447
            if ly04:
                rd_n = rdn_n/(exf_one + rdn_n*dzTmp)                           # :449
            else:
                rd_n = rdn_n/(exf_one - rdn_n/karman*psimh)                    # :451
            ustar_n = rd_n*f["sh"][i, j]                                       # :453
            tau_n = exf.atmrho*rd_n*f["wspeed"][i, j]                          # :458
            rdn = rdn.at[i, j].set(jnp.where(on, rdn_n, rdn[i, j]))
            rd = rd.at[i, j].set(jnp.where(on, rd_n, rd[i, j]))
            ustar = ustar.at[i, j].set(jnp.where(on, ustar_n, ustar[i, j]))
            tau = tau.at[i, j].set(jnp.where(on, tau_n, tau[i, j]))
        rhn = (exf_one-stable)*exf.cstanton_1 + stable*exf.cstanton_2          # :463
        ren = exf.cdalton                                                      # :464
        rh = rhn/(exf_one + rhn*(ztln-psixh)/karman)                           # :467
        re = ren/(exf_one + ren*(ztln-psixh)/karman)                           # :468
        qstar = qstar.at[i, j].set(jnp.where(on, re*delq[i, j], qstar[i, j]))  # :471
        tstar = tstar.at[i, j].set(jnp.where(on, rh*deltap[i, j], tstar[i, j]))   # :472
    # :479-531 turbulent fluxes
    hs = exf.atmcp*tau[i, j]*tstar[i, j]                                       # :492
    hl = exf.flamb*tau[i, j]*qstar[i, j]                                       # :493
    evap = -recip_rhoConstFresh*tau[i, j]*qstar[i, j]                          # :497
    f["hs"] = f["hs"].at[i, j].set(jnp.where(on, hs, 0.))                      # :492, :524
    f["hl"] = f["hl"].at[i, j].set(jnp.where(on, hl, 0.))                      # :493, :525
    f["evap"] = f["evap"].at[i, j].set(jnp.where(on, evap, 0.))                # :497, :523
    f["hflux"] = f["hflux"].at[i, j].set(jnp.where(on, f["hflux"][i, j], 0.))  # :522
    if exf.useAtmWind and exf.useRelativeWind:                                 # :501-506
        tmpbulk = tau[i, j]*rd[i, j]
        us = tmpbulk*(f["uwind"][i, j]
                      - 0.5*(state.uVel[i, j, ks]+state.uVel[i+1, j, ks]))
        vs = tmpbulk*(f["vwind"][i, j]
                      - 0.5*(state.vVel[i, j, ks]+state.vVel[i, j+1, ks]))
    elif exf.useAtmWind:                                                       # :507-516
        tmpbulk = tau[i, j]*rd[i, j]                                           # :513
        us = tmpbulk*f["uwind"][i, j]                                          # :514
        vs = tmpbulk*f["vwind"][i, j]                                          # :515
    if exf.useAtmWind:                                                         # :520-521 (atemp = 0: 0.)
        f["ustress"] = f["ustress"].at[i, j].set(jnp.where(on, us, 0.))
        f["vstress"] = f["vstress"].at[i, j].set(jnp.where(on, vs, 0.))
    return f
