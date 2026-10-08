"""OPPS_CALC and STATE1: pkg/opps/opps_calc.F @63cdc0b (OPPS_ORGCODE, the original NLOPPS, is #undef'd at :2).

OPPS_CALC works on one water column (OPPS_INTERFACE calls it inside its DO J / DO I loops). The port runs it on all
columns of the caller's loops at once: every column array of the Fortran (tracerEnv(Nr,nTracer), Pd, Dd, De, Wd, Md,
PlumeEntrainment, mda, wda, Pda, Paa, Ad of length Nr) is a [tile, k, j, i] array (tracers stacked in front), every
column scalar (dz1, dz2, oldflux, newflux, wsqr, radius, entrainrate, maxdepth, ntime, dt) a [tile, j, i] array. The
control flow of one column (loop exits, GOTO 1 / GOTO 1000, loop bounds kMax-1 and maxdepth) becomes per-column
masks: a column takes a statement exactly when the Fortran column would execute it, and keeps its values otherwise.
Columns are independent (OPPS_CALC reads and writes only its own column), so the vectorisation is exact.

Loops (literal order): the outer DO k=1,KMax-1 (:133-430) is a recursion (each k reads tracerEnv as the earlier k
left it): `scan_k` over k = 1..Nr-1, a column taking iteration k iff k <= kMax-1. The plume descent DO k2=k,KMax-1
(:177-288) and the CFL / mass-flux loop DO k2=k,maxDepth-1 (:328-354) are recursions in k2 (running scalars):
`scan_k` over k2 = 1..Nr-1 with the per-column range as a mask. Inside one time step `nn` the level updates (:371-412)
read only mda, Pda, Paa and their own tracerEnv level, so they are written for all levels at once. The time
integration DO nn=1,ntime (:369-416) has a data-dependent trip count per column (ntime, :334) and the Fortran has no
upper bound for it (ntime grows with the plume speed Wd; no PARAMETER, no structural bound): following the project
rule "fixed iteration counts, static config" (main-session decision 2026-10-02) it is a masked loop of a static
nTimeMax steps (`mitjax/pkg/opps/opps_h.py` NTIME_MAX: twice the largest ntime measured over every oracle column),
exact for every column with ntime <= nTimeMax. An overflow is never silent: such a column's tracers become NaN and
the per-tile count `ntimeOver` is returned for the host, where `opps_calc_host` STOPs the run (as CALC_R_STAR's
counters do).

Ported branches (vermix input.opps; tools/cpp_live.py): ALLOW_OPPS; ALLOW_OPPS_DEBUG blocks print only (no STDOUT in
the port: values are not affected); STATE1 in z coordinates with selectP_inEOS_Zc = 2 (MDJWF/JMD95P/UNESCO/TEOS10
default), <= 1, else-branch, and p coordinates; the ALLOW_NONHYDROSTATIC arm (selectP_inEOS_Zc = 3, :480-486) raises.

Literals: `.5` (:145, :197), `0.0` (:162, :203, :254), `0.5` (:334) are REAL*4 and exact; `0.5*int(dtts/dt)` (:334)
is REAL*4 arithmetic (the INTEGER converted to REAL*4) and NINT rounds half away from zero.
"""

import jax.numpy as jnp
import numpy as np
from jax import lax

from mitjax.model.src.find_rho import find_rho_scalar
from mitjax.ops.fortran_minmax import MIN
from mitjax.ops.safe import safe_div, safe_sqrt
from mitjax.ops.scan_k import scan_k

_INT_MIN = -2147483648.0    # gfortran INT() of a REAL*8 outside the INTEGER*4 range: cvttsd2si's 0x80000000


def state1_pressure(*, cfg, grid, params, state):
    """The pLoc of STATE1 (@63cdc0b pkg/opps/opps_calc.F:478-508) for every kRef = 1..Nr and every column at once:
    a [tile, k, j, i] array (the same expression per point as the Fortran's scalar statement)."""
    sz = cfg.size
    Nr = sz.Nr
    maskC = grid.maskC.data
    sel = params.selectP_inEOS_Zc
    if params.usingZCoords:                                             # :478
        if cfg.cpp.ALLOW_NONHYDROSTATIC and sel == 3:                   # :480-486
            raise NotImplementedError("STATE1: selectP_inEOS_Zc = 3 (non-hydrostatic pressure) is not ported")
        phiRef2k = jnp.asarray(params.phiRef.data)[np.arange(2, 2*Nr+1, 2) - 1]       # phiRef(2*kRef)
        if sel == 2:                                                    # :488-496
            return params.rhoConst*(state.totPhiHyd.data
                                    + phiRef2k[None, :, None, None]
                                    )*maskC
        if sel <= 1:                                                    # :500-501
            return jnp.asarray(params.pRef4EOS.data)[None, :, None, None]*maskC
        return params.rhoConst*phiRef2k[None, :, None, None]*maskC     # :502-503
    if params.usingPCoords:                                             # :505-507
        return jnp.asarray(grid.rC.data)[None, :, None, None]*maskC
    raise NotImplementedError("STATE1: neither z nor p coordinates")


def state1(sLoc, tLoc, pLoc, *, cfg, params, eos):
    """_RL FUNCTION STATE1( sLoc, tLoc, i, j, kRef, bi, bj, myThid )   @63cdc0b pkg/opps/opps_calc.F:440-515

    C     | o SUBROUTINE STATE1
    C     |   Calculates rho(S,T,p)
    C     |   It is absolutely necessary to compute
    C     |   the full rho and not sigma=rho-rhoConst, because
    C     |   density is used as a scale factor for fluxes and velocities

    Vectorised over columns: sLoc, tLoc, pLoc arrays of one shape; pLoc is the pressure of level kRef of each column
    (`state1_pressure`, :478-508, gathered at kRef by the caller). Returns FIND_RHO_SCALAR( tLoc, sLoc, pLoc ) (:510).
    """
    return find_rho_scalar(tLoc, sLoc, pLoc, cfg=cfg, params=params, eos=eos)


def _lev(a, k, axis):
    """a(..., k, ...): level k (Fortran index, traced int) of the level axis `axis`."""
    return lax.dynamic_index_in_dim(a, k-1, axis, keepdims=False)


def _set_lev(a, k, v, m, axis):
    """a(..., k, ...) = v on the columns where m, unchanged elsewhere."""
    return lax.dynamic_update_index_in_dim(a, jnp.where(m, v, _lev(a, k, axis)), k-1, axis)


def _ntime(dtts, dt):
    """:334-337  ntime = nint(0.5*int(dtts/dt)); if (ntime.eq.0) ntime = 1 (float64 holding the INTEGER value)."""
    q = dtts/dt
    n = jnp.where(jnp.abs(q) < 2147483648.0, jnp.trunc(q), _INT_MIN)   # INT(): INTEGER*4 truncation
    x = jnp.float32(0.5)*n.astype(jnp.float32)                          # REAL*4 0.5 times REAL(INT(..),4)
    x = x.astype(jnp.float64)
    r = jnp.where(x >= 0.0, jnp.floor(x + 0.5), -jnp.floor(-x + 0.5))  # NINT: half away from zero (exact on halves)
    return jnp.where(r == 0.0, 1.0, r)


def opps_calc(tracerEnv, OPPSconvectCount, wVel, kMax, nTracer, nTracerInUse, col, myTime, myIter, *,
              cfg, grid, params, eos, state, op, nTimeMax):
    """OPPS_CALC( tracerEnv, OPPSconvectCount, wVel, kMax, nTracer, nTracerInUse, I, J, bi, bj, myTime, myIter,
    myThid )   @63cdc0b pkg/opps/opps_calc.F:15-436

    C     | SUBROUTINE OPPS_CALC                                                |
    C     | o Compute all OPPS fields defined in OPPS.h                         |
    C     | This subroutine is based on the routine 3dconvection.F              |
    C     | by E. Skyllingstad (?)                                              |
    C     OPPSconvectCount :: counter for freqency of convection events

    Vectorised column routine (module docstring): `col` [tile, j, i] bool marks the columns (I, J) the caller's
    loops call OPPS_CALC for (the one addition to the Fortran signature, replacing I, J); tracerEnv: [nTracer, tile,
    k, j, i] (tracerEnv(K,ktr) of every column), OPPSconvectCount, wVel: [tile, k, j, i], kMax: [tile, j, i] integer
    values. Returns (tracerEnv, OPPSconvectCount, counters): tracers 1..nTracerInUse updated on the called columns,
    the count +1 at level k of each column for every plume that mixed (:421), and `counters` = {"ntimeOver": int32
    [tile]} for `opps_calc_host`. `state`: totPhiHyd for STATE1's pressure. nTimeMax: the static bound of the time
    integration (module docstring; the caller passes opps_h.NTIME_MAX).

    Guards: every division and SQRT is computed plainly on the lanes where the Fortran column executes it (so a
    non-finite Fortran value is reproduced) and guarded on the other lanes (finite derivatives there: no NaN may sit
    in a carried array, the reverse pass sums it into broadcast scalars such as dt). PlumeEntrainment(k2+1) =
    newflux/StartingFlux (:201) is guarded where StartingFlux = 0: a masked lane in the sense of mitjax/ops/safe.py
    (main-session decision 2026-10-02), since the plume stops at k2 = k there (:203-215) before the value is read.
    """
    sz = cfg.size
    Nr = sz.Nr
    nT = nTracerInUse
    drF = jnp.asarray(grid.drF.data)
    recip_drF = jnp.asarray(grid.recip_drF.data)
    dtts = jnp.asarray(params.dTtracerLev.data)[0]                      # :129  dtts = dTtracerLev(1)
    P = state1_pressure(cfg=cfg, grid=grid, params=params, state=state)
    lv = jnp.arange(1, Nr+1, dtype=jnp.int32)[None, :, None, None]     # Fortran level of each storage level
    kMax = jnp.asarray(kMax).astype(jnp.int32)
    tE = jnp.stack([tracerEnv[n] for n in range(nTracer)])              # [ntr, tile, k, j, i]
    zero3 = jnp.zeros_like(wVel)
    zeroT = jnp.zeros((nT,) + wVel.shape, wVel.dtype)
    zero2 = jnp.zeros_like(wVel[:, 0])
    rho = lambda s, t, p: state1(s, t, p, cfg=cfg, params=params, eos=eos)    # noqa: E731

    def outer(carry, x):                                                # DO k=1,KMax-1 (:133-430)
        tE, cnt, Pd, Dd, De, Wd, PE, Md, mda, wda, Pda, Paa, Ad, nOver = carry
        k = x["k"]
        act = col & (k <= kMax-1)
        for ktr in range(nT):                                           # :137-139
            Pd = Pd.at[ktr].set(_set_lev(Pd[ktr], k, _lev(tE[ktr], k, 1), act, 1))
        Ddk = rho(_lev(Pd[1], k, 1), _lev(Pd[0], k, 1), _lev(P, k, 1))  # :140
        Dd = _set_lev(Dd, k, Ddk, act, 1)
        De = _set_lev(De, k, Ddk, act, 1)                               # :141
        Wdk = -(.5*(_lev(wVel, k, 1)+_lev(wVel, k+1, 1)))               # :145
        Wd = _set_lev(Wd, k, Wdk, act, 1)
        wsqr = Wdk*Wdk                                                  # :161
        PE = _set_lev(PE, k, 0.0, act, 1)                               # :162
        radius = jnp.broadcast_to(op.PlumeRadius, Wdk.shape)            # :172
        StartingFlux = radius*radius*Wdk*Ddk                            # :173
        oldflux = StartingFlux                                          # :174
        dz2 = jnp.broadcast_to(_lev(drF, k, 0), Wdk.shape)              # :176

        def plume(c, y):                                                # DO k2=k,KMax-1 (:177-288)
            run, stopped, maxdepth, dz1, dz2, oldflux, wsqr, radius, Pd, Dd, De, Wd, PE = c
            k2 = y["k2"]
            inr = run & (k2 >= k) & (k2 <= kMax-1)
            Pk2 = _lev(P, k2+1, 1)
            D1 = rho(_lev(Pd[1], k2, 1), _lev(Pd[0], k2, 1), Pk2)                    # :178
            D2 = rho(_lev(tE[1], k2+1, 1), _lev(tE[0], k2+1, 1), Pk2)                 # :179-180
            De = _set_lev(De, k2+1, D2, inr, 1)                                       # :181
            go = inr & ((D2-D1 < op.STABILITY_THRESHOLD) | (k2 != k))                 # :190
            dz1n = dz2                                                                # :191
            dz2n = _lev(drF, k2+1, 0)                                                 # :192
            Wdk2, Ddk2 = _lev(Wd, k2, 1), _lev(Dd, k2, 1)
            newflux = oldflux+op.e2*radius*Wdk2*Ddk2*.5*(dz1n+dz2n)                   # :196-197
            # :201  PlumeEntrainment(k2+1) = newflux/StartingFlux where StartingFlux /= 0; masked lane (mitjax/ops/safe.py
            # rule): StartingFlux = 0 needs Wd(k) = 0, newflux = 0 then stops the plume at k2 = k (:203-215) before
            # PlumeEntrainment is read (:308), so the 0 here (the Fortran's 0/0) is never read
            PE = _set_lev(PE, k2+1, safe_div(newflux, StartingFlux, StartingFlux != 0.0), go, 1)
            stopF = go & (newflux <= 0.0)                                             # :203
            cont1 = go & ~(newflux <= 0.0)
            entrainrate = safe_div(newflux - oldflux, newflux, cont1 & (newflux != 0.0))          # :219
            for ktr in range(nT):                                                     # :224-229
                pmix = ((dz1n*_lev(tE[ktr], k2, 1)+dz2n*_lev(tE[ktr], k2+1, 1))
                        / (dz1n+dz2n))
                Pdk2 = _lev(Pd[ktr], k2, 1)
                Pd = Pd.at[ktr].set(_set_lev(Pd[ktr], k2+1, Pdk2
                                             - entrainrate*(pmix - Pdk2), cont1, 1))
            Dd1 = rho(_lev(Pd[1], k2+1, 1), _lev(Pd[0], k2+1, 1), Pk2)                # :234
            Dd = _set_lev(Dd, k2+1, Dd1, cont1, 1)
            De1 = _lev(De, k2+1, 1)
            Dek2 = _lev(De, k2, 1)
            wsqrn = (wsqr - wsqr*jnp.abs(entrainrate)+params.gravity                  # :248-250
                     * (safe_div(dz1n*(Ddk2-Dek2), Dek2, cont1 | (Dek2 != 0.0))
                        + safe_div(dz2n*(Dd1-De1), De1, cont1 | (De1 != 0.0))))
            stopW = cont1 & (wsqrn <= 0.0)                                            # :254
            cont2 = cont1 & ~(wsqrn <= 0.0)
            Wd = _set_lev(Wd, k2+1, safe_sqrt(wsqrn, cont2 & (wsqrn > 0.0),          # :277 (NaN on a NaN wsqr)
                                              fill=jnp.where(cont2, jnp.nan, 0.0)), cont2, 1)
            den = Wdk2*Ddk2
            q = safe_div(newflux, den, cont2 | (den != 0.0))
            radn = safe_sqrt(q, cont2 & (q > 0.0),                                    # :282
                             fill=jnp.where(cont2 & ~(q == 0.0), jnp.nan, q))
            stopC = inr & ~go                                                         # :283-287
            stop = stopF | stopW | stopC
            maxdepth = jnp.where(stop, k2, maxdepth)                                  # :212, :255, :284
            dz1 = jnp.where(go, dz1n, dz1)
            dz2 = jnp.where(go, dz2n, dz2)
            oldflux = jnp.where(cont1, newflux, oldflux)                              # :220
            wsqr = jnp.where(cont1, wsqrn, wsqr)
            radius = jnp.where(cont2, radn, radius)
            return (run & ~stop, stopped | stop, maxdepth, dz1, dz2, oldflux, wsqr, radius,
                    Pd, Dd, De, Wd, PE), None

        c0 = (act, jnp.zeros_like(act), jnp.zeros(act.shape, jnp.int32), zero2, dz2, oldflux, wsqr, radius,
              Pd, Dd, De, Wd, PE)
        c, _ = scan_k(plume, c0, range(1, Nr), lambda k2: {"k2": jnp.int32(k2)})
        _, stopped, maxdepth, _, _, _, _, _, Pd, Dd, De, Wd, PE = c
        maxdepth = jnp.where(act & ~stopped, kMax, maxdepth)            # :292  MaxDepth=kMax
        doMix = act & ~(stopped & (maxdepth == k))                      # :213, :274, :285  goto 1000
        md3 = maxdepth[:, None]

        Ad = _set_lev(Ad, k, jnp.broadcast_to(op.FRACTIONAL_AREA, act.shape), doMix, 1)   # :296
        for _ in range(op.MAX_ABE_ITERATIONS):                          # :301-417
            Mdk = _lev(Wd, k, 1)*_lev(Ad, k, 1)                         # :305
            Md = _set_lev(Md, k, Mdk, doMix, 1)
            mM = doMix[:, None] & (lv >= k+1) & (lv <= md3)             # :307-317
            Md = jnp.where(mM, Mdk[:, None]*PE, Md)

            def cfl(c, y):                                              # DO k2=k,maxDepth-1 (:328-354)
                dt, ntime, mda, wda, Pda, Paa = c
                k2 = y["k2"]
                inr = doMix & (k2 >= k) & (k2 <= maxdepth-1)
                Wdk2 = _lev(Wd, k2, 1)
                wnz = inr & (Wdk2 != 0.0)
                dt = jnp.where(wnz, MIN(dt, safe_div(_lev(drF, k2, 0), Wdk2, wnz), p="a"), dt)   # :329
                ntime = jnp.where(inr, _ntime(dtts, dt), ntime)                                  # :334-337
                d1, d2 = _lev(drF, k2, 0), _lev(drF, k2+1, 0)
                mda = _set_lev(mda, k2, (_lev(Md, k2, 1)*d1+_lev(Md, k2+1, 1)*d2)                # :343-344
                               / (d1+d2), inr, 1)
                wda = _set_lev(wda, k2, (Wdk2*d1+_lev(Wd, k2+1, 1)*d2)                           # :346-347
                               / (d1+d2), inr, 1)
                for ktr in range(nT):                                                            # :349-352
                    Pda = Pda.at[ktr].set(_set_lev(Pda[ktr], k2, _lev(Pd[ktr], k2, 1), inr, 1))
                    Paa = Paa.at[ktr].set(_set_lev(Paa[ktr], k2, _lev(tE[ktr], k2+1, 1), inr, 1))
                return (dt, ntime, mda, wda, Pda, Paa), None

            c0 = (jnp.broadcast_to(dtts, act.shape), jnp.zeros(act.shape), mda, wda, Pda, Paa)   # :327  dt = dtts
            (dt, ntime, mda, wda, Pda, Paa), _ = scan_k(cfl, c0, range(1, Nr), lambda k2: {"k2": jnp.int32(k2)})
            dt = jnp.where(doMix, MIN(dt, dtts, p="b"), dt)             # :355
            Pda = jnp.where(doMix[:, None] & (lv == md3), Pd, Pda)      # :364-366  Pda(maxdepth) = Pd(maxdepth)
            kmx = md3-1                                                 # :368
            top = doMix[:, None] & (lv == k)
            inner = doMix[:, None] & (lv > k) & (lv <= kmx)
            bot = doMix[:, None] & (lv == kmx+1)
            dt3 = dt[:, None]
            rdr = recip_drF[None, :, None, None]
            low = doMix[:, None] & (lv >= 1) & (lv <= kmx)

            def step(tE, Paa, on):                                      # one pass of DO nn (:371-412)
                on3 = on[:, None]
                tEn = []
                for ktr in range(nT):
                    F = mda*(Pda[ktr]-Paa[ktr])
                    Fm1 = jnp.concatenate([jnp.zeros_like(F[:, :1]), F[:, :-1]], axis=1)   # F at level l-1
                    t = tE[ktr]
                    t = jnp.where(on3 & top, t - F*dt3*rdr, t)                            # :373-376
                    t = jnp.where(on3 & inner, t + (Fm1 - F)*dt3*rdr, t)                   # :385-395
                    t = jnp.where(on3 & bot, t + Fm1*dt3*rdr, t)                           # :401-404
                    tEn.append(t)
                tE = tE.at[:nT].set(jnp.stack(tEn))
                above = jnp.concatenate([tE[:nT, :, 1:], tE[:nT, :, -1:]], axis=2)       # tracerEnv(l+1)
                Paa = jnp.where((on3 & low)[None], above, Paa)                            # :408-412
                return tE, Paa

            def bounded(c, y):                                          # DO nn=1,ntime (:369-416), masked
                tE, Paa = c
                return step(tE, Paa, doMix & (y["nn"] <= ntime)), None

            (tE, Paa), _ = scan_k(bounded, (tE, Paa), range(1, nTimeMax+1),
                                  lambda nn: {"nn": jnp.float64(nn)})
            over = doMix & (ntime > nTimeMax)                           # never silent: NaN here, STOP on the host
            tE = jnp.where(over[None, :, None], jnp.nan, tE)
            nOver = nOver + jnp.sum(over, axis=(1, 2), dtype=jnp.int32)
        cnt = jnp.where(doMix[:, None] & (lv == k), cnt + 1.0, cnt)    # :421
        return (tE, cnt, Pd, Dd, De, Wd, PE, Md, mda, wda, Pda, Paa, Ad, nOver), None

    carry = (tE, OPPSconvectCount, zeroT, zero3, zero3, zero3, zero3, zero3, zero3, zero3, zeroT, zeroT, zero3,
             jnp.zeros(wVel.shape[:1], jnp.int32))
    carry, _ = scan_k(outer, carry, range(1, Nr), lambda k: {"k": jnp.int32(k)})
    tE, cnt, nOver = carry[0], carry[1], carry[-1]
    return [tE[n] for n in range(nTracer)], cnt, {"ntimeOver": nOver}


def opps_calc_host(counters, myIter, nTimeMax):
    """The host-side check of the static time-loop bound of OPPS_CALC (main-session decision 2026-10-02: the masked
    loop with nTimeMax steps, an overflow never silent). `counters["ntimeOver"]`: per tile, the number of plumes
    (column, k) whose ntime (opps_calc.F:334-337) exceeded nTimeMax in this call; those columns hold NaN. Raises
    RuntimeError (the run STOPs) naming the bound and the iteration if any count is nonzero; returns None otherwise.
    Not Fortran output (the Fortran has no bound)."""
    n = np.asarray(counters["ntimeOver"])
    if int(n.sum()) > 0:
        tiles = [int(t) + 1 for t in np.nonzero(n)[0]]
        raise RuntimeError(f"MITJAX BOUND opps_calc.F:369 DO nn=1,ntime: ntime > nTimeMax = {nTimeMax} for "
                           f"{int(n.sum())} plumes (tiles {tiles}) at Iter={myIter:10d}\n"
                           "ABNORMAL END: OPPS_CALC time-loop bound (raise nTimeMax, mitjax/pkg/opps/opps_h.py)")
