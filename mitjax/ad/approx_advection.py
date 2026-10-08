"""useApproxAdvectionInAdMode as a backward-only switch (plan decision 17 revised, decision 11 pattern; lane
M4ADCS32ICE session 2).

What TAF does (63cdc0b). With `useApproxAdvectionInAdMode = .TRUE.` (data.autodiff, AUTODIFF_PARM01;
autodiff_readparms.F:89 default .FALSE.), GAD_ADVECTION (gad_advection.F:193-205, #ifdef ALLOW_AUTODIFF_TAMC),
GAD_CALC_RHS (gad_calc_rhs.F:176-184) and SEAICE_ADVECTION (seaice_advection.F:176-186, #ifdef ALLOW_AUTODIFF) replace
the scheme ENUM_DST3_FLUX_LIMIT (33) by ENUM_DST3 (30) (GAD.h:43-49: ENUM_DST3 = 30, ENUM_DST3_FLUX_LIMIT = 33)
`IF ( inAdMode .AND. useApproxAdvectionInAdMode )`. inAdMode is .TRUE. only inside the reverse sweep of a step
(ADAUTODIFF_INADMODE_SET, autodiff_inadmode_set_ad.F:53, the adjoint of forward_step.F:1208; TAF's tangent never sets
it). A routine's adjoint that recomputes its forward does so in AD mode; values restored from a global tape are those
of the taping sweep (forward mode). Hence two granularities, by where the routine keeps its intermediates:
- GAD_ADVECTION keeps every pass's localTij / af on LOCAL tapes (`CADJ INIT loctape_gad_adv_k_pass = COMMON`,
  gad_advection.F:286-291, stores :406-783): adGAD_ADVECTION recomputes the whole routine with scheme 30 from its
  inputs. Hook "routine": `gad_advection_call` = value GAD_ADVECTION(33), reverse derivative = that of
  GAD_ADVECTION(30) at the same inputs (horizontal and vertical scheme, :198-201).
- SEAICE_ADVECTION stores iceFld (:257) and each pass's localTij / af on the GLOBAL tapes comlev1_bibj_k_gadice(_pass)
  (:332-336, :365-366, :547-550, :579-580), written in the taping sweep with scheme 33: its adjoint linearises each
  flux at the forward-mode localTij with the AD-mode branch (GAD_DST3_ADV_X/Y). Hook "flux" (sea-ice default):
  `flux_call` around each flux call = value DST3FL, reverse derivative = DST3's at the same inputs. The other
  granularity ("routine") is selectable for the measurement of an open question: whether TAF's reverse sweep
  linearises downstream routines at stored forward values or at values recomputed in AD mode (not settled by the
  cs32x15.seaice grdchk points).
- GAD_CALC_RHS's swap acts only where it calls the advection itself (not with multiDimAdvection): no hook needed for
  a run whose tracers use multi-dimensional advection; `check_run` refuses the others.
The commented-out arms of GAD_DST3_ADV_X/Y/R (`c IF (inAdMode .AND. useApproxAdvectionInAdMode)`, smallNo the same
1.0D-20 either way) have no effect.

The static options (not Fortran parameters; absent = off, the forward's derivative):
    params.mjx_approx_advection_in_ad   None | "routine"            (GAD_ADVECTION; Params static)
    sp.mjx_approx_advection_in_ad       None | "flux" | "routine"   (SEAICE_ADVECTION; SeaiceParams static)
set together by drivers/ad_switches.with_run_switches from the run's data.autodiff. The value is never changed:
`backward_as` returns the forward function's output and only its VJP is the other scheme's, so the forward program
of every setting is the same function (fc bit for bit). Forward-mode differentiation (jax.jvp) of a hooked call is
not defined (custom_vjp): the TLM of TAF keeps scheme 33, i.e. our exact mode (switch off).
The transforms live here (mitjax/ad/), not in the physics modules (test_banned_transforms.py).
"""

import jax

from mitjax.pkg.generic_advdiff.gad_h import ENUM_DST3, ENUM_DST3_FLUX_LIMIT   # GAD.h:43-49: 30, 33
OPTION = "mjx_approx_advection_in_ad"
GAD_LEVELS = (None, "routine")
SEAICE_LEVELS = (None, "flux", "routine")


def ad_scheme(advectionSchArg):
    """The AD-mode scheme of gad_advection.F:198-201 / seaice_advection.F:181-182: 30 for 33, else unchanged."""
    return ENUM_DST3 if advectionSchArg == ENUM_DST3_FLUX_LIMIT else advectionSchArg


def backward_as(f_fwd, f_ad, *args):
    """f_fwd(*args), whose reverse-mode derivative is the VJP of f_ad at the same args (the recomputation in AD mode).
    Every traced value the two functions read must be in `args` (custom_vjp does not differentiate closed-over
    tracers); f_fwd and f_ad close over static values only."""
    @jax.custom_vjp
    def g(*a):
        return f_fwd(*a)

    def g_fwd(*a):
        return f_fwd(*a), a

    def g_bwd(a, ct):
        _, pull = jax.vjp(f_ad, *a)
        return pull(ct)

    g.defvjp(g_fwd, g_bwd)
    return g(*args)


def gad_level(params):
    lev = params.static_items().get(OPTION)
    if lev not in GAD_LEVELS:
        raise ValueError(f"params.{OPTION} = {lev!r}: one of {GAD_LEVELS}")
    return lev


def seaice_level(sp):
    lev = getattr(sp, OPTION, None)
    if lev not in SEAICE_LEVELS:
        raise ValueError(f"sp.{OPTION} = {lev!r}: one of {SEAICE_LEVELS}")
    return lev


def gad_advection_call(implicitAdvection, advectionSchArg, vertAdvecSchArg, trIdentity, deltaTLev, uFld, vFld, wFld,
                       tracer, gTracer, myTime, myIter, *, cfg, grid, params):
    """GAD_ADVECTION (mitjax/pkg/generic_advdiff/gad_advection.py) as the caller's CALL; with
    params.mjx_approx_advection_in_ad = "routine" and a scheme 33 (horizontal or vertical) its reverse derivative is
    that of GAD_ADVECTION with the AD-mode schemes (module docstring)."""
    from mitjax.pkg.generic_advdiff.gad_advection import gad_advection
    lev = gad_level(params)
    sch_ad, vsch_ad = ad_scheme(advectionSchArg), ad_scheme(vertAdvecSchArg)
    if lev is None or (sch_ad, vsch_ad) == (advectionSchArg, vertAdvecSchArg):
        return gad_advection(implicitAdvection, advectionSchArg, vertAdvecSchArg, trIdentity, deltaTLev, uFld, vFld,
                             wFld, tracer, gTracer, myTime, myIter, cfg=cfg, grid=grid, params=params)

    def run(sch, vsch):
        def f(dTL, u, v, w, tr, gT, t, it, g, p):
            return gad_advection(implicitAdvection, sch, vsch, trIdentity, dTL, u, v, w, tr, gT, t, it, cfg=cfg,
                                 grid=g, params=p)
        return f
    return backward_as(run(advectionSchArg, vertAdvecSchArg), run(sch_ad, vsch_ad),
                       deltaTLev, uFld, vFld, wFld, tracer, gTracer, myTime, myIter, grid, params)


def flux_call(flux, advectionScheme, sch_ad, deltaT, *args):
    """One flux call `flux(scheme, deltaT, *args)` of SEAICE_ADVECTION (its _adv_x / _adv_y): with sch_ad differing
    from advectionScheme, value of `advectionScheme`, reverse derivative of `sch_ad` at the same inputs (hook
    "flux")."""
    if sch_ad is None or sch_ad == advectionScheme:
        return flux(advectionScheme, deltaT, *args)
    return backward_as(lambda *a: flux(advectionScheme, *a), lambda *a: flux(sch_ad, *a), deltaT, *args)


def seaice_advection_call(tracerIdentity, advectionSchArg, uFld, vFld, uTrans, vTrans, iceFld, r_hFld, gFld, afx,
                          afy, myTime, myIter, *, cfg, sp, grid, SIMaskU, SIMaskV, useCubedSphereExchange):
    """SEAICE_ADVECTION (mitjax/pkg/seaice/seaice_advection.py) as SEAICE_ADVDIFF's CALL; with
    sp.mjx_approx_advection_in_ad = "routine" and scheme 33 its reverse derivative is that of the whole routine with
    scheme 30 (the O1 alternative; "flux" is applied inside the routine per flux call)."""
    from mitjax.pkg.seaice.seaice_advection import seaice_advection
    lev = seaice_level(sp)
    if lev != "routine" or ad_scheme(advectionSchArg) == advectionSchArg:
        return seaice_advection(tracerIdentity, advectionSchArg, uFld, vFld, uTrans, vTrans, iceFld, r_hFld, gFld,
                                afx, afy, myTime, myIter, cfg=cfg, sp=sp, grid=grid, SIMaskU=SIMaskU,
                                SIMaskV=SIMaskV, useCubedSphereExchange=useCubedSphereExchange)

    def run(sch):
        def f(u, v, uT, vT, ice, rh, gF, ax, ay, t, it, p, g, mU, mV):
            return seaice_advection(tracerIdentity, sch, u, v, uT, vT, ice, rh, gF, ax, ay, t, it, cfg=cfg, sp=p,
                                    grid=g, SIMaskU=mU, SIMaskV=mV, useCubedSphereExchange=useCubedSphereExchange)
        return f
    return backward_as(run(advectionSchArg), run(ad_scheme(advectionSchArg)), uFld, vFld, uTrans, vTrans, iceFld,
                       r_hFld, gFld, afx, afy, myTime, myIter, sp, grid, SIMaskU, SIMaskV)
