"""Backward-only switches: what the derivative of a routine includes, as the original's adjoint decides it (plan
"Decisions 2026-10-02" item 3, revised; LESSONS_CARRIED L-AD-16: one implementation per switch, three tests; L-AD-17:
the option names what it cuts). A switch here never changes a forward value: it is read only inside a derivative rule
(`mitjax/ad/cg2d_rule.py`), so the forward program is the same with every setting.

CG2D derivative (`cg2d_operator_adjoint`)
------------------------------------------
TAF does not differentiate CG2D: its flow directives (pkg/autodiff/cg2d.flow:4-10 @63cdc0b) declare `ACTIVE = 1,2`
(cg2d_b, cg2d_x) and name the hand-written CG2D_MAD (pkg/autodiff/cg2d_mad.F) as the adjoint. CG2D_MAD solves the
adjoint system with the same CG2D (:116-133) and then, under `#ifdef ALLOW_AUTODIFF_TAMC` (:73, :239) and
`#if ( defined NONLIN_FRSURF || defined ALLOW_DEPTH_CONTROL )` (:175, :238):
    IF ( cg2dFullAdjoint ) THEN            (:181)  aC2d_ad, aW2d_ad, aS2d_ad += - cg2d_b_ad * cg2d_x * recip_cg2dNorm
                                                   (:186-204; the full adjoint: the operator's derivative)
    ELSE                                   (:220)  aW2d_ad = aS2d_ad = aC2d_ad = pW_ad = pS_ad = pC_ad = 0
                                                   (:223-236; the operator passive)
Without NONLIN_FRSURF or ALLOW_DEPTH_CONTROL the operator has no adjoint at all (CG2D.h is not even included,
:47-49). cg2dFullAdjoint is AUTODIFF_PARAMS.h's run-time parameter (data.autodiff, namelist AUTODIFF_PARM01; default
.FALSE., pkg/autodiff/autodiff_readparms.F:87), read into Cg2dParams.cg2dFullAdjoint by cg2d_h.ini_parms_cg2d.

The mitjax CG2D derivative is the implicit rule of cg2d_rule.cg2d_implicit. The option `Cg2dParams.mjx_cg2d_derivative`
(not a Fortran parameter; static) selects what it includes:
    "run"   (default) as the run's own TAF adjoint: the operator (aW2d, aS2d, aC2d) and the op_consts (cg2dNorm, the
            preconditioner pW, pS, pC) are active only when the build is a TAF build (ALLOW_AUTODIFF_TAMC), NONLIN_FRSURF
            or ALLOW_DEPTH_CONTROL is compiled and cg2dFullAdjoint = .TRUE.; otherwise passive (CG2D_MAD's ELSE branch).
            A build without pkg/autodiff (ALLOW_AUTODIFF undefined) has no TAF adjoint: the derivative is the exact one.
    "exact" always the full implicit derivative (operator and cg2dNorm active): the exact gradient of the converged
            solve. With cg2dFullAdjoint = .TRUE. this is what "run" gives (matched TAF's full CG2D_MAD to 14 digits on
            global_ocean.90x40x15/input_ad.bottomdrag); TAF's full branch treats cg2dNorm as a constant (recip_cg2dNorm,
            :176-179) and the preconditioner as passive (:213-215), which the implicit rule matches for the
            preconditioner (never differentiated) and exceeds for cg2dNorm (set once in INI_CG2D from the grid: no
            control reaches it in any ported run).
Where no control reaches the operator (linear free surface, nonlinFreeSurf <= 2 without selectImplicitDrag = 2:
forward_step.F:865-871 never calls UPDATE_CG2D) the two settings give the same gradient. Where one does (cs32x15/
input_ad: nonlinFreeSurf = 4, r*, cg2dFullAdjoint = .FALSE.), "run" is TAF's gradient and "exact" the exact one.

Tangent-linear model: TAF's tangent of CG2D is CG2D itself applied to the tangents (cg2d.flow:5 FTLNAME = cg2d, with
ACTIVE = 1,2), so TAF's TLM keeps the operator passive even with cg2dFullAdjoint = .TRUE., where its adjoint does not.
mitjax has one rule for both directions (the tangent and its transpose, cg2d_rule), so the TLM follows the same setting
as the adjoint; with cg2dFullAdjoint = .TRUE. and a moving operator (input_ad.bottomdrag) our TLM includes the
operator and TAF's does not.

SEAICE_LSR derivative (`lsr_derivative`; lane M4ADCS32ICE session 2)
--------------------------------------------------------------------
Plan decision 14 (option A1): where TAF tapes every executed LSR sweep (SEAICE_LSR_ADJOINT_ITER, lab_sea/code_ad) the
port differentiates the executed sweeps (mitjax/ad/lsr_sweeps.taped_sweeps). Plan decision 17: for
global_ocean.cs32x15/input_ad.seaice, whose build does not define SEAICE_LSR_ADJOINT_ITER (code_ad/SEAICE_OPTIONS.h:180;
TAF overwrites its per-sweep tape, lnkey = nlkey, seaice_lsr.F:785-788, :805-807: "the adjoint is necessarily wrong"),
the port's LSR derivative is A1 too -- by decision, not by the CPP flag. The static option
`SeaiceParams.mjx_lsr_derivative` (not a Fortran parameter; absent = "run") selects it:
    "run"     (default) as the build says: A1 with SEAICE_LSR_ADJOINT_ITER, otherwise the LSR is forward only (its
              derivative raises, mitjax/ad/seaice_lsr_rule.py): no gradient silently through an untaped solve;
    "sweeps"  A1 for any build: the DO m loop as the fixed-length scan of SOLV_MAX_FIXED sweeps with the Fortran's
              convergence flags; its forward is bit for bit the while_loop's (the same body on the same carry).
Refused (they would be silently wrong): "sweeps" when the loop bound SEAICElinearIterMax is not a host integer or
exceeds SOLV_MAX_FIXED (the scan would truncate the Fortran loop -- SEAICE_CHECK's bound, seaice_check.F:518-526, is
compiled only under ALLOW_AUTODIFF_TAMC, and a non-AD build's default is 1500, seaice_readparms.F:891); an unknown
option. Select explicitly: `model.replace(pkc={**model.pkc, "sp": with_lsr_derivative(model.pkc["sp"], "sweeps")})`.

Select explicitly: `with_cg2d_derivative(cg2d_params, "exact")`, e.g.
`adjoint.value_and_grad(model=adjoint.model.replace(cg2d_params=with_cg2d_derivative(adjoint.model.cg2d_params, "exact")))`.
CG2D_NSA (useNSACGSolver) is differentiated by TAF itself, iterations and operator alike (it has no flow directive);
its implicit rule (model/src/cg2d_nsa.py) stays exact and does not read this option.
"""

import dataclasses

import numpy as np

__all__ = ["CG2D_DERIVATIVE_OPTIONS", "cg2d_operator_adjoint", "with_cg2d_derivative",
           "LSR_DERIVATIVE_OPTIONS", "lsr_derivative", "with_lsr_derivative"]

CG2D_DERIVATIVE_OPTIONS = ("run", "exact")
_AD_HEADER = "AUTODIFF_OPTIONS.h"            # cg2d_mad.F:1 #include "AUTODIFF_OPTIONS.h"


def cg2d_operator_adjoint(cfg, cg2d_params):
    """True when the derivative of CG2D includes the operator's (aW2d, aS2d, aC2d and the op_consts), False when the
    operator is passive (module docstring). cfg: the build (cfg.cpp), cg2d_params: cg2d_h.Cg2dParams."""
    sel = cg2d_params.mjx_cg2d_derivative
    if sel not in CG2D_DERIVATIVE_OPTIONS:
        raise ValueError(f"mjx_cg2d_derivative = {sel!r}: one of {CG2D_DERIVATIVE_OPTIONS}")
    if sel == "exact":
        return True
    if not cfg.cpp.ALLOW_AUTODIFF:
        return True                          # no pkg/autodiff, no TAF adjoint: the exact derivative
    if not cfg.cpp.flag("ALLOW_AUTODIFF_TAMC", _AD_HEADER):
        raise NotImplementedError("CG2D derivative: ALLOW_AUTODIFF without ALLOW_AUTODIFF_TAMC (no CG2D_MAD) is not "
                                  "covered; select mjx_cg2d_derivative = 'exact' explicitly")
    nlfs_or_depth = (cfg.cpp.flag("NONLIN_FRSURF", _AD_HEADER)              # cg2d_mad.F:175
                     or cfg.cpp.flag("ALLOW_DEPTH_CONTROL", _AD_HEADER))
    return bool(nlfs_or_depth and cg2d_params.cg2dFullAdjoint)             # cg2d_mad.F:181 IF ( cg2dFullAdjoint )


def with_cg2d_derivative(cg2d_params, option):
    """cg2d_params (Cg2dParams) with the CG2D derivative option `option` ("run" or "exact")."""
    if option not in CG2D_DERIVATIVE_OPTIONS:
        raise ValueError(f"CG2D derivative option {option!r}: one of {CG2D_DERIVATIVE_OPTIONS}")
    return dataclasses.replace(cg2d_params, mjx_cg2d_derivative=option)


LSR_DERIVATIVE_OPTIONS = ("run", "sweeps")
_SEAICE_HEADER = "SEAICE_OPTIONS.h"


def lsr_derivative(cfg, sp):
    """"sweeps" (A1: SEAICE_LSR's DO m loop as mitjax/ad/lsr_sweeps.taped_sweeps) or "forward_only" (the
    while_loop, derivative refused) for the build `cfg` and the sea-ice parameters `sp` (module docstring)."""
    sel = getattr(sp, "mjx_lsr_derivative", "run")
    if sel not in LSR_DERIVATIVE_OPTIONS:
        raise ValueError(f"mjx_lsr_derivative = {sel!r}: one of {LSR_DERIVATIVE_OPTIONS}")
    if sel == "run" and not cfg.cpp.flag("SEAICE_LSR_ADJOINT_ITER", _SEAICE_HEADER):
        return "forward_only"
    from mitjax.ad.lsr_sweeps import SOLV_MAX_FIXED
    n = sp.SEAICElinearIterMax
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise RuntimeError(f"SEAICE_LSR derivative 'sweeps': SEAICElinearIterMax = {n!r} is not a host integer "
                           "(the scan length must bound the Fortran loop)")
    if n > SOLV_MAX_FIXED:                                                  # seaice_check.F:518-526
        raise RuntimeError(f"SEAICE_CHECK:SEAICElinearIterMax = {n} > SOLV_MAX_FIXED = {SOLV_MAX_FIXED}")
    return "sweeps"


def with_lsr_derivative(sp, option):
    """sp (SeaiceParams) with the SEAICE_LSR derivative option `option` ("run" or "sweeps")."""
    if option not in LSR_DERIVATIVE_OPTIONS:
        raise ValueError(f"SEAICE_LSR derivative option {option!r}: one of {LSR_DERIVATIVE_OPTIONS}")
    return sp.replace(mjx_lsr_derivative=option)
