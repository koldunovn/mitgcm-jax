"""GAD_FLUX_LIMITER.h of pkg/generic_advdiff: the statement function Limiter(Cr) that GAD_FLUXLIMIT_ADV_X/Y/R and
GAD_FLUXLIMIT_IMPL_R include (@63cdc0b pkg/generic_advdiff/GAD_FLUX_LIMITER.h:1-42).

C !DESCRIPTION:
C Contains statement function defining limiter function.
C ...
C The current limiter of choice is the "Superbee" limiter:
C Limiter(Cr)=max(0,max(min(1,2*Cr),min(2,Cr)))
C which is the default.

The live line is :41-42 (Superbee); the Upwind, Lax-Wendroff and Min-Mod alternatives at :38-40 are commented out.
No M1 experiment overrides the header (verification/{advect_xz,advect_xy,global_ocean.90x40x15,
tutorial_baroclinic_gyre,tutorial_global_oce_optim}/code* hold no GAD_FLUX_LIMITER.h). `0.D0`, `1.D0`, `2.D0` are
double literals. max/min are `mitjax.ops.fortran_minmax.MAX/MIN`: gfortran's value on ties of +0/-0 and NaN, JAX's
own derivative (subgradient at a switch [L-AD-7], [L-CONF-12]). gfortran expands the statement function inline at
each use, and which argument wins ties and NaN differs between the including routines (the oracle's compilation,
$MJX_REFERENCE/minmax_sites: MAX(0.D0, ...) keeps the second argument in GAD_FLUXLIMIT_ADV_X/Y/R and the first in
GAD_FLUXLIMIT_IMPL_R), so each caller passes the winners of its own expansion as `p`.
"""

from mitjax.ops.fortran_minmax import MAX, MIN


# the MAX/MIN sites of one expansion of Limiter, in the order gfortran compiles them (`p` of Limiter)
LIMITER_SITES = ("MIN", "MIN", "MAX", "MAX")     # min(1.D0,2.D0*Cr), min(2.D0,Cr), max(min,min), max(0.D0,max)


def Limiter(Cr, *, p):
    """GAD_FLUX_LIMITER.h:32, 41-42
          _RL Limiter
          Limiter(Cr)=max(0.D0,max(min(1.D0,2.D0*Cr),
         &                         min(2.D0,Cr)))
    p: the winners of the four sites of the caller's expansion (LIMITER_SITES order), from the oracle table at the
    caller's statement."""
    p1, p2, p3, p4 = p
    return MAX(0., MAX(MIN(1., 2.*Cr, p=p1),
                       MIN(2., Cr, p=p2), p=p3), p=p4)
