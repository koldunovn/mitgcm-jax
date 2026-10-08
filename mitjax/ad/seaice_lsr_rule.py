"""SEAICE_LSR forward only: the LSR solve of a build that does not tape its sweeps, with the build's own
derivative choice ("run"; lane M4OFF session 3, before plan decision 14).

Solver iterations are never differentiated (project rule, docs/PORTING_RULES.md §3), with one exception, plan
decision 14 (A1): the executed LSR sweeps as a fixed-length scan (mitjax/ad/lsr_sweeps.taped_sweeps), what TAF tapes
where the build defines SEAICE_LSR_ADJOINT_ITER, and the port's choice for every other build (master plan decision 17,
docs plan decision 12 a: the API's default lsr_derivative=None; mitjax/ad/modes.py "sweeps"). With "run" on a build
without SEAICE_LSR_ADJOINT_ITER (TAF overwrites its per-sweep tape there, pkg/seaice/seaice_lsr.F:805-807)
`seaice_lsr_forward_only` runs the literal forward (mitjax/pkg/seaice/seaice_lsr.py) and raises NotImplementedError
for any JVP, VJP or grad through it, so that no gradient is taken silently through an untaped solve: JAX would
otherwise differentiate the while_loop iterations in forward mode (jax.jvp of lax.while_loop is defined) and raise an
unrelated error in reverse mode."""

from functools import partial

import jax

from mitjax.pkg.seaice.seaice_lsr import seaice_lsr

MESSAGE = ("SEAICE_LSR: no derivative through the LSR sea-ice momentum solve with lsr_derivative='run' on a build "
           "without SEAICE_LSR_ADJOINT_ITER: the build does not tape the LSR sweeps (TAF's own adjoint is wrong there, "
           "pkg/seaice/seaice_lsr.F:805-807), so the solve is forward only. To differentiate the executed sweeps "
           "(plan decision 14, A1: the derivative TAF's tangent-linear model computes) pass lsr_derivative='sweeps' "
           "(or None, the API's default) to mitjax.load(...).gradient / grdchk, --lsr-derivative sweeps on the "
           "command line, or with_lsr_derivative(sp, 'sweeps') (mitjax/ad/modes.py) to the drivers.")


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _forward_only(fn, args):
    return fn(*args)


@_forward_only.defjvp
def _forward_only_jvp(fn, primals, tangents):
    raise NotImplementedError(MESSAGE)


def seaice_lsr_forward_only(myTime, myIter, sf, *, cfg, sp, op, grid, state, ex):
    """seaice_lsr(...) (mitjax/pkg/seaice/seaice_lsr.py), whose derivative raises NotImplementedError (MESSAGE).
    Every traced input is an argument of the custom_jvp function (none closed over), so a derivative with respect
    to any of them reaches the rule."""
    return _forward_only(_Static(cfg, ex), (myTime, myIter, sf, sp, op, grid, state))


class _Static:
    """The static part of the call (cfg, the exchanger), hashable by identity as custom_jvp's nondiff argument."""

    def __init__(self, cfg, ex):
        self.cfg, self.ex = cfg, ex

    def __call__(self, myTime, myIter, sf, sp, op, grid, state):
        return seaice_lsr(myTime, myIter, sf, cfg=self.cfg, sp=sp, op=op, grid=grid, state=state, ex=self.ex)
