"""PTRACERS_PARAMS.h and PTRACERS_START.h: pkg/ptracers/PTRACERS_PARAMS.h, PTRACERS_START.h @63cdc0b, as one
parameter object `ptr` (the interface of PARAMS.h's `Params`, mitjax/model/src/ini_parms.py: `ptr.NAME` by Fortran
name).

Static (LOGICAL, INTEGER, CHARACTER: they select branches at trace time): PTRACERS_numInUse, and per tracer (tuples
indexed iTracer-1) PTRACERS_advScheme, PTRACERS_ImplVertAdv, PTRACERS_MultiDimAdv, PTRACERS_SOM_Advection,
PTRACERS_AdamsBashGtr, PTRACERS_AdamsBash_Tr, PTRACERS_useGMRedi, PTRACERS_useDWNSLP, PTRACERS_useKPP,
PTRACERS_linFSConserve, PTRACERS_stayPositive, PTRACERS_initialFile, PTRACERS_startAB (PTRACERS_START.h),
PTRACERS_StepFwd (PTRACERS_START.h, .TRUE. for every tracer when PTRACERS_startAllTrc), PTRACERS_doAB_onGpTr,
PTRACERS_addSrelax2EmP, PTRACERS_startAllTrc, PTRACERS_calcSurfCor, PTRACERS_Iter0, and the per-tracer flags
`PTRACERS_diffKh_ne_0`, `PTRACERS_diffK4_ne_0`, `PTRACERS_EvPrRn_ne_UNSET` (REAL values that select code: decided
once on the host, KERNEL_GUIDE §4; a gradient must not move them across the threshold).

Traced (REAL): PTRACERS_dTLev (FArray k=(1,Nr), untiled), per tracer (tuples) PTRACERS_diffKh, PTRACERS_diffK4,
PTRACERS_EvPrRn, PTRACERS_diffKrNr (FArray k=(1,Nr)), PTRACERS_ref (FArray k=(1,Nr)); lambdaTr1ClimRelax.
"""

import jax

from mitjax.model.src.ini_parms import Params


@jax.tree_util.register_pytree_node_class
class PtracersParams(Params):
    """PTRACERS_PARAMS.h + PTRACERS_START.h by Fortran name (static and traced values as in `Params`)."""

    def __getattr__(self, name):
        s, t = object.__getattribute__(self, "_s"), object.__getattribute__(self, "_t")
        if name in t:
            return t[name]
        if name in s:
            return s[name]
        raise AttributeError(f"PTRACERS_PARAMS.h {name} is not provided (mitjax/pkg/ptracers/ptracers_readparms.py)")

    def replace(self, static=None, traced=None):
        return PtracersParams({**self._s, **(static or {})}, {**self._t, **(traced or {})})
