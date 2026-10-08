"""PTRACERS_FIELDS.h: pkg/ptracers/PTRACERS_FIELDS.h @63cdc0b (the passive-tracer state) as a pytree.

    _RL  pTracer (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,PTRACERS_num)          PTRACERS_FIELDS.h:23-24
    _RL  gpTrNm1 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,PTRACERS_num)          PTRACERS_FIELDS.h:25-26
    _RL  surfaceForcingPTr (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,PTRACERS_num)   PTRACERS_FIELDS.h:27-28

The last (tracer) dimension is a Python tuple indexed by iTracer-1: `ptf.pTracer[iTracer-1]` is the FArray
`pTracer(:,:,:,:,:,iTracer)`. The SOM moments of PTRACERS_ALLOW_DYN_STATE (`_Ptracers_som`, PTRACERS_MOD.h) are
`som` (a tuple per tracer, None when the tracer has no SOM advection). totSurfCorPTr / meanSurfCorPTr
(PTRACERS_FIELDS.h:30-32) are used only with PTRACERS_linFSConserve, which raises.
"""

from dataclasses import dataclass, replace

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PtracersFields:
    pTracer: tuple
    gpTrNm1: tuple
    surfaceForcingPTr: tuple
    som: tuple = ()

    def replace(self, **kw):
        return replace(self, **kw)

    def set(self, name, iTracer, value):
        """A copy with field `name` of tracer iTracer (1-based) replaced."""
        t = list(getattr(self, name))
        t[iTracer - 1] = value
        return replace(self, **{name: tuple(t)})


def ptf_of_state(state, ptr):
    """PtracersFields of the State's per-tracer fields (mitjax/model/state.py PTRACERS_DECLARATIONS); `ptr`:
    PTRACERS_PARAMS.h (PTRACERS_num)."""
    rng = range(1, ptr.PTRACERS_num + 1)
    som = tuple(getattr(state, f"ptracers_som_{n:02d}") if f"ptracers_som_{n:02d}" in state else None for n in rng)
    return PtracersFields(pTracer=tuple(getattr(state, f"pTracer_{n:02d}") for n in rng),
                          gpTrNm1=tuple(getattr(state, f"gpTrNm1_{n:02d}") for n in rng),
                          surfaceForcingPTr=tuple(getattr(state, f"surfaceForcingPTr_{n:02d}") for n in rng),
                          som=som)


def _as(new, old):
    """`new`'s values under `old`'s name and declaration (FArray names are pytree metadata: a scan carry keeps them)."""
    from mitjax.farray import FArray
    if isinstance(old, tuple):
        return tuple(_as(a, b) for a, b in zip(new, old))
    return FArray(new.data, old.name, tiled=old.tiled, _dims=old.dims)


def state_with_ptf(state, ptf):
    """The State with the per-tracer fields of `ptf` (a None SOM entry leaves the State's field as it is); the
    State's field names are kept."""
    upd = {}
    for n0, (p, g, s) in enumerate(zip(ptf.pTracer, ptf.gpTrNm1, ptf.surfaceForcingPTr)):
        n = n0 + 1
        for key, val in ((f"pTracer_{n:02d}", p), (f"gpTrNm1_{n:02d}", g), (f"surfaceForcingPTr_{n:02d}", s)):
            upd[key] = _as(val, getattr(state, key))
        if n0 < len(ptf.som) and ptf.som[n0] is not None:
            key = f"ptracers_som_{n:02d}"
            upd[key] = _as(ptf.som[n0], getattr(state, key))
    return state.replace(**upd)
