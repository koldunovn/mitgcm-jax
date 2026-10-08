"""Helpers of the GOADK in-model gates (M2 Task 24): global_ocean.90x40x15/input_ad* through the driver `Model`
(mitjax/drivers/model.py), free-running from INITIALISE_VARIA, every dumped stage of the first steps compared with lane
A's dumps-on run of the variant (`jdon`, iterations 0-2; `ctrlxx`, the planted-control runs, iterations 0-3).

The step is the model's own FORWARD_STEP (`forward_step(probe=...)`): the probe records each stage's values as the
step passes it (the per-level DYNAMICS stages D00a / D00c with their level k), they come back from the jitted step and
are compared with the oracle's record of that stage on every point of every tile, halos included (element equality,
bit patterns, finite: r1_gate.compare_field). Nothing is teacher-forced: the carry of step n+1 is our own step n's.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.tests import grid_gate as gg
from mitjax.tests import r1_gate as rg

EXP = "global_ocean.90x40x15"
TENSOR = ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "GM_PsiX", "GM_PsiY")


@functools.lru_cache(maxsize=None)
def model(inp="input_ad", kind="jdon", xx=None):
    """The driver Model of the variant in the run directory of its dumps-on run (the oracle's input files), with
    the planted control of a `ctrlxx` run when kind = 'ctrlxx' (its own xx file, read as the Fortran reads it)."""
    from mitjax.config.params import load
    from mitjax.drivers.model import Model
    from mitjax.io.dump import DumpSet
    from mitjax.tests import goadk_gate as G
    e = load(EXP, inp)
    top = G.run_top(inp, kind)
    ds = DumpSet(top / "dumps")
    ctl = None
    if kind == "ctrlxx":
        sz = e.cfg.size
        dim, name, _ = G.CONTROL[inp]
        nz = sz.Nr if dim == 3 else 1
        raw = G.read_global(top / "rundir" / f"{name}.0000000000.data", nz, sz)
        ctl = {"genarr": {(dim, 1): G.xyz(name, raw, sz) if dim == 3 else G.xy(name, raw[:, 0], sz)}}
    m = Model(e, top / "rundir", xx=ctl)
    return m, ds


def step_fn(m, until=None):
    """jit(f)(arrays, carry, iloop, myTime, myIter) -> (carry, myTime, myIter, out, probes): one FORWARD_STEP of the
    Model with every probe recorded ({stage or "<stage>@<k>": values})."""
    from mitjax.model.src.forward_step import forward_step
    cfg, fp, pks = m.cfg, m.fp, m.pks

    def f(a, carry, iloop, myTime, myIter):
        probes = {}

        def probe(stage, values):
            key = f"{stage[0]}@{stage[1]}" if isinstance(stage, tuple) else stage   # per-level stages: "<stage>@k"
            probes[key] = dict(values) if isinstance(stage, tuple) else values
        state, ff, phi0surf = carry[:3]
        pk = carry[4] if len(carry) > 4 else None
        # the step's inputs the S00_begin / S02 / S03 / P01 records also hold (FFIELDS.h, phi0surf, the tensor) and
        # the GRID.h the step sees (G00_geometry: under NONLIN_FRSURF the State's hFac / recip_hFac)
        from mitjax.model.src.forward_step import nlfs_load
        g0, _ = nlfs_load(cfg=cfg, grid=a.grid, cg2dh=None, state=state)
        probes["_in"] = dict(ff=ff, phi0surf=phi0surf, gm=None if pk is None else pk.get("gm"), grid=g0)
        state, ff, phi0surf, myTime, myIter, out = forward_step(
            iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos, cg2dh=a.cg2dh,
            cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex, pk=pk, pkc=a.pkc,
            pks=pks, probe=probe, until=until)
        from mitjax.farray import FArray
        flow = tuple(FArray(fl.data, c.name, tiled=c.tiled, _dims=c.dims) for fl, c in zip(out["flow"], carry[3]))
        new = (state, ff, phi0surf, flow) + ((out["pk"],) if pk else ())
        return new, myTime, myIter, {k: v for k, v in out.items() if k not in ("flow", "pk")}, probes
    return jax.jit(f)


def stage_values(stage, v):
    """{name: FArray} of a probe value (State, FFields, GMREDI.h, (state, ff, phi0surf), cost.h, dict)."""
    if isinstance(stage, tuple):
        return dict(v)
    if stage == "S04_oceanic_phys":
        s_, f_, p_ = v
        vals = rg.state_fields(s_)
        vals.update({n: getattr(f_, n) for n in f_.names()})
        vals["phi0surf"] = p_
        return vals
    if stage in ("P05_gmredi_tensor", "P06_gmredi_exch"):
        return {n: getattr(v, n) for n in TENSOR if n in v}
    if hasattr(v, "names") and hasattr(v, "_f") and not hasattr(v, "loadedRec"):
        return rg.state_fields(v)
    if hasattr(v, "names"):
        return {n: getattr(v, n) for n in v.names()}
    if isinstance(v, dict):
        return v
    return {}


def compare_step(m, ds, it, probes, pairs=None):
    """{stage: {field: compare_field}} of every (stage, field) the oracle dumps at iteration `it` that the probes
    hold; the per-level D00a / D00c records `<name>_kNNN` against the level k probe."""
    out = {}
    keys = sorted({(s, n) for (_, s, n) in ds.keys(it)})
    for (s, n) in keys:
        if s.startswith("D00"):
            base, kk = n.rsplit("_k", 1)
            p = probes.get(f"{s}@{int(kk)}")
            if p is None or base not in p:
                continue
            v = p[base]
            if len(v.dims) == 3:
                lv = np.asarray(v.data)[:, int(kk) - v.dims[2][1]]
            else:
                lv = np.asarray(v.data)
            out.setdefault(s, {})[n] = rg.compare_field(lv, ds.field(it, s, n))
            continue
        inp = probes.get("_in", {})
        if s == "G00_geometry" and "grid" in inp:
            g = inp["grid"]
            if n in g and hasattr(getattr(g, n), "data"):
                out.setdefault(s, {})[n] = rg.compare_field(_as_record(getattr(g, n), ds.field(it, s, n)),
                                                            ds.field(it, s, n))
            continue
        src = "S16_blocking_exchanges" if s == "S17_monitor" else s     # S17: MONITOR reads S16's state
        if src not in probes:
            continue
        vals = dict(stage_values(s, probes[src]))
        if s == "S00_begin":                                             # the carry's FFIELDS.h, phi0surf, tensor
            vals.update({k: getattr(inp["ff"], k) for k in inp["ff"].names()})
            vals["phi0surf"] = inp["phi0surf"]
            if inp.get("gm") is not None:
                vals.update(stage_values("P05_gmredi_tensor", inp["gm"]))
        if s in ("S02_load_fields", "S03_ctrl_map_forcing", "P01_external_forcing_surf"):
            vals.setdefault("phi0surf", probes["S04_oceanic_phys"][2] if s == "P01_external_forcing_surf"
                            else inp.get("phi0surf"))
        if s == "S04_oceanic_phys" and "P06_gmredi_exch" in probes:     # DO_OCEANIC_PHYS' last tensor writer
            vals.update(stage_values("P06_gmredi_exch", probes["P06_gmredi_exch"]))
        if s == "S18_cost_tile":
            if hasattr(probes[s], n):
                recs = ds.tiles(it, s, n)
                ref = np.array([recs[t].data.ravel()[0] for t in sorted(recs)])
                ours = np.broadcast_to(np.asarray(getattr(probes[s], n), np.float64), ref.shape)
                out.setdefault(s, {})[n] = rg.compare_field(ours[:, None, None, None], ref[:, None, None, None])
            continue
        if n in vals:
            out.setdefault(s, {})[n] = rg.compare_field(np.asarray(vals[n].data), ds.field(it, s, n))
    return out


def _as_record(v, ref):
    """Our GRID.h array in the shape of its jaxdump record: a vertical (k) array as the kind-V record (one value per
    level at every point), a (j) array per tile broadcast along i, 2-D/3-D as they are."""
    a = np.asarray(v.data)
    axes = [d[0] for d in v.dims]
    if axes == ["k"]:
        return np.broadcast_to(a[None, :, None, None], ref.shape)
    if axes == ["j"]:
        return np.broadcast_to(a[:, None, :, None], ref.shape)
    return a


def bad(res):
    return {s: {n: v for n, v in r.items() if v[0] == "shape" or any(v[1:])} for s, r in res.items()
            if any(v[0] == "shape" or any(v[1:]) for v in r.values())}


def not_compared(ds, it, res):
    return sorted({(s, n) for (_, s, n) in ds.keys(it)} - {(s, n) for s, r in res.items() for n in r})


def run_free(m, ds, nsteps, until=None):
    """Steps 1..nsteps from the Model's initial carry: [(compare_step result, out)] per dumped iteration."""
    f = step_fn(m, until)
    carry = m.initial_carry()
    myTime, myIter = m.start_counters()
    its = set(ds.iterations())
    out = []
    for n in range(nsteps):
        carry, myTime, myIter, o, probes = f(m.arrays, carry, jnp.int32(n + 1), myTime, myIter)
        if n in its:
            out.append((n, compare_step(m, ds, n, probes), o))
    return out, carry
