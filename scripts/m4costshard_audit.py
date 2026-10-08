#!/usr/bin/env python3
"""Audit of the cost path under shard_map (lane M4COSTSHARD; measurement script): for global_ocean.cs32x15 input_ad,
input_ad.seaice and input_ad.seaice_dynmix (the code_ad build: COST_TEST) at P devices (fake CPU devices, default 6):

1. every leaf of the carried package state (carry[4]: cost.h / SEAICE_COST.h CostCommon, CTRL_GENARR.h, GMREDI.h, ...)
   with its shape and how drivers/sharded_grad places it (tile_specs: tiled FArrays sharded, the rest replicated);
2. a P-device trace (jax.jit(...).trace, no compile) of CTRL_MAP_INI_GENARR + one time step: per cost leaf the local
   shape and its sharding type (varying = built from the local tiles, invariant = replicated or gathered) after the
   step, and whether the step wrote it;
3. P-device traces of the whole objective J (map, the loop, COST_FINAL) with three final costs: "old" (GenarrAdjoint's
   until f2fa646: Model.cost_final with the set-up's maskC and the local State), "maskC" (the local traced maskC,
   lane M4ADCS32ICE's sharded-gradient study at 4ac8b87), "gathered" (adjoint_run.final_cost_gathered, this lane's
   fix): trace
   ok or the error; for every trace the constants of the jaxpr with a leading tile axis of nTiles (closures over a
   global [tile] array) and, for "gathered", the cost leaves after COST_FINAL (shape, varying or not).

    m4costshard_audit.py OUT_DIR VARIANT [P]       VARIANT ad | seaice | dynmix
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

from mitjax.xla_flags import gate_xla_flags

NP = int(sys.argv[3]) if len(sys.argv) > 3 else 6
os.environ["XLA_FLAGS"] = gate_xla_flags("").replace("device_count=4", f"device_count={max(NP, 4)}")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import PartitionSpec  # noqa: E402

OUT, VAR = Path(sys.argv[1]), sys.argv[2]
INP = {"ad": "input_ad", "seaice": "input_ad.seaice", "dynmix": "input_ad.seaice_dynmix"}[VAR]
EXP = ("global_ocean.cs32x15", INP)
if (OUT / "audit.json").exists():
    raise SystemExit(f"{OUT}/audit.json exists: choose a new OUT_DIR")
OUT.mkdir(parents=True, exist_ok=True)

from mitjax import paths  # noqa: E402
from mitjax.drivers import adjoint_run as AR  # noqa: E402
from mitjax.drivers.checkpoint import SAVE_NAMES, integrate  # noqa: E402
from mitjax.drivers.model import Model  # noqa: E402
from mitjax.drivers.run import load_experiment, make_rundir  # noqa: E402
from mitjax.drivers.sharded_grad import place_model, tile_specs  # noqa: E402
from mitjax.eesupp import exch_maps as EM  # noqa: E402
from mitjax.eesupp.shard import TileSharding  # noqa: E402
from mitjax.farray import FArray  # noqa: E402
from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr  # noqa: E402

R = {"variant": VAR, "P": NP, "t": {}}


def save():
    (OUT / "audit.json").write_text(json.dumps(R, indent=1, default=str))


def is_node(x):
    return isinstance(x, FArray)


def leaves(tree):
    """[(path, leaf)] with FArrays as leaves."""
    return jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_node)[0]


def pstr(path):
    return jax.tree_util.keystr(path)


def vma(a):
    """The mesh axes the value varies over (empty: invariant = replicated or gathered), as ShardedExchanger.vary
    reads the type."""
    d = a.data if isinstance(a, FArray) else a
    t = jax.typeof(d)
    return sorted(getattr(getattr(t, "mat", None), "varying", None) or getattr(t, "vma", None) or frozenset())


def shape(a):
    return list(np.shape(a.data if isinstance(a, FArray) else a))


def tile_consts(closed, nT):
    """Constants with a leading axis of nTiles anywhere in a ClosedJaxpr (closures over a global [tile] array)."""
    from collections import Counter
    found = Counter()

    def note(kind, c):
        s = np.shape(c)
        if len(s) >= 1 and s[0] == nT:
            found[f"{kind} {list(s)} {np.result_type(c)}"] += 1

    def walk(cj):
        jx = cj.jaxpr if hasattr(cj, "jaxpr") else cj
        for c in getattr(cj, "consts", []):
            note("const", c)
        for e in jx.eqns:
            for v in e.invars:
                if hasattr(v, "val"):
                    note("literal", v.val)
            for p in e.params.values():
                for q in (p if isinstance(p, (list, tuple)) else [p]):
                    if hasattr(q, "eqns") or hasattr(q, "jaxpr"):
                        walk(q)
    walk(closed)
    return dict(found)


t0 = time.time()
exp_dir = paths.UPSTREAM / "verification" / EXP[0]
m = Model(load_experiment(exp_dir, INP), make_rundir(exp_dir, INP, OUT / "run"))
key = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
a = AR.GenarrAdjoint(m, key=key)
model, st0, xs, step = a.model, a.st0, a.xs, a._stepfn
nT = m.cfg.size.nSx * m.cfg.size.nSy * m.cfg.size.nPx * m.cfg.size.nPy
R["nTiles"] = nT
R["t"]["setup"] = time.time() - t0

# 1. the carried package state and its placement
sh = TileSharding(EM.load_cube_maps(*EXP), NP)
pk0 = st0[0][4]
specs = tile_specs(sh, pk0)
R["carry_pk"] = [dict(path=pstr(p), shape=shape(x), tiled_farray=isinstance(x, FArray) and x.tiled,
                      placed="sharded" if s == sh.TILES else "replicated",
                      per_tile_plain=(not isinstance(x, FArray)) and len(np.shape(x)) >= 1 and np.shape(x)[0] == nT)
                 for (p, x), s in zip(leaves(pk0), jax.tree.leaves(specs, is_leaf=lambda v: isinstance(v, PartitionSpec)))]
R["model_pkc_cost_fixed"] = [dict(path=pstr(p), shape=shape(x), tiled_farray=isinstance(x, FArray) and x.tiled)
                             for p, x in leaves(model.pkc.get("cost_fixed"))]
save()

th4 = sh.put_tree(a.theta_farray())
mp, s04 = place_model(sh, model), sh.put_tree(st0)
th_s, mo_s, st_s = tile_specs(sh, a.theta_farray()), tile_specs(sh, model), tile_specs(sh, st0)

# 2. map + one step, per cost leaf
rec = {}


def body_step(th, mo, s, x):
    mo2, s1 = AR.genarr_apply(m, key, th.data, mo, s)
    s2 = step(mo2, s1, x[0])
    before = {pstr(p): v for p, v in leaves(s1[0][4]["cost"])}
    rec["cost"] = [dict(path=pstr(p), shape=shape(v), vma=vma(v), written=before[pstr(p)] is not v)
                   for p, v in leaves(s2[0][4]["cost"])]
    rec["pk_other"] = [dict(path=pstr(p), shape=shape(v), vma=vma(v))
                       for p, v in leaves({k: w for k, w in s2[0][4].items() if k != "cost"})
                       if len(shape(v)) >= 1 and shape(v)[0] == nT]
    return jnp.float64(0.0)


t1 = time.time()
try:
    f = sh.shard_map(body_step, in_specs=(th_s, mo_s, st_s, sh.REP), out_specs=sh.REP)
    tr = f.trace(th4, mp, s04, xs)
    R["step"] = dict(ok=True, **rec, tile_consts=tile_consts(tr.jaxpr, nT))
except Exception as e:                                              # noqa: BLE001 -- a measurement
    R["step"] = dict(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:400]}",
                     where=traceback.format_exc().splitlines()[-6:])
R["t"]["step_trace"] = time.time() - t1
save()


# 3. the whole objective with three final costs
def fc_old(mo, st):
    pk = st[0][4]
    cost, _ = m.cost_final(pk["cost"], pk.get("genarr"), fixed=mo.pkc["cost_fixed"], state=st[0][0],
                           ecco=pk.get("ecco"))
    return cost.fc


def fc_maskc(mo, st):
    pk = st[0][4]
    cost, _ = m.cost_final(pk["cost"], pk.get("genarr"), fixed=mo.pkc["cost_fixed"], maskC=mo.grid.maskC,
                           state=st[0][0], ecco=pk.get("ecco"))
    return cost.fc


fin = {}


def fc_gathered(mo, st):
    # final_cost_gathered with the cost leaves after COST_FINAL recorded
    ex = mo.ex
    pk = st[0][4]
    g = AR._gather
    cost, _ = m.cost_final(g(pk["cost"], ex), g(pk.get("genarr"), ex), fixed=g(mo.pkc["cost_fixed"], ex),
                           maskC=g(mo.grid.maskC, ex), state=g(st[0][0], ex), ecco=g(pk.get("ecco"), ex))
    fin["cost_after_final"] = [dict(path=pstr(p), shape=shape(v), vma=vma(v)) for p, v in leaves(cost)
                               if not (isinstance(v, FArray) and v.tiled)]
    ref = AR.final_cost_gathered(m, mo, st)                      # the driver's function (same value)
    return cost.fc + 0.0 * ref


for name, fc in (("old", fc_old), ("maskC", fc_maskc), ("gathered", fc_gathered)):
    def J(th, mo, s, x, fc=fc):
        mo2, s1 = AR.genarr_apply(m, key, th.data, mo, s)
        s_n, _ = integrate(step, mo2, s1, x, schedule="none", save_names=SAVE_NAMES)
        return fc(mo2, s_n)
    t1 = time.time()
    try:
        tr = sh.shard_map(J, in_specs=(th_s, mo_s, st_s, sh.REP), out_specs=sh.REP).trace(th4, mp, s04, xs)
        R[f"J_{name}"] = dict(ok=True, tile_consts=tile_consts(tr.jaxpr, nT))
        if name == "gathered":
            R[f"J_{name}"].update(fin)
    except Exception as e:                                          # noqa: BLE001 -- a measurement
        R[f"J_{name}"] = dict(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:400]}",
                              where=[ln for ln in traceback.format_exc().splitlines() if "mitjax/" in ln][-4:])
    R["t"][f"J_{name}"] = time.time() - t1
    save()
R["t"]["total"] = time.time() - t0
save()
print(json.dumps(R, indent=1, default=str))
