#!/usr/bin/env python3
"""Sharded gradients of the global_ocean.cs32x15 code_ad builds (lane M4COSTSHARD; measurement script): dfc/dxx_theta
(genarr3d, the grdchk variable) through the whole run, GenarrAdjoint's objective (CTRL_MAP_INI_GENARR, the per-step
checkpointed loop, COST_FINAL on the gathered fields: adjoint_run.final_cost_gathered) at P = 1 (single device) and
under jit(shard_map(check_vma=True)) at P = 6, 2 (TileSharding over the 12 cube tiles, fake CPU devices,
exch_maps.load_cube_maps; GenarrAdjoint.sharded_value_and_grad_fn).

Variants: ad (input_ad, ocean only), seaice (input_ad.seaice), dynmix (input_ad.seaice_dynmix). Modes:
  run   -- the run's own adjoint settings: ad = the default model (cg2dFullAdjoint = .FALSE.: operator passive);
           seaice / dynmix = the LSR derivative A1 + drivers/ad_switches.with_run_switches (approximate advection /
           free-drift switch), as test_m4adcs32ice_*adjoint.py;
  exact -- ad = the exact CG2D derivative (ad/modes.with_cg2d_derivative "exact"); seaice / dynmix = A1 alone.
Reports per P: fc (bitwise vs P = 1 expected), finite on every lane incl. padding, padding exactly 0, the number of
gradient points that differ in bits and max|g_P - g_1| / max|g_1| (R5 criterion 1e-14), the 4 grdchk points.
REF_DIR (optional): an earlier P = 1 (g1.npy, result.json J1) to compare P = 1 with bit for bit (the old final cost).
MJX_SHARD_P1_FROM=<earlier OUT_DIR> (env): P = 1 taken from there (same objective, mode and commit's code
path). MJX_SHARD_NSTEPS=k (env): the first k steps only (localisation). MJX_SHARD_OLD_P1=1 (env): also the P = 1 gradient with the pre-fix final cost (Model.cost_final with the set-up's
maskC on the local State, GenarrAdjoint until f2fa646), compared bit for bit.

    m4costshard_shardgrad.py OUT_DIR VARIANT MODE [NPROCS] [REF_DIR]      NPROCS: comma list (default 6,2)
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

from mitjax.xla_flags import gate_xla_flags

os.environ["XLA_FLAGS"] = gate_xla_flags("").replace("device_count=4", "device_count=6")
import jax  # noqa: E402
import numpy as np  # noqa: E402

OUT, VAR, MODE = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
NPROCS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "6,2").split(",") if x]
REF = Path(sys.argv[5]) if len(sys.argv) > 5 else None
assert VAR in ("ad", "seaice", "dynmix") and MODE in ("run", "exact"), (VAR, MODE)
if (OUT / "result.json").exists():
    raise SystemExit(f"{OUT}/result.json exists: choose a new OUT_DIR")
OUT.mkdir(parents=True, exist_ok=True)
assert len(jax.devices()) == 6, jax.devices()

from mitjax import paths  # noqa: E402
from mitjax.drivers import adjoint_run as AR  # noqa: E402
from mitjax.drivers.model import Model  # noqa: E402
from mitjax.drivers.run import load_experiment, make_rundir  # noqa: E402
from mitjax.drivers.sharded_grad import place_model, seed  # noqa: E402
from mitjax.eesupp import exch_maps as EM  # noqa: E402
from mitjax.eesupp.shard import TileSharding  # noqa: E402
from mitjax.io import stdout as so  # noqa: E402
from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr  # noqa: E402

INP = {"ad": "input_ad", "seaice": "input_ad.seaice", "dynmix": "input_ad.seaice_dynmix"}[VAR]
EXP = ("global_ocean.cs32x15", INP)
R = {"variant": VAR, "mode": MODE, "nprocs": NPROCS, "t": {}}


def save():
    (OUT / "result.json").write_text(json.dumps(R, indent=1, default=float))


def release():
    jax.clear_caches()
    gc.collect()


def bits(a, b):
    return int(np.count_nonzero(np.ascontiguousarray(a).view(np.int64) != np.ascontiguousarray(b).view(np.int64)))


def mode_model(m, a):
    """The Model arrays of MODE (see the module docstring)."""
    if VAR == "ad":
        if MODE == "run":
            return a.model
        from mitjax.ad.modes import with_cg2d_derivative
        return a.model.replace(cg2d_params=with_cg2d_derivative(a.model.cg2d_params, "exact"))
    from mitjax.ad.modes import with_lsr_derivative
    exact = a.model.replace(pkc={**a.model.pkc, "sp": with_lsr_derivative(a.model.pkc["sp"], "sweeps")})
    if MODE == "exact":
        return exact
    from mitjax.drivers.ad_switches import with_run_switches
    return with_run_switches(m, exact)


t0 = time.time()
exp_dir = paths.UPSTREAM / "verification" / EXP[0]
m = Model(load_experiment(exp_dir, INP), make_rundir(exp_dir, INP, OUT / "run"))
key = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
a = AR.GenarrAdjoint(m, key=key)
model = mode_model(m, a)
NST = int(os.environ.get("MJX_SHARD_NSTEPS", "0"))
if NST:                  # localisation runs: the first NST steps of the window only (fc then is COST_FINAL after them)
    a.xs = a.xs[:NST]
    R["nsteps"] = NST
theta = a.theta_farray()
sz = m.cfg.size
pts = []
for d in sorted((paths.REFERENCE_RUNS / EXP[0] / INP).glob("job*-fdzero")):        # lane A's FD oracle
    pts = [(p.bi - 1 + (p.bj - 1) * sz.nSx, p.k - 1, p.j - 1 + sz.OLy, p.i - 1 + sz.OLx)
           for p in so.grdchk(so.read_stdout(d / "rundir" / "output.txt")).points]
    R["points_from"] = str(d)
    break
R["t"]["setup"] = time.time() - t0

t1 = time.time()
P1_FROM = os.environ.get("MJX_SHARD_P1_FROM", "")
if P1_FROM:              # P = 1 of an earlier run of this script on the same objective (J1, g1.npy)
    J1, g1 = json.load(open(Path(P1_FROM) / "result.json"))["J1"], np.load(Path(P1_FROM) / "g1.npy")
    R["p1_from"] = P1_FROM
else:
    J1, g1 = jax.block_until_ready(a.value_and_grad_seed_fn()(theta, model, a.st0, a.xs, seed(1.0)))
    J1, g1 = float(J1), np.asarray(g1.data)
R["t"]["p1"] = time.time() - t1
R.update(J1=J1, J1_printed=f"{J1:.14E}", g1_points=[f"{g1[p]:.14E}" for p in pts],
         g1_finite=bool(np.all(np.isfinite(g1))), g1_absmax=float(np.max(np.abs(g1))))
np.save(OUT / "g1.npy", g1)
if REF is not None:
    ref = np.load(REF / "g1.npy")
    R["p1_vs_ref"] = dict(ref=str(REF), J_bitwise=J1 == json.load(open(REF / "result.json"))["J1"],
                          bits_differ=bits(g1, ref), rel=float(np.max(np.abs(g1 - ref)) / np.max(np.abs(ref))))
save()
release()

if os.environ.get("MJX_SHARD_OLD_P1"):
    from mitjax.drivers.checkpoint import SAVE_NAMES, integrate

    def J_old(th, mo, s, x):
        mo2, s1 = AR.genarr_apply(m, key, th.data, mo, s)
        s_n, _ = integrate(a._stepfn, mo2, s1, x, schedule="step", save_names=SAVE_NAMES)
        pk = s_n[0][4]
        cost, _ = m.cost_final(pk["cost"], pk.get("genarr"), fixed=mo2.pkc["cost_fixed"], state=s_n[0][0],
                               ecco=pk.get("ecco"))
        return cost.fc

    def body_old(th, mo, s, x, sd):
        val, pull = jax.vjp(lambda t: J_old(t, mo, s, x), th)
        return val, pull(sd)[0]
    t1 = time.time()
    Jo, go = jax.block_until_ready(jax.jit(body_old)(theta, model, a.st0, a.xs, seed(1.0)))
    Jo, go = float(Jo), np.asarray(go.data)
    R["p1_vs_old"] = dict(J_old=Jo, J_bitwise=Jo == J1, bits_differ=bits(g1, go),
                          rel=float(np.max(np.abs(g1 - go)) / np.max(np.abs(go))), t=time.time() - t1)
    save()
    release()

for nproc in NPROCS:
    t1 = time.time()
    sh = TileSharding(EM.load_cube_maps(*EXP), nproc)
    f = a.sharded_value_and_grad_fn(sh, model)
    Jp, gp = jax.block_until_ready(f(sh.put_tree(theta), place_model(sh, model), sh.put_tree(a.st0), a.xs,
                                     seed(1.0)))
    gp = np.asarray(gp.data)
    g = gp[:sh.layout.nTiles]
    R[f"P{nproc}"] = dict(J=float(Jp), J_printed=f"{float(Jp):.14E}", J_bitwise=float(Jp) == J1,
                          finite=bool(np.all(np.isfinite(gp))), pad_zero=bool(np.all(gp[sh.layout.nTiles:] == 0.0)),
                          bits_differ=bits(g, g1), rel=float(np.max(np.abs(g - g1)) / np.max(np.abs(g1))),
                          points=[f"{g[p]:.14E}" for p in pts], t=time.time() - t1)
    np.save(OUT / f"g{nproc}.npy", g)
    save()
    del f
    release()
R["t"]["total"] = time.time() - t0
save()
print(json.dumps(R, indent=1, default=float))
