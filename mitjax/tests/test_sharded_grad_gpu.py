"""Sharded gradients on real GPUs (plan Task 17, tier 2; [L-PAR-11], [L-AD-32], [L-AD-33]): R2
tutorial_baroclinic_gyre (4 tiles of 31 x 31), J = weighted sum of theta and etaN after 10 steps
(mitjax/tests/shardgrad_gate.py), dJ/d(initial State, FFIELDS.h, phi0surf) through `integrate` with the sqrt
schedule: the single-device program on one A100 (P=1) vs mitjax/drivers/sharded_grad.py on P A100s (P =
$MJX_SHARDGRAD_P, default 4: one tile per GPU; ShardedExchanger ppermute rounds inside shard_map(check_vma=True)).
Each program also returns the final State of its own forward pass, so the forward comparison costs no second program.

Run by scripts/run_tier2.sbatch (`--gpus=4`: all three tests in one job; `--gpus=1 ... p1_gpu`: test_p1_gpu), with
the GPU-only XLA flag set (MJX_XLA_FLAG_SET=gpu: the gate flags without algsimp disabled, mitjax/xla_flags.py). The
model is built on the host CPU and placed on the GPUs. A GPU run need not be bitwise reproducible (scatter-add atomics
in the transposes), so every GPU comparison is made against its MEASURED repeat floor: each program runs twice with the
same inputs; floor = max over leaves of max|a - a'| / max|a| (shardgrad_gate.summary, per-leaf relative).

Cost (measured, scripts/shardgrad/compile_study.py, job 27833512, one A100): the P=1 program traces in 50 s, lowers in
8 s and compiles in 254 s with the GPU-only flag set (701 s with algsimp disabled, the CPU gate set); a warm call takes
0.9 s. With the gate set the P=1 first call took 809 s (job 27832390) and the P=2 sharded program had not returned
after 16 min on 2 A100s (job 27832889, TIMEOUT). With the GPU set the 4-A100 job (27833699) took 13:41: P=4 first call
338 s, P=1 417 s.

Measured (job 27833699, PASS): J bitwise across runs and P; gradient floors 1.50e-15 (P=1) and 2.44e-15 (P=4) per
leaf; P=4 vs P=1 2.55e-15 (the CPU value: the exchange-transpose summation order); final State bitwise P=4 vs P=1 and
across runs; every point finite; seed 0 -> 0; seed 2 vs 2 x seed 1 within the floor.

* test_p1_gpu (1 GPU) / test_sharded_gpu (P GPUs): two runs (floor), every gradient point finite (sharded: the padded
  layout, padding exactly 0), linearity guard of the reverse program (seed 0 -> exactly 0 everywhere; seed 2 vs
  2 x seed 1 within 2 x floor); arrays (npz) and numbers (json) saved to $MJX_TIER2_OUT.
* test_sharded_vs_p1_gpu: sharded vs P=1 gradient within floor(P=1) + floor(sharded); final State (the forward)
  within forward-floor(P=1) + forward-floor(sharded) (deterministic forward programs: bitwise); J likewise. Both
  come from this process, or from the saved arrays of the jobs named by MJX_SHARDGRAD_P1_DIR / MJX_SHARDGRAD_PN_DIR
  (P A100s vs 1 A100 across two jobs; then no GPU is needed for this test).
"""

import json
import os

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def _gpus():
    """The GPUs of this process (no backend is touched at import or collection: the CPU tiers collect this file)."""
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


NP = int(os.environ.get("MJX_SHARDGRAD_P", "4"))


def _need(n):
    if len(_gpus()) < n:
        pytest.skip(f"tier 2: needs {n} GPU(s) (sbatch --gpus={n} scripts/run_tier2.sbatch)")


_C = {}


def _G():
    from mitjax.tests import shardgrad_gate as G
    return G


def _setup():
    if "su" not in _C:
        with jax.default_device(jax.devices("cpu")[0]):
            _C["su"] = _G().Setup(_G().model())
    return _C["su"]


def _flat(tree):
    return {k: v for k, v in _G().leaves(tree)}


def _runs(f, unpad=lambda t: t):
    """f(seed) -> (J, g, state): seed 1 twice (repeat floor), seeds 0 and 2 (linearity guard). Returns the flat
    {path: array} forms (unpadded) and the checks."""
    G = _G()
    get = jax.device_get
    (Ja, ga, sa), ta = G.timed(f, 1.0)
    (Jb, gb, sb), tb = G.timed(f, 1.0)
    (_, gz, _), _ = G.timed(f, 0.0)
    (_, g2, _), _ = G.timed(f, 2.0)
    ga, gb, gz, g2 = get(ga), get(gb), get(gz), get(g2)
    r = dict(J=[float(Ja), float(Jb)], xla_flags=os.environ.get("XLA_FLAGS", ""), ga=_flat(unpad(ga)),
             gb=_flat(unpad(gb)), sa=_flat(unpad(get(sa))),
             sb=_flat(unpad(get(sb))))
    c = dict(J=r["J"], times=[ta, tb], xla_flags=os.environ.get("XLA_FLAGS", ""),
             floor=G.summary(G.compare(r["ga"], r["gb"])),
             forward_floor=G.summary(G.compare(r["sa"], r["sb"])),
             nonfinite=G.nonfinite(ga), nonfinite_seed0=G.nonfinite(gz),
             guard_zero_nonzero=sum(int(np.count_nonzero(x)) for _, x in G.leaves(gz)),
             guard_two=G.summary(G.compare(jax.tree.map(lambda a: 2.0 * np.asarray(a), ga), g2)))
    return r, c, ga


def _save(tag, r, c):
    d = os.environ.get("MJX_TIER2_OUT")
    print(f"{tag}:", json.dumps(c, default=str))
    if not d:
        return
    for k in (("ga", "gb", "sa", "sb") if r is not None else ()):
        np.savez(os.path.join(d, f"{tag}_{k}.npz"), **r[k])
    with open(os.path.join(d, f"{tag}.json"), "w") as fh:
        json.dump(c, fh, indent=1, default=str)


def _load(tag, d):
    p = os.path.join(d, f"{tag}.json")
    if not os.path.exists(p):
        pytest.skip(f"{p} does not exist (yet): run this comparison again once that job has saved its arrays")
    with open(p) as fh:
        c = json.load(fh)
    r = {k: dict(np.load(os.path.join(d, f"{tag}_{k}.npz"))) for k in ("ga", "gb", "sa", "sb")}
    r["J"] = c["J"]
    r["xla_flags"] = c.get("xla_flags")
    return r


def _assert_checks(c):
    assert not c["nonfinite"] and not c["nonfinite_seed0"], (c["nonfinite"], c["nonfinite_seed0"])
    assert c["guard_zero_nonzero"] == 0, c["guard_zero_nonzero"]
    assert c["guard_two"][1] <= 2 * c["floor"][1], (c["guard_two"], c["floor"])


def test_sharded_gpu():
    _need(NP)
    G = _G()
    sd = G.Sharded(_setup(), NP, devices=_gpus()[:NP])
    r, c, ga = _runs(sd.value_and_grad, unpad=sd.sh.unpad_tree)
    c["padding_nonzero"] = G.padding_nonzero(sd.sh, ga)
    c["devices"] = [str(x) for x in sd.sh.mesh.devices.flat]
    c["P"] = NP
    _C["pn"] = r
    _save("pn_gpu", r, c)
    _assert_checks(c)
    assert not c["padding_nonzero"], c["padding_nonzero"]


def test_p1_gpu():
    _need(1)
    G = _G()
    su = _setup()
    g0 = _gpus()[0]
    th, mo, s0, xs = jax.device_put((su.theta, su.model, su.st0, su.xs), g0)
    f = su.p1_seed_fn()
    r, c, ga = _runs(lambda seed: f(th, mo, s0, xs, jnp.float64(seed)))
    c["devices"] = [str(g0)]
    _C["p1"] = r
    _save("p1_gpu", r, c)
    _assert_checks(c)
    assert any(np.any(x != 0) for x in r["ga"].values())


def test_sharded_vs_p1_gpu():
    G = _G()
    r1, r4 = _C.get("p1"), _C.get("pn")
    d1, d4 = os.environ.get("MJX_SHARDGRAD_P1_DIR"), os.environ.get("MJX_SHARDGRAD_PN_DIR")
    if r1 is None and d1:
        r1 = _load("p1_gpu", d1)
    if r4 is None and d4:
        r4 = _load("pn_gpu", d4)
    if r1 is None or r4 is None:
        pytest.skip("needs P=1 and sharded results (this process, or MJX_SHARDGRAD_P1_DIR / MJX_SHARDGRAD_PN_DIR)")
    f1x, f4x = r1.get("xla_flags"), r4.get("xla_flags")
    if f1x is not None and f4x is not None and f1x != f4x:
        # another flag set is another program: the 1-A100 job 27832390 (gate set) vs the 4-A100 job 27833699 (GPU set)
        # differ in the forward by rounding (per leaf 7.8e-13, 6.9e5 points; J equal), so P is not what is compared
        pytest.skip(f"the two runs used different XLA_FLAGS ({f1x!r} vs {f4x!r})")
    f1 = G.summary(G.compare(r1["ga"], r1["gb"]))
    f4 = G.summary(G.compare(r4["ga"], r4["gb"]))
    ff1 = G.summary(G.compare(r1["sa"], r1["sb"]))
    ff4 = G.summary(G.compare(r4["sa"], r4["sb"]))
    dg = G.compare(r1["ga"], r4["ga"])
    ds = G.compare(r1["sa"], r4["sa"])
    res = dict(grad_pn_vs_p1=G.summary(dg), floor_p1=f1, floor_pn=f4, forward_pn_vs_p1=G.summary(ds),
               forward_floor_p1=ff1, forward_floor_pn=ff4, J1=r1["J"], Jn=r4["J"], p1_dir=d1, pn_dir=d4,
               grad_worst=dict(sorted(dg.items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else kv[1][0]))[:6]),
               forward_worst=dict(sorted(ds.items(), key=lambda kv: -kv[1][0])[:6]))
    _save("pn_vs_p1_gpu", None, res)
    assert res["grad_pn_vs_p1"][1] <= f1[1] + f4[1], res
    assert res["forward_pn_vs_p1"][1] <= ff1[1] + ff4[1], res
    jf = abs(r1["J"][0] - r1["J"][1]) + abs(r4["J"][0] - r4["J"][1])
    assert abs(r4["J"][0] - r1["J"][0]) <= jf, res
