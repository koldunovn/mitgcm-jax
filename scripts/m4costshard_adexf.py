#!/usr/bin/env python3
"""The EXF adjoint monitor (`%MON ad_exf_*`, iwhen = 3) of the step-7 sea-ice adjoint runs vs TAF's output_adm files,
block by block (lane M4COSTSHARD; measurement script).

ADEXF_MONITOR( 3 ) reports adfu, adfv, adQnet, adEmPmR (adQsw) at the reverse of EXF_ADJOINT_SNAPSHOTS( 3 ), the last
call of SEAICE_MODEL (seaice_model.F:405): the cotangents of FFIELDS.h right after SEAICE_MODEL in each step. This is
a point inside the step, so this script taps it: SEAICE_MODEL (mitjax.pkg.seaice.seaice_model.seaice_model, which
DO_OCEANIC_PHYS imports at call time) is wrapped -- in this process only -- so that its returned FFIELDS go through
drivers/checkpoint._tap (identity forward; the reverse pass reports the cotangent as the gradient of a zero "sink",
one per step, fed with the step's x). The forward is unchanged (fc and the control gradient are compared bit for bit
with the untapped program). Records: drivers/adjoint_run.ad_exf_records; compared per (tsnumber, record) with the
blocks of the TAF output that follow `Begin AD_MONITOR EXF statistics for iwhen =  3`.

Variants (the run's adjoint mode, as their gates): col = 1D_ocean_ice_column/input_ad (GenarrAdjoint xx_theta),
lab = lab_sea/input_ad (Adjoint xx_atemp rec 1; LSR A1), seaice / dynmix = global_ocean.cs32x15 input_ad.seaice /
input_ad.seaice_dynmix (GenarrAdjoint xx_theta, A1 + with_run_switches).

    m4costshard_adexf.py OUT_DIR VARIANT
"""
import json
import sys
import time
from pathlib import Path

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

OUT, VAR = Path(sys.argv[1]), sys.argv[2]
EXPS = {"col": ("1D_ocean_ice_column", "input_ad", "output_adm.txt"),
        "lab": ("lab_sea", "input_ad", "output_adm.txt"),
        "seaice": ("global_ocean.cs32x15", "input_ad.seaice", "output_adm.seaice.txt"),
        "dynmix": ("global_ocean.cs32x15", "input_ad.seaice_dynmix", "output_adm.seaice_dynmix.txt")}
EXP = EXPS[VAR]
if (OUT / "result.json").exists():
    raise SystemExit(f"{OUT}/result.json exists: choose a new OUT_DIR")
OUT.mkdir(parents=True, exist_ok=True)

import mitjax.pkg.seaice.seaice_model as SM  # noqa: E402
from mitjax import paths  # noqa: E402
from mitjax.drivers import adjoint_run as AR  # noqa: E402
from mitjax.drivers.checkpoint import SAVE_NAMES, _tap, integrate  # noqa: E402
from mitjax.drivers.model import Model  # noqa: E402
from mitjax.drivers.run import load_experiment, make_rundir  # noqa: E402
from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr  # noqa: E402

R = {"variant": VAR, "t": {}}


def save():
    (OUT / "result.json").write_text(json.dumps(R, indent=1, default=float))


def digits(a, b):
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


t0 = time.time()
exp_dir = paths.UPSTREAM / "verification" / EXP[0]
m = Model(load_experiment(exp_dir, EXP[1]), make_rundir(exp_dir, EXP[1], OUT / "run"))
FIELDS = [n for n in AR.AD_EXF3 if n in m.ff0 and (n != "Qsw" or m.cfg.cpp.SHORTWAVE_HEATING)]
R["fields"] = FIELDS
if VAR == "lab":
    a = AR.Adjoint(m, iarr=1, rec=1)
    model = a.model
    theta0 = a.theta0

    def prep(th, mo):
        return a._params_fn(th, mo), a.st0
else:
    if VAR == "col":
        key = (3, 1)
    else:
        key = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
    a = AR.GenarrAdjoint(m, key=key)
    model = a.model
    if VAR in ("seaice", "dynmix"):
        from mitjax.ad.modes import with_lsr_derivative
        from mitjax.drivers.ad_switches import with_run_switches
        model = with_run_switches(m, model.replace(pkc={**model.pkc, "sp": with_lsr_derivative(model.pkc["sp"],
                                                                                                 "sweeps")}))
    theta0 = a.theta0

    def prep(th, mo):
        return AR.genarr_apply(m, key, th, mo, a.st0)

# --- the tap after SEAICE_MODEL (this process only)
_orig = SM.seaice_model
CUR = {"sink": None, "calls": 0}


def _stats(_, ct):
    return {n: getattr(ct, n).data for n in FIELDS}


def _tapped(*args, **kw):
    out = _orig(*args, **kw)
    if CUR["sink"] is None:
        return out
    CUR["calls"] += 1
    sf, ff, exf_f = out[:3]
    return (sf, _tap(_stats, (), ff, CUR["sink"]), exf_f) + tuple(out[3:])


SM.seaice_model = _tapped


def step2(mo, st, x):
    il, sink = x
    CUR["sink"] = sink
    try:
        return a._stepfn(mo, st, il)
    finally:
        CUR["sink"] = None


n = int(a.xs.shape[0])
sinks0 = {nm: jnp.zeros((n,) + tuple(getattr(m.ff0, nm).data.shape)) for nm in FIELDS}


def J(th, sinks, mo):
    mo2, s0 = prep(th, mo)
    s_n, _ = integrate(step2, mo2, s0, (a.xs, sinks), schedule="step", save_names=SAVE_NAMES)
    return a._final_cost(mo2, s_n)


R["t"]["setup"] = time.time() - t0
t1 = time.time()
fc, (g, gs) = jax.jit(jax.value_and_grad(J, argnums=(0, 1)))(theta0, sinks0, model)
R["t"]["grad_tapped"] = time.time() - t1
R["tap_traced_calls"] = CUR["calls"]
g = np.asarray(g)
gs = {k: np.asarray(v) for k, v in gs.items()}
np.savez(OUT / "adexf_stats.npz", **gs)
SM.seaice_model = _orig
jax.clear_caches()
t1 = time.time()
fc0, g0 = a.value_and_grad(model=model)
R["t"]["grad_plain"] = time.time() - t1
g0 = np.asarray(g0)
R.update(fc=float(fc), fc_untapped_bitwise=float(fc) == float(fc0),
         grad_untapped_bits_differ=int(np.count_nonzero(np.ascontiguousarray(g).view(np.int64)
                                                        != np.ascontiguousarray(g0).view(np.int64))))
save()

ours = AR.ad_exf_records(m, gs)
(OUT / "adexf_ours.txt").write_text("\n".join(ours) + "\n")


def blocks(lines):
    """{tsnumber: {record: value}} of the iwhen = 3 blocks."""
    out, cur, iw = {}, None, None
    for r in lines:
        if "Begin AD_MONITOR EXF statistics for iwhen" in r:
            iw = int(r.split("=")[-1])
        if "%MON ad_exf_tsnumber" in r:
            cur = {} if iw == 3 else None
            if cur is not None:
                out[int(r.split("=")[1])] = cur
        if cur is not None and "%MON ad_exf_" in r:
            cur[r.split("%MON")[1].split("=")[0].strip()] = float(r.split("=")[1])
        if "End AD_MONITOR EXF statistics" in r:
            cur, iw = None, None
    return out


bo = blocks(ours)
bt = blocks((exp_dir / "results" / EXP[2]).read_text(errors="replace").splitlines())
R["blocks"] = dict(ours=sorted(bo), taf=sorted(bt))
cmp = {}
for ts in sorted(bt, reverse=True):
    if ts not in bo:
        cmp[ts] = "missing in ours"
        continue
    cmp[ts] = {nm: (digits(bo[ts][nm], v), bo[ts][nm], v) if nm in bo[ts] else "missing" for nm, v in bt[ts].items()
               if nm not in ("ad_exf_tsnumber", "ad_exf_time_sec")}
    cmp[ts]["time_sec_equal"] = bo[ts].get("ad_exf_time_sec") == bt[ts].get("ad_exf_time_sec")
R["compare"] = cmp
R["min_digits"] = {nm: min(c[nm][0] for c in cmp.values() if isinstance(c, dict) and isinstance(c.get(nm), tuple))
                   for nm in next((c for c in cmp.values() if isinstance(c, dict)), {})
                   if nm != "time_sec_equal"}
R["t"]["total"] = time.time() - t0
save()
print(json.dumps({k: R[k] for k in ("fc", "fc_untapped_bitwise", "grad_untapped_bits_differ", "blocks",
                                     "min_digits")}, indent=1, default=float))
