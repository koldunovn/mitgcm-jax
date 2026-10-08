"""Drivers of the Task 8 prototype: run a routine of either style over all levels (the k loop of the caller,
pkg/mom_vecinv/mom_vecinv.F, pkg/generic_advdiff/gad_advection.F), load the replay harness output, plant errors for
negative controls. JAX transforms (jit) live here, never in the style modules.
"""

import importlib.util
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import proto_setup as ps  # noqa: E402
import style_farray  # noqa: E402
import style_slices  # noqa: E402

STYLES = {"farray": style_farray, "slices": style_slices}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


replay_io = _load_by_path("_mjx_replay_io", REPO / "reference" / "replay" / "replay_io.py")

# (routine, case) -> output names of replay_out.bin
CASES = ([("mom_calc_ke", s) for s in replay_io.KESCHEMES]
         + [("gad_dst3_adv_x", True), ("gad_dst3_adv_x", False)]
         + [("mom_vi_hdissip", c) for c in replay_io.HDISSIP_CASES])


def out_names(routine, case):
    if routine == "mom_calc_ke":
        return (f"KE_{'m1' if case == -1 else case}",)
    if routine == "gad_dst3_adv_x":
        return ("uT_calcCFL" if case else "uT_givenCFL",)
    c = replay_io.HDISSIP_CASES.index(case)
    return (f"uDissip_c{c}", f"vDissip_c{c}")


def level_fn(mod, style, cfg, routine, case, k):
    """f(fields_k, grid, params, deltaTloc) -> tuple of raw outputs [tile, j, i] for level k: one call of the routine
    as its caller makes it (fields_k: {name: [tile, j, i]} of replay_io.IN3)."""
    def L(fields, name):
        return ps.level_array(cfg, fields[name], name, style)

    def f(fields, grid, params, deltaTloc):
        if routine == "mom_calc_ke":
            KE = mod.mom_calc_ke(k, case, L(fields, "uFld"), L(fields, "vFld"), L(fields, "KEprior"),
                                 cfg=cfg, grid=grid)
            return (ps.unwrap(KE),)
        if routine == "gad_dst3_adv_x":
            uF = L(fields, "uVel") if case else L(fields, "uCFL")
            # the caller's maskLocW(i,j) = _maskW(i,j,k,bi,bj)  (gad_advection.F:328)
            maskLocW = ps.level_array(cfg, ps.unwrap(grid.maskW)[:, k - 1], "maskLocW", style)
            uT = mod.gad_dst3_adv_x(k, case, deltaTloc, L(fields, "uTrans"), uF, maskLocW, L(fields, "tracer"),
                                    L(fields, "uTprior"), cfg=cfg, grid=grid)
            return (ps.unwrap(uT),)
        harmonic, biharmonic, useVV = case
        uD, vD = mod.mom_vi_hdissip(k, *(L(fields, n) for n in ("hDiv", "vort3", "dStar", "zStar", "hFacZ",
                                                                  "viscAh_Z", "viscAh_D", "viscA4_Z", "viscA4_D")),
                                    harmonic, biharmonic, useVV, L(fields, "uDissipPrior"), L(fields, "vDissipPrior"),
                                    cfg=cfg, grid=grid, params=params)
        return ps.unwrap(uD), ps.unwrap(vD)
    return f


def all_levels_fn(mod, style, cfg, routine, case):
    """f(fields, grid, params, deltaTloc) -> tuple of outputs [tile, k, j, i]: the routine for k = 1..Nr (a Python
    loop over levels, as the Fortran caller's DO k loop)."""
    def f(fields, grid, params, deltaTloc):
        per_k = []
        for k in range(1, cfg.Nr + 1):
            fk = {n: a[:, k - 1] for n, a in fields.items()}
            per_k.append(level_fn(mod, style, cfg, routine, case, k)(fk, grid, params, deltaTloc))
        return tuple(jnp.stack([o[n] for o in per_k], axis=1) for n in range(len(per_k[0])))
    return f


class Replay:
    """The replay harness run: inputs, Fortran outputs, grid, and the cfg of its build."""

    def __init__(self, rundir, options_dir):
        self.rundir = Path(rundir)
        self.size, self.fields, self.scal, self.one_d = replay_io.read_inputs(self.rundir)
        self.out = replay_io.read_outputs(self.rundir, self.size)
        self.grid_np = replay_io.read_grid(self.rundir, self.size)
        self.cfg = ps.cfg_from_build(options_dir, self.size, useDiagnostics=False)

    def args(self, style, routine=None):
        """(fields, grid, params, deltaTloc) as jax arrays for a style; fields restricted to what `routine` reads."""
        need = {"mom_calc_ke": ("uFld", "vFld", "KEprior"),
                "gad_dst3_adv_x": ("uTrans", "uVel", "uCFL", "tracer", "uTprior"),
                "mom_vi_hdissip": ("hDiv", "vort3", "dStar", "zStar", "hFacZ", "viscAh_Z", "viscAh_D", "viscA4_Z",
                                   "viscA4_D", "uDissipPrior", "vDissipPrior")}
        names = need[routine] if routine else replay_io.IN3
        fields = {n: jnp.asarray(self.fields[n]) for n in names}
        return (fields, ps.make_grid(self.cfg, self.grid_np, style), ps.make_params(self.grid_np),
                jnp.asarray(self.scal["deltaTloc"], jnp.float64))


def current_replay():
    """The replay run named by reference/replay/CURRENT (relative to $MJX_REFERENCE); (Replay, run top dir)."""
    from mitjax import paths
    rel = (REPO / "reference" / "replay" / "CURRENT").read_text().split()[0]
    rundir = paths.REFERENCE / rel
    out_dir = rundir
    while out_dir.name != "runs" and out_dir != out_dir.parent:
        out_dir = out_dir.parent
    return Replay(rundir, out_dir.parent / "options")


def bit_diff(a, b):
    """Number of elements whose float64 bit patterns differ."""
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return int(np.count_nonzero(a.view(np.int64) != b.view(np.int64)))


def planted(mod, old, new, name="planted"):
    """A fresh copy of a style module with `old` replaced by `new` (exactly one occurrence): a negative control."""
    src = Path(mod.__file__).read_text()
    if src.count(old) != 1:
        raise ValueError(f"plant target occurs {src.count(old)} times in {mod.__file__}: {old!r}")
    m = type(sys)(f"{mod.__name__}_{name}")
    m.__file__ = mod.__file__
    exec(compile(src.replace(old, new), f"<{name}>", "exec"), m.__dict__)
    return m


def normalize_hlo(text, names=True):
    """Optimized HLO text without debug locations: drop the FileNames/FunctionNames/FileLocations/StackFrames tables
    and every `metadata={...}` attribute. With names=True also rename the module (jit_<python name>) and every
    identifier (`%name`, and the parameter
    names of the ENTRY signature) to v<n> in order of first appearance: parameter names come from the argument's
    pytree path (an FArray's storage is child 0 of its node, `grid_5__0_`, a plain array is `grid_5_`)."""
    out, skip = [], False
    for line in text.splitlines():
        if line.strip() in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            skip = True
            continue
        if skip:
            if line.strip() == "":
                skip = False
            continue
        out.append(re.sub(r", metadata=\{[^{}]*\}", "", line))
    text = "\n".join(out)
    if names:
        text = re.sub(r"^HloModule [^,]+,", "HloModule m,", text, flags=re.M)
        canon = {}
        def ren(m):
            return m.group(1) + canon.setdefault(m.group(2), f"v{len(canon)}")
        text = re.sub(r"(%)([A-Za-z_][\w.\-]*)", ren, text)
        text = re.sub(r"([(,] ?)([A-Za-z_][\w.\-]*)(?=: (?:f64|f32|s32|s64|pred|u32)\[)", ren, text)
    return text
