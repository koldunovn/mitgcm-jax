"""Lane APIF (docs plan 20261006 S5): the API follow-ups that need no model run, each with its planted control.

* XLA flags by platform (contract 1): the API's set omits --xla_cpu_max_isa=AVX off x86-64 (platform mocked); the gate
  set is unchanged; the user's own XLA_FLAGS entries win (the gates still refuse a conflict).
  Control: an API set that ignores the platform keeps AVX on arm64; a merge that refuses user flags raises.
* devices above the JAX device count (contract 11): DeviceCountError (a ValueError, also a RuntimeError) naming the
  count, before any run directory. Control: the check removed -> the run directory exists when the error comes.
* MinMaxDefaultWarning (contract 7): exported as mitjax.MinMaxDefaultWarning; once per (build, site) per process, so
  a second build in the same process warns again; `fortran_minmax._WARNED.clear()` (notebook 04) still resets it.
  Control: the per-site key of before (one identity for every build) -> the second build does not warn.
* RUNDIR line (contract 6): the API's run directory of a prepare_run variant prints no `RUNDIR` line.
  Control: drivers.run.make_rundir itself (the CLI path) prints it.
* cg2d_derivative (contract 9): _mode_arrays sets the CG2D option over the mode's choice. Control: the argument
  ignored -> mode "exact" keeps "exact".
* LSR default (contract 10): lsr_derivative=None selects "sweeps" exactly where the build's own choice is
  "forward_only". Control: the default removed -> "forward_only" stays.
* clear_caches (contract 8): gradient() of a genarr control calls jax.clear_caches() after its program (drivers
  mocked). Control: the call removed -> zero calls.
* tools/build_notebooks.py (contract 12): names select notebooks; a notebook whose cells equal its source keeps its
  outputs; --check unchanged. Control: the old always-rewrite -> the committed outputs are dropped.
* TileMap (contract 4, synthetic): exch1 tiles land at ((bj-1)*sNy, (bi-1)*sNx); the Fortran-file gates are in
  test_api_followups.py.
Seconds each (one lab_sea config copy, two config loads, one prepare_run directory).
"""

import dataclasses
import json
import os
import shutil
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mitjax import paths
from mitjax import xla_flags as XF

ISA = "--xla_cpu_max_isa=AVX"


def _new_dir(tag):
    d = paths.RUNS / "tests_port" / f"apif-{tag}-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    return d


def _exp(name):
    return paths.UPSTREAM / "verification" / name


# ------------------------------------------------------------------------------------------------- 1. XLA flags

def _flags_ok(api_flag_set):
    """The contract of api_flag_set on four platform names (the gate set itself must stay as it is)."""
    want_x86 = XF.flag_set()
    ok = XF.GATE_FLAGS == ("--xla_cpu_max_isa=AVX", "--xla_disable_hlo_passes=algsimp,multi_output_fusion",
                           "--xla_force_host_platform_device_count=4")
    for m in ("x86_64", "AMD64"):
        ok &= api_flag_set(m) == want_x86
    for m in ("arm64", "aarch64"):
        ok &= api_flag_set(m) == tuple(f for f in want_x86 if f != ISA)
    return bool(ok)


def test_api_xla_flags_by_platform(monkeypatch):
    import platform
    monkeypatch.delenv("MJX_XLA_FLAG_SET", raising=False)
    assert _flags_ok(XF.api_flag_set)
    monkeypatch.setattr(platform, "machine", lambda: "arm64")                  # the default: platform.machine()
    monkeypatch.setenv("XLA_FLAGS", "")
    s = XF.set_api_xla_flags()
    assert ISA not in s.split() and "--xla_force_host_platform_device_count=4" in s.split(), s
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("XLA_FLAGS", "")
    assert XF.set_api_xla_flags().split() == list(XF.GATE_FLAGS)
    # planted: the platform ignored (the API set = the gate set)
    assert not _flags_ok(lambda machine=None: XF.flag_set())


def test_user_xla_flags_win(monkeypatch):
    monkeypatch.delenv("MJX_XLA_FLAG_SET", raising=False)
    user = "--xla_force_host_platform_device_count=8 --xla_dump_to=/x"
    got = XF.merge_user_flags(user, XF.api_flag_set("x86_64")).split()
    assert got == ["--xla_force_host_platform_device_count=8", "--xla_dump_to=/x", ISA,
                   "--xla_disable_hlo_passes=algsimp,multi_output_fusion"], got
    monkeypatch.setenv("XLA_FLAGS", user)
    assert XF.set_api_xla_flags("x86_64").split() == got
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_max_isa=AVX2")
    assert XF.set_api_xla_flags("x86_64").split()[0] == "--xla_cpu_max_isa=AVX2"   # the user's ISA kept
    # the user's own pass list keeps the gate set's passes (GPU race 2026-10-09: multi_output_fusion; algsimp);
    # planted: the merge that leaves the user's list alone drops them
    for mine, want in (("--xla_disable_hlo_passes=foo", "foo,algsimp,multi_output_fusion"),
                       ("--xla_disable_hlo_passes=multi_output_fusion", "multi_output_fusion,algsimp"),
                       ("--xla_disable_hlo_passes=algsimp,multi_output_fusion", "algsimp,multi_output_fusion")):
        merged = XF.merge_user_flags(mine, XF.api_flag_set("x86_64")).split()
        assert merged[0] == f"--xla_disable_hlo_passes={want}" and len(merged) == 3, merged
    with monkeypatch.context() as mp:
        mp.setattr(XF, "UNION_FLAGS", ())
        leave_alone = XF.merge_user_flags("--xla_disable_hlo_passes=foo", XF.api_flag_set("x86_64"))
    assert "multi_output_fusion" not in leave_alone, leave_alone
    # the gates are unchanged: a conflicting user flag is refused (and is the planted merge of the API)
    with pytest.raises(ValueError, match="already sets"):
        XF.gate_xla_flags(user)


# ----------------------------------------------------------------------------------------------- 11. devices

def test_devices_above_count_refused_before_rundir(monkeypatch):
    import jax

    import mitjax
    from mitjax import api
    n = len(jax.devices())
    exp = mitjax.load(_exp("tutorial_barotropic_gyre"), variant="input")
    base = _new_dir("devices")
    for call in (lambda o: exp.run(o, devices=n + 1), lambda o: exp.gradient(o, devices=n + 1)):
        out = base / f"o{time.time_ns()}"
        with pytest.raises(ValueError, match=f"devices={n + 1}: JAX has {n} device") as e:
            call(out)
        assert isinstance(e.value, RuntimeError) and isinstance(e.value, api.DeviceCountError)
        assert not out.exists()
    with pytest.raises(ValueError, match="devices=0"):
        exp.run(base / "zero", devices=0)
    # planted: no check -> the run directory is made before the (later) failure
    monkeypatch.setattr(api, "_check_devices", lambda d: int(d))
    out = base / "planted"
    with pytest.raises(Exception):
        exp.gradient(out, devices=n + 1)      # gyre has no data.ctrl: refused after the run directory is made
    assert out.exists()


# ------------------------------------------------------------------------------------------------ 7. warnings

def _copy_lab_sea(name, edit):
    src = _exp("lab_sea")
    dst = _new_dir("minmax") / name
    for d in ("code_ad", "input_ad"):
        shutil.copytree(src / d, dst / d, symlinks=False)
    edit(dst)
    return dst


def _plant_option(exp_dir):
    h = exp_dir / "code_ad" / "SEAICE_OPTIONS.h"
    h.write_text(h.read_text() + "\n#define MJX_PLANTED_OPTION\n")


def _plant_tiling(exp_dir):
    h = exp_dir / "code_ad" / "SIZE.h"
    t = h.read_text()
    assert "sNx =  10," in t and "nSx =   2," in t, "lab_sea code_ad SIZE.h changed"
    h.write_text(t.replace("sNx =  10,", "sNx =  20,").replace("nSx =   2,", "nSx =   1,"))     # 2 x 10 -> 1 x 20


def _warned(cfg):
    from mitjax.ops import fortran_minmax as fm
    from mitjax.pkg.seaice import seaice_growth as G
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for s in G.MINMAX_DEFAULT:
            fm.build_winner(cfg, G.MINMAX_BUILD, s, G.MINMAX_DEFAULT)
    return [x for x in w if issubclass(x.category, fm.MinMaxDefaultWarning)]


def test_minmax_warning_once_per_build_and_site(monkeypatch):
    import mitjax
    from mitjax.config import build_identity as bi
    from mitjax.config import params
    from mitjax.ops import fortran_minmax as fm
    from mitjax.pkg.seaice import seaice_growth as G
    assert mitjax.MinMaxDefaultWarning is fm.MinMaxDefaultWarning
    assert "MinMaxDefaultWarning" in dir(mitjax)
    a = params.load("lab_sea", "input_ad", exp_dir=_copy_lab_sea("lab_sea", _plant_option)).cfg
    b = params.load("lab_sea", "input_ad", exp_dir=_copy_lab_sea("lab_sea", _plant_tiling)).cfg
    assert bi.label(a) is None and bi.label(b) is None and bi.identity(a) != bi.identity(b)
    nsites = len(G.MINMAX_DEFAULT)

    def counts():
        return len(_warned(a)), len(_warned(a)), len(_warned(b)), len(_warned(b))

    monkeypatch.setattr(fm, "_WARNED", set())
    assert counts() == (nsites, 0, nsites, 0)
    fm._WARNED.clear()                                     # notebook 04's reset keeps working
    assert len(_warned(a)) == nsites
    # planted: the key of before (the site only): the second build is silent
    monkeypatch.setattr(fm, "_WARNED", set())
    monkeypatch.setattr(bi, "identity", lambda cfg: "same")
    assert counts() == (nsites, 0, 0, 0)


# ---------------------------------------------------------------------------------------------- 6. RUNDIR line

def test_no_rundir_line_from_api(capsys):
    import mitjax
    from mitjax.drivers.run import make_rundir
    exp = mitjax.load(_exp("advect_cs"), variant="input")                  # input/prepare_run links the cube grid
    base = _new_dir("rundir")
    rd = exp._rundir(base / "api")
    out = capsys.readouterr().out
    assert rd.exists() and "RUNDIR" not in out, out
    make_rundir(exp.exp_dir, "input", base / "driver")                      # the CLI path keeps printing it
    assert "RUNDIR " in capsys.readouterr().out


# --------------------------------------------------------------------------------------- 9, 10. derivative options

@dataclasses.dataclass(frozen=True)
class _Cg2d:
    mjx_cg2d_derivative: str = "run"


class _Arrays(SimpleNamespace):
    def replace(self, **kw):
        return _Arrays(**{**vars(self), **kw})


def test_cg2d_derivative_option(monkeypatch):
    from mitjax import api
    exp = object.__new__(api.Experiment)
    model = _Arrays(cg2d_params=_Cg2d(), pkc={})
    got = {c: exp._mode_arrays(None, model, "exact", None, c).cg2d_params.mjx_cg2d_derivative
           for c in (None, "run", "exact")}
    assert got == {None: "exact", "run": "run", "exact": "exact"}, got
    with pytest.raises(ValueError, match="cg2d_derivative"):
        exp._mode_arrays(None, model, "exact", None, "full")
    # planted: the argument ignored
    orig = api.Experiment._mode_arrays
    monkeypatch.setattr(api.Experiment, "_mode_arrays",
                        lambda self, m, mo, mode, lsr, cg2d=None: orig(self, m, mo, mode, lsr, None))
    assert exp._mode_arrays(None, model, "exact", None, "run").cg2d_params.mjx_cg2d_derivative == "exact"


class _Cpp:
    def __init__(self, flags):
        self.flags = flags

    def flag(self, name, header=None):
        return name in self.flags


@dataclasses.dataclass(frozen=True)
class _Sp:
    SEAICElinearIterMax: int = 100
    mjx_lsr_derivative: str = "run"

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


def test_lsr_default(monkeypatch):
    from mitjax import api
    from mitjax.ad.modes import lsr_derivative
    exp = object.__new__(api.Experiment)
    for flags, want_opt in (((), "sweeps"), (("SEAICE_LSR_ADJOINT_ITER",), "run")):
        m = SimpleNamespace(cfg=SimpleNamespace(cpp=_Cpp(flags)))
        model = _Arrays(cg2d_params=_Cg2d(), pkc={"sp": _Sp()})
        sp = exp._mode_arrays(m, model, "exact", None).pkc["sp"]
        assert sp.mjx_lsr_derivative == want_opt and lsr_derivative(m.cfg, sp) == "sweeps", (flags, sp)
        sp = exp._mode_arrays(m, model, "exact", "run").pkc["sp"]               # the explicit opt-out
        assert lsr_derivative(m.cfg, sp) == ("sweeps" if flags else "forward_only")
    # planted: no default
    monkeypatch.setattr(api, "_lsr_default_applies", lambda m, sp, choice: False)
    m = SimpleNamespace(cfg=SimpleNamespace(cpp=_Cpp(())))
    sp = exp._mode_arrays(m, _Arrays(cg2d_params=_Cg2d(), pkc={"sp": _Sp()}), "exact", None).pkc["sp"]
    assert lsr_derivative(m.cfg, sp) == "forward_only"


# --------------------------------------------------------------------------------------------- 8. clear_caches

def _api_without_clear_caches():
    """mitjax/api.py with the genarr branch's jax.clear_caches() removed (the planted control), as a new module."""
    import types

    import mitjax.api as A
    src = Path(A.__file__).read_text()
    line = "            jax.clear_caches()\n            return out\n"
    assert src.count(line) == 1
    mod = types.ModuleType("_mjx_api_planted")
    exec(compile(src.replace(line, "            return out\n"), "_mjx_api_planted", "exec"), mod.__dict__)
    return mod


def _count_clear_caches(monkeypatch, api):
    import jax

    from mitjax.drivers import adjoint_run as AR
    from mitjax.drivers import grdchk as GD
    calls = []
    monkeypatch.setattr(jax, "clear_caches", lambda: calls.append("clear"))
    rundir = _new_dir("clear")
    monkeypatch.setattr(api.Experiment, "model", lambda self, out: SimpleNamespace(rundir=rundir))
    monkeypatch.setattr(api.Experiment, "_control",
                        lambda self, m, c: SimpleNamespace(kind="genarr", key="k", name="xx_x"))
    monkeypatch.setattr(api.Experiment, "_mode_arrays", lambda self, m, mo, *a: mo)

    class _G:
        def __init__(self, m, key):
            self.model = None

        def value_and_grad(self, model):
            calls.append("program")
            return 1.0, np.zeros(3)
    monkeypatch.setattr(AR, "GenarrAdjoint", _G)
    monkeypatch.setattr(GD, "write_adxx", lambda m, ctl, adxx: "adxx_x")
    g = object.__new__(api.Experiment).gradient("unused")
    assert g.fc == 1.0
    return calls


def test_gradient_clears_caches(monkeypatch):
    from mitjax import api
    assert _count_clear_caches(monkeypatch, api) == ["program", "clear"]
    assert _count_clear_caches(monkeypatch, _api_without_clear_caches()) == ["program"]        # planted


# ------------------------------------------------------------------------------------------ 12. build_notebooks

def _bn():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mjx_build_notebooks", paths.REPO / "tools" / "build_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nb_copy(bn, monkeypatch):
    d = _new_dir("notebooks") / "notebooks"
    shutil.copytree(bn.NB_DIR, d, symlinks=False)
    monkeypatch.setattr(bn, "NB_DIR", d)
    monkeypatch.setattr(bn, "SRC_DIR", d / "src")
    return d


def _outputs(path):
    return [c.get("outputs") for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]


def _planted_output(path):
    nb = json.loads(path.read_text())
    code = next(c for c in nb["cells"] if c["cell_type"] == "code")
    code["outputs"] = [{"name": "stdout", "output_type": "stream", "text": ["planted output\n"]}]
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def test_build_notebooks_names_and_kept_outputs(monkeypatch, capsys):
    bn = _bn()
    d = _nb_copy(bn, monkeypatch)
    nb01, nb03 = d / "01_quickstart.ipynb", d / "03_first_gradient.ipynb"
    _planted_output(nb01)
    before = _outputs(nb01)
    assert bn.main(["--check"]) == 0
    assert [s.stem[:2] for s in bn.select(["05", "06_adjoint_modes_cs32x15.ipynb", "notebooks/src/07_parallel.py"])] \
        == ["05", "06", "07"]
    with pytest.raises(SystemExit):
        bn.select(["99"])
    assert bn.main([]) == 0
    assert "unchanged notebooks/01_quickstart.ipynb" in capsys.readouterr().out
    assert _outputs(nb01) == before                                    # outputs of an unchanged notebook kept
    src03 = d / "src" / "03_first_gradient.py"
    src03.write_text(src03.read_text() + "\n# %% [markdown]\n# planted cell\n")
    assert bn.main(["--check"]) == 1
    assert bn.main(["03"]) == 0
    assert bn.main(["--check"]) == 0 and _outputs(nb01) == before
    assert all(o == [] for o in _outputs(nb03))                         # a changed notebook: rebuilt, no outputs
    # planted: the tool of before (every notebook rewritten) drops the committed outputs
    monkeypatch.setattr(bn, "build", lambda src: (bn.NB_DIR / (src.stem + ".ipynb")).write_text(
        json.dumps(bn.notebook(src.stem, src.read_text()), indent=1, ensure_ascii=False) + "\n") and "wrote")
    bn.main([])
    assert _outputs(nb01) != before


# ---------------------------------------------------------------------------------------- 4. TileMap (synthetic)

def test_tilemap_exch1_tile_order():
    from mitjax.api import TileMap
    sz = SimpleNamespace(sNx=3, sNy=2, OLx=1, OLy=1, nSx=2, nSy=3, nPx=1, nPy=1)
    cfg = SimpleNamespace(size=sz, cpp=SimpleNamespace(ALLOW_EXCH2=False))
    tm = TileMap.of(cfg)
    T = sz.nSx * sz.nSy
    a = np.zeros((T, 4, sz.sNy + 2, sz.sNx + 2))
    for t in range(T):
        a[t] = t + 1                                                     # tile number bi + (bj-1)*nSx
    g = tm.global_field(a)
    assert g.shape == (4, sz.sNy * sz.nSy, sz.sNx * sz.nSx)
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            blk = g[:, (bj - 1) * sz.sNy:bj * sz.sNy, (bi - 1) * sz.sNx:bi * sz.sNx]
            assert np.all(blk == bi + (bj - 1) * sz.nSx), (bi, bj)
    assert tm.global_field(np.arange(5.0)).shape == (5,)                # an untiled profile unchanged
    with pytest.raises(NotImplementedError, match="nPx"):
        TileMap.of(SimpleNamespace(size=dataclasses.replace(_Size(), nPx=2), cpp=cfg.cpp))


@dataclasses.dataclass(frozen=True)
class _Size:
    sNx: int = 3
    sNy: int = 2
    OLx: int = 1
    OLy: int = 1
    nSx: int = 2
    nSy: int = 3
    nPx: int = 1
    nPy: int = 1
