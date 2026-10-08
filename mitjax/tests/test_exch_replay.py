"""Exchange maps replayed from the exchange code (docs plan 20261006 Task 3), gated against the Fortran probe.

1. For EVERY registered probe map file (exch_maps.MAP_SHA256 and VARIANT_MAP_SHA256; sha256 checked on load) the
   run-time replay `exch_maps.exchange_maps(exp)` of that build has the same layout and, for every probe key, the
   same (src, comp, sign) bit for bit -- exch1 (eesupp/exch1_tables.py), exch0 (vermix: DISCONNECTED_TILES) and the
   single-facet exch2 build (global_ocean.90x40x15/code, pkg/exch2/exch2_cube_tables.py). The exch2 C-grid kinds
   (UVs, UVn, UV3s, Ds) are evaluated (EXCH2_RX2_CUBE sums, exchange.apply_rx2) on the probe's own coded fields and
   equal the probe's measured result.
2. Negative controls: an off-by-one planted in the exch1 replay (exchange width - 1) and in the exch2 replay
   (rx1_cube's exchWidthX - 1) differ from the probe.
3. Cache ($MJX_CACHE/exch_maps, keyed by layout + topology): the second load reads the file and gives the same maps;
   a renamed copy of the experiment hits the same file (no experiment name in the key); a planted cache file with
   another layout (another experiment's maps under this key's name) is refused.
Costs: about 3 min (configuration loads with cpp, host-side replays, one Model set-up).
"""

import os
import shutil
import time

import numpy as np
import pytest

from mitjax import paths
from mitjax.eesupp import exch_maps as EM


def _probe_registry():
    """[(map file, sha256, experiment, input dir)] of every registered probe map (the probe run's input dir)."""
    allp = {}
    for d in (EM.M1_PROBES, EM.M2_PROBES, EM.M3_PROBES, EM.M4_PROBES):
        allp.update(d)
    out = []
    for name, sha in EM.MAP_SHA256.items():
        exp = name[:name.index("-t")]
        out.append((name, sha, exp, allp[exp][0]))
    for name, sha in EM.VARIANT_MAP_SHA256.items():
        (exp, code), = [k for k in EM.VARIANT_PROBES if name.startswith(f"{k[0]}-{k[1]}-t")]
        out.append((name, sha, exp, EM.VARIANT_PROBES[(exp, code)][0]))
    return out


REGISTRY = _probe_registry()


@pytest.fixture
def cache(monkeypatch):
    """A new, empty MJX_CACHE for one test (and no in-process memo)."""
    d = paths.RUNS / "tests_port" / f"cache-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    monkeypatch.setattr(paths, "CACHE", d)
    monkeypatch.setattr(EM, "_REPLAY_MEMO", {})
    return d


def compare(replay, probe):
    """[differing keys] of the replay vs the probe map (layout first)."""
    from mitjax.eesupp.exchange import apply_rx2
    if replay.layout != probe.layout:
        return ["layout"]
    bad = []
    pu, pv = EM.probe_inputs(probe.layout, 1e6)
    fu, fv = pu.reshape(-1), pv.reshape(-1)
    for key, (s, c, g) in sorted(probe.maps.items()):
        kind = key[:-2] if key[-2:] in ("_u", "_v") else key
        if kind in replay.rx2:
            swap, passes = replay.rx2[kind][0], replay.rx2[kind][1]
            uo, vo = apply_rx2(passes, swap, fu, fv)
            got = np.asarray(uo if key.endswith("_u") else vo)
            own = fu if key.endswith("_u") else fv
            want = np.where(c == 0, own, np.where(c == 1, fu[s], fv[s]) * g)
            if got.tobytes() != want.tobytes():
                bad.append(key)
        elif key not in replay.maps or any(not np.array_equal(a, b) for a, b in zip(replay.maps[key], (s, c, g))):
            bad.append(key)
    return bad


def _load(exp, inp):
    from mitjax.config.params import load
    return load(exp, inp)


@pytest.mark.parametrize("name,sha,exp,inp", REGISTRY, ids=[r[0] for r in REGISTRY])
def test_replay_equals_probe(cache, name, sha, exp, inp):
    probe = EM.ExchangeMaps.load(EM.map_dir() / name, sha)
    rep = EM.exchange_maps(_load(exp, inp))
    assert rep.meta["source"] == "replay"
    bad = compare(rep, probe)
    print(name, rep.meta["family"], len(probe.maps), "keys, differing:", bad)
    assert bad == []


def test_registry_is_complete():
    """Every registered probe map is gated above (15 files at Task 3: 14 of MAP_SHA256 + the code_ad variant)."""
    assert len(REGISTRY) == len(EM.MAP_SHA256) + len(EM.VARIANT_MAP_SHA256) == 15
    fams = {}
    for name, sha, exp, inp in REGISTRY:
        e = _load(exp, inp)
        fams[name] = EM.replay_key(e)[0]
    assert sorted(set(fams.values())) == ["exch0", "exch1", "exch2"], fams


def test_key_reproducible_across_processes():
    """The cache key of an exch1, a single-facet exch2 and a cube build is the same in a new process and for a copy of
    the experiment elsewhere (no address, no path in the key; first measured otherwise: FIntArray's repr)."""
    import subprocess
    import sys
    from mitjax.config.params import load
    cases = (("tutorial_baroclinic_gyre", "input"), ("global_ocean.90x40x15", "input"),
             ("global_ocean.cs32x15", "input"))
    here = [EM.replay_key(load(e, i))[2] for e, i in cases]
    code = ("from mitjax.config.params import load; from mitjax.eesupp import exch_maps as EM; "
            f"print(' '.join(EM.replay_key(load(e, i))[2] for e, i in {cases!r}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env=dict(os.environ, JAX_PLATFORMS="cpu")).stdout.split()
    assert out == here, (out, here)
    dst = paths.RUNS / "tests_port" / f"keycopy-{os.getpid()}-{time.time_ns()}" / "global_ocean.cs32x15"
    dst.mkdir(parents=True)
    for d in ("code", "input"):
        shutil.copytree(paths.UPSTREAM / "verification" / "global_ocean.cs32x15" / d, dst / d)
    assert EM.replay_key(load("global_ocean.cs32x15", "input", exp_dir=dst))[2] == here[2]


def test_negative_control_exch1_off_by_one(cache, monkeypatch):
    from mitjax.eesupp import exch1_tables as X1
    orig = X1.exch1_rx
    monkeypatch.setattr(X1, "exch1_rx", lambda sym, c, sz, ox, oy, wx, wy, up: orig(sym, c, sz, ox, oy, wx - 1, wy,
                                                                                        up))
    name, sha, exp, inp = [r for r in REGISTRY if r[2] == "tutorial_baroclinic_gyre"][0]
    bad = compare(EM.exchange_maps(_load(exp, inp)), EM.ExchangeMaps.load(EM.map_dir() / name, sha))
    print("planted exch1 off-by-one differs in", bad)
    assert len(bad) == len(EM.SCALAR) + 2 * len(EM.VECTOR)


def test_negative_control_exch2_off_by_one(cache, monkeypatch):
    from mitjax.pkg.exch2 import exch2_cube_tables as C
    orig = C.rx1_cube

    def planted(sym, c, w2, updateCorners, *a, **k):
        if a or k:                                    # EXCH2_S3D_RX's width-1 call: unchanged
            return orig(sym, c, w2, updateCorners, *a, **k)
        return orig(sym, c, w2, updateCorners, exchWidthX=sym.L.OLx - 1)
    monkeypatch.setattr(C, "rx1_cube", planted)
    name, sha, exp, inp = [r for r in REGISTRY if r[0].startswith("global_ocean.90x40x15-t")][0]
    bad = compare(EM.exchange_maps(_load(exp, inp)), EM.ExchangeMaps.load(EM.map_dir() / name, sha))
    print("planted exch2 off-by-one differs in", bad)
    assert {"XY", "3D", "Z", "SMs", "As_u", "Bn_v"} <= set(bad)


def test_cache_hit_and_name_free_key(cache, monkeypatch):
    from mitjax.config.params import load
    exp = "tutorial_baroclinic_gyre"
    a = EM.exchange_maps(load(exp, "input"))
    f = EM.cache_path(a.meta["family"], a.layout, a.meta["cache_key"])
    assert a.meta["source"] == "replay" and f.is_file() and exp not in f.name
    monkeypatch.setattr(EM, "_REPLAY_MEMO", {})
    b = EM.exchange_maps(load(exp, "input"))
    assert b.meta["source"] == "cache" and sorted(a.maps) == sorted(b.maps)
    assert all(np.array_equal(x, y) for k in a.maps for x, y in zip(a.maps[k], b.maps[k]))
    # a renamed copy of the experiment: same layout and topology, same cache file
    src = paths.UPSTREAM / "verification" / exp
    dst = cache / "exps" / "my_renamed_gyre"
    dst.mkdir(parents=True)
    for d in ("code", "input"):
        shutil.copytree(src / d, dst / d)
    monkeypatch.setattr(EM, "_REPLAY_MEMO", {})
    c = EM.exchange_maps(load("my_renamed_gyre", "input", exp_dir=dst))
    assert c.meta["source"] == "cache" and c.meta["cache_key"] == a.meta["cache_key"]


def test_planted_wrong_layout_cache_refused(cache, monkeypatch):
    """Another experiment's maps written under this build's cache name: refused (stored key and layout differ)."""
    from mitjax.config.params import load
    other = EM.exchange_maps(load("tutorial_barotropic_gyre", "input"))          # 1 tile of 62x62
    e = load("tutorial_baroclinic_gyre", "input")                              # 4 tiles of 31x31
    family, L, key, _, _ = EM.replay_key(e)
    target = EM.cache_path(family, L, key)
    shutil.copyfile(other.meta["cache_file"], target)
    monkeypatch.setattr(EM, "_REPLAY_MEMO", {})
    with pytest.raises(ValueError, match="cache refused"):
        EM.exchange_maps(e)


def test_model_uses_the_replay(cache):
    """drivers.model.Model's default exchanger is the replay (no probe file read)."""
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import load_experiment, make_rundir
    exp_dir = paths.UPSTREAM / "verification" / "tutorial_barotropic_gyre"
    out = paths.RUNS / "tests_port" / f"model-{os.getpid()}-{time.time_ns()}"
    m = Model(load_experiment(exp_dir, "input"), make_rundir(exp_dir, "input", out))
    assert m.exch_maps is not None and m.exch_maps.meta["family"] == "exch1"
    assert sorted(m.ex.m) == sorted(m.exch_maps.maps)
