"""jaxdump shim on master (plan Task 4): anchors and their counts on 63cdc0b (planted drift fails), the iteration
shift, the 72-column limit, SUBSTEPS.md in sync, the stages compiled into each instrumented M1 build, invisibility
verdicts of every M1 variant (read from the run outputs), the invisibility checker's negative controls, and the
exchange-probe dumps of every M1 layout.

Oracle-dependent tests (compiled stages, invisibility verdicts, probe dumps) fail, not skip, when their data is
missing: they read the builds of JD_COMMIT and the runs of RUN_JOB (reference/jobs/jaxdump_runs.sbatch)."""

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mitjax import paths
from mitjax.io.dump import DumpSet, probe_decode

REPO = Path(__file__).resolve().parents[2]
JD = REPO / "reference" / "jaxdump"
UP7, PLAIN_COMMIT = "63cdc0b", "f48c062"
JD_COMMIT = "5f16129"      # instrumented builds: build job 27826872
RUN_JOB = "job27826873"    # plain / jdoff / jdon runs of every M1 variant (reference/jobs/jaxdump_runs.sbatch)
# optim rerun after the checker fix f245016 (its STDOUT ends at the grdchk STOP, before the communication statistics)
RUN_JOB_OF = {("tutorial_global_oce_optim", "input_ad"): "job27826998"}
VARIANTS = [("tutorial_barotropic_gyre", "input", "code"), ("tutorial_baroclinic_gyre", "input", "code"),
            ("advect_xy", "input", "code"), ("advect_xy", "input.ab3_c4", "code"), ("advect_xz", "input", "code"),
            ("advect_xz", "input.nlfs", "code"), ("advect_xz", "input.pqm", "code"),
            ("global_ocean.90x40x15", "input", "code"), ("tutorial_global_oce_optim", "input_ad", "code_ad")]
# stages each build must compile (beyond the core list): the per-package stage lists of plan Task 4
CORE = {"S00_begin", "G00_geometry", "X00_exch_probe", "S02_load_fields", "S04_oceanic_phys", "S06_dynamics",
        "S09_solve_for_pressure", "C01_cg2d_inputs", "C02_cg2d_solution", "S11_integr_continuity",
        "S15_tracers_correction", "S16_blocking_exchanges", "S17_monitor", "T11_temp_gT", "T13_temp_impl"}
PKG = {"global_ocean.90x40x15": {"P04_ggl90", "P05_gmredi_tensor", "S19_sbo_calc", "S01_update_rstar_F",
                                 "S07_update_rstar_T", "S12_calc_rstar"},
       "tutorial_global_oce_optim": {"S03_ctrl_map_forcing", "S18_cost_tile", "E01_cost_final", "P05_gmredi_tensor",
                                     "T01_residual_flow"},
       "advect_xz": {"S01_update_rstar_F", "S12_calc_rstar"}}
ABSENT = {"tutorial_barotropic_gyre": {"P04_ggl90", "S19_sbo_calc", "S03_ctrl_map_forcing", "E01_cost_final"}}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


instrument = _load(JD / "instrument.py", "_mjx_instrument")
invis = _load(JD / "invisibility.py", "_mjx_invisibility")


class _Planted(instrument.Source):
    """Master's sources with one text edit in one file (negative controls)."""

    def __init__(self, fname, old, new, count=1):
        super().__init__(git=paths.UPSTREAM)
        self.fname, self.old, self.new, self.count = fname, old, new, count

    def text(self, fname):
        t, label = super().text(fname)
        if fname == self.fname:
            assert self.old in t
            t = t.replace(self.old, self.new, self.count)
        return t, label


def test_anchor_counts_on_master():
    counts = instrument.anchor_counts(instrument.Source(git=paths.UPSTREAM))
    bad = {k: v for k, v in counts.items() if v[0] != v[1]}
    assert not bad, bad
    plan = instrument.plan(instrument.Source(git=paths.UPSTREAM))  # also asserts the iteration update's place
    stages = {r["stage"] for _, _, _, rows in plan.values() for r in rows}
    assert {e[4] for e in instrument.STAGES} <= stages
    assert {e[8]["pkg"] for e in instrument.STAGES} == set(instrument.PACKAGES)


def test_planted_anchor_drift_fails():
    dup = _Planted("dynamics.F", "         CALL CALC_PHI_HYD(", "         CALL CALC_PHI_HYD(\n         CALL CALC_PHI_HYD(")
    with pytest.raises(SystemExit, match="found 2 times, expected 1"):
        instrument.plan(dup)
    gone = _Planted("solve_for_pressure.F", "       CALL CG2D(", "       CALL CG2D_X(")
    with pytest.raises(SystemExit, match="found 0 times, expected 1"):
        instrument.plan(gone)


def _iteration_args(fortran_text):
    """Iteration arguments of the generated JAXDUMP calls (the argument before myThid of each call)."""
    calls = re.findall(r"CALL JAXDUMP_\w+\((.*?)\)\s*$", re.sub(r"\n     &", " ", fortran_text), re.M)
    return {c.split(",")[-2].strip() for c in calls}


def test_init_stages_anchors_and_iteration(tmp_path):
    """The init stages: anchors once each on master; INI_FIELDS and INITIALISE_VARIA have no myIter (IMPLICIT NONE: a
    myIter argument would not compile), so their calls pass nIter0 and READ_PICKUP's pass its myIter argument (=
    nIter0, ini_fields.F:38). Planted: the anchor exchange duplicated in read_pickup.F fails; the stage without the
    `iter` option generates myIter (the check bites)."""
    instrument.instrument(instrument.Source(git=paths.UPSTREAM), tmp_path / "jd")
    assert _iteration_args((tmp_path / "jd" / "initialise_varia.F").read_text()) == {"nIter0"}
    assert _iteration_args((tmp_path / "jd" / "ini_fields.F").read_text()) == {"nIter0"}
    assert _iteration_args((tmp_path / "jd" / "read_pickup.F").read_text()) == {"myIter"}
    rp = (tmp_path / "jd" / "read_pickup.F").read_text()
    assert rp.index("'I00_pickup_read'") < rp.index("CALL EXCH_UV_3D_RL( uVel, vVel")
    e = next(s for s in instrument.STAGES if s[4] == "I02_ini_fields")
    no_iter = {k: v for k, v in e[8].items() if k != "iter"}
    assert _iteration_args("\n".join(instrument._calls(e[4], e[5], e[6], no_iter))) == {"myIter"}
    dup = _Planted("read_pickup.F", "      CALL EXCH_UV_3D_RL( uVel, vVel, .TRUE., Nr, myThid )\n",
                   "      CALL EXCH_UV_3D_RL( uVel, vVel, .TRUE., Nr, myThid )\n" * 2)
    with pytest.raises(SystemExit, match="found 2 times, expected 1"):
        instrument.plan(dup)


def test_planted_iteration_update_moved_fails():
    # the counter update before DYNAMICS: records after it would carry the wrong iteration
    src = instrument.Source(git=paths.UPSTREAM)
    t, _ = src.text("forward_step.F")
    upd = "      myIter = nIter0 + iLoop\n"
    assert t.count(upd) == 1
    t2 = t.replace(upd, "").replace("        CALL DYNAMICS( myTime, myIter, myThid )\n",
                                    upd + "        CALL DYNAMICS( myTime, myIter, myThid )\n")
    planted = _Planted("forward_step.F", t, t2)
    with pytest.raises(SystemExit, match="iteration-counter update not between"):
        instrument.plan(planted)
    inside = _Planted("forward_step.F", upd, "#ifdef ALLOW_FOO\n" + upd + "#endif\n")
    with pytest.raises(SystemExit, match="inside a CPP block"):
        instrument.plan(inside)


def test_generated_code_fits_fixed_form(tmp_path):
    rows = instrument.instrument(instrument.Source(git=paths.UPSTREAM), tmp_path / "jd")
    for f in sorted((tmp_path / "jd").glob("*.F")):
        for n, ln in enumerate(f.read_text().split("\n"), 1):
            if ln[:1] not in ("C", "c", "*", "!", "#") and ln.strip():
                assert len(ln) <= 72, f"{f.name}:{n} has {len(ln)} columns"
    shift = [r for r in rows if r["stage"] in ("S00_begin", "(iteration shift 1)")]
    assert len(shift) == 2 and all(r["cpp"] == [] for r in shift)
    fs = (tmp_path / "jd" / "forward_step.F").read_text()
    assert fs.index("JAXDUMP_ITERSHIFT( 0 )") < fs.index("'S00_begin'") < fs.index("CALL DYNAMICS(") \
        < fs.index("JAXDUMP_ITERSHIFT( 1 )") < fs.index("'S07_update_rstar_T'")


def test_substeps_md_in_sync():
    text = (JD / "SUBSTEPS.md").read_text()
    table = instrument.markdown(instrument.Source(git=paths.UPSTREAM))
    assert table in text, "regenerate: python3 reference/jaxdump/instrument.py --git $MJX_UPSTREAM --markdown"


def _jd_bin(exp, code):
    return paths.REFERENCE / "bin" / f"{exp}-{code}-{UP7}-{JD_COMMIT}-jaxdump"


def test_compiled_stages_per_build():
    for exp, code in sorted({(e, c) for e, _, c in VARIANTS}):
        f = _jd_bin(exp, code) / "jaxdump_stages.txt"
        assert f.exists(), f"missing instrumented build {f.parent}"
        got = {ln.split()[0] for ln in f.read_text().split("\n") if ln.strip()}
        assert CORE | PKG.get(exp, set()) <= got, (exp, sorted(CORE | PKG.get(exp, set()) - got))
        assert not (ABSENT.get(exp, set()) & got), (exp, ABSENT.get(exp, set()) & got)


def test_invisibility_verdicts():
    for exp, inp, _ in VARIANTS:
        job = RUN_JOB_OF.get((exp, inp), RUN_JOB)
        f = paths.REFERENCE_RUNS / exp / inp / f"invisibility-{job}.txt"
        assert f.exists(), f"missing {f}"
        lines = f.read_text().split("\n")
        assert not any(ln.startswith("VISIBLE") for ln in lines), (exp, inp, lines[:3])
        assert sum(ln.startswith(f"INVISIBLE {exp}/{inp} dumps-off") for ln in lines) == 1
        assert sum(ln.startswith(f"INVISIBLE {exp}/{inp} dumps-on") for ln in lines) == 1
        for mode in ("plain", "jdoff", "jdon"):
            top = paths.REFERENCE_RUNS / exp / inp / f"{job}-{mode}"
            assert '"sha256"' in (top / "MANIFEST.json").read_text()


# ---- invisibility checker: negative controls on synthetic run directories ----------------------------------------
STDOUT = """(PID.TID 0000.0001) // Build host:        l10221.lvt.dkrz.de
(PID.TID 0000.0001) %MON dynstat_eta_max              =   1.2345678901234E-01
(PID.TID 0000.0001)           User time:   1.5420019626617432E-003
(PID.TID 0000.0001) // ======================================================
(PID.TID 0000.0001) // Tile <-> Tile communication statistics
(PID.TID 0000.0001) // ======================================================
(PID.TID 0000.0001) //         No. X exchanges =              0
PROGRAM MAIN: Execution ended Normally
"""


def _run_dir(top, stdout=STDOUT, state=b"\x00\x01\x02\x03"):
    rd = top / "rundir"
    rd.mkdir(parents=True)
    (rd / "output.txt").write_text(stdout)
    (rd / "T.0000000010.data").write_bytes(state)
    (rd / "data").symlink_to("/dev/null")  # an input link: not compared
    (top / "dumps").mkdir()
    (top / "dumps" / "jd_0000000000_t0001.bin").write_bytes(b"")
    return top


def _verdict(tmp_path, jdoff_kw, jdon_kw, plain_kw=None):
    p = _run_dir(tmp_path / "e" / "input" / "j-plain", **(plain_kw or {}))
    a = _run_dir(tmp_path / "e" / "input" / "j-jdoff", **jdoff_kw)
    b = _run_dir(tmp_path / "e" / "input" / "j-jdon", **jdon_kw)
    return invis.main([str(p), str(a), str(b)])


def test_invisibility_checker_negative_controls(tmp_path, capsys):
    exempt = STDOUT.replace("l10221", "l10261").replace("1.5420019626617432E-003", "1.9E-003")
    assert _verdict(tmp_path / "a", {"stdout": exempt}, {"stdout": exempt.replace("=              0",
                                                                                   "=             12")}) == 0
    assert "INVISIBLE e/input dumps-on" in capsys.readouterr().out
    # planted: one state byte, one %MON digit, a communication counter with dumps OFF
    assert _verdict(tmp_path / "b", {"state": b"\x00\x01\x02\x04"}, {}) == 1
    assert "file T.0000000010.data differs" in capsys.readouterr().out
    assert _verdict(tmp_path / "c", {}, {"stdout": STDOUT.replace("E-01", "E-02")}) == 1
    assert "VISIBLE e/input dumps-on: STDOUT line 2 differs" in capsys.readouterr().out
    assert _verdict(tmp_path / "d", {"stdout": STDOUT.replace("=              0", "=             12")}, {}) == 1
    assert "VISIBLE e/input dumps-off" in capsys.readouterr().out
    # a run that stops before the communication statistics: fine when all stop there, visible when one does
    head = STDOUT.split("(PID.TID 0000.0001) // ======")[0]
    assert _verdict(tmp_path / "e", {"stdout": head}, {"stdout": head}) == 1        # plain has the block
    out = capsys.readouterr().out
    assert "VISIBLE e/input dumps-off: STDOUT has" in out
    assert _verdict(tmp_path / "f", {"stdout": head}, {"stdout": head}, {"stdout": head}) == 0  # all stopped there
    assert "VISIBLE e/input dumps-on: communication statistics block in only one STDOUT" in out


def test_nc_build_stamp_blanking():
    def nc(host):
        name = b"build_host"
        return b"CDF\x01" + b"\x00" * 8 + len(name).to_bytes(4, "big") + name + b"\x00\x00" \
            + (2).to_bytes(4, "big") + len(host).to_bytes(4, "big") + host + b"\x00" * (-len(host) % 4) + b"DATA"
    a, b = nc(b"l10221.lvt"), nc(b"l10261.lvt")
    assert a != b and invis.blank_nc_stamps(a) == invis.blank_nc_stamps(b)
    assert invis.blank_nc_stamps(a[:-1] + b"X") != invis.blank_nc_stamps(b)  # data bytes still compared


# ---- exchange probe dumps of every M1 layout -----------------------------------------------------------------------
LAYOUTS = {"tutorial_barotropic_gyre": "input", "advect_xy": "input", "advect_xz": "input",
           "tutorial_baroclinic_gyre": "input", "global_ocean.90x40x15": "input",
           "tutorial_global_oce_optim": "input_ad"}
PROBE_FIELDS = ("xT", "xUVs_u", "xUVs_v", "xUVn_u", "xUVn_v", "xZ", "xAs_u", "xAs_v", "xAn_u", "xAn_v", "xBs_u",
                "xBs_v", "xBn_u", "xBn_v", "x3D", "xSMs", "xDs_u", "xDs_v", "xUV3s_u", "xUV3s_v", "xS3D")


# exp: (halo points of all tiles, points changed by EXCH_S3D_RL, points changed by the vector exchanges)
PROBE_CHANGED = {"tutorial_barotropic_gyre": (512, 248, 512), "advect_xy": (432, 120, 432),
                 "advect_xz": (304, 52, 304), "tutorial_baroclinic_gyre": (1056, 496, 1056),
                 "global_ocean.90x40x15": (5616, 1440, 5610), "tutorial_global_oce_optim": (1104, 520, 1104)}
VECTOR_FIELDS = ("xUVs_u", "xUVs_v", "xUVn_u", "xUVn_v", "xDs_u", "xDs_v", "xUV3s_u", "xUV3s_v")


def probe_summary(ds):
    """Per probe field: points changed by the exchange, halo-sourced copies, sign flips; checks every value decodes,
    interior points are unchanged, and every copy comes from the same global point modulo the face size."""
    it = ds.iterations()[0]
    cbase, ntl = ds.scalar(it, "X00_exch_probe", "xCbase"), ds.scalar(it, "X00_exch_probe", "xNtiles")
    shapes = ds.face_shapes()
    out = {}
    for fld in PROBE_FIELDS:
        recs = ds.tiles(it, "X00_exch_probe", fld)
        changed = halo_src = flips = 0
        for t, r in recs.items():
            nyp, nxp = r.shape[1:]
            d = probe_decode(r.data[0], cbase, nxp, nyp)
            ja, ia = np.indices((nyp, nxp))
            own = (d["tile"] == t) & (d["ja"] == ja) & (d["ia"] == ia)
            interior = (ja >= r.oly) & (ja < nyp - r.oly) & (ia >= r.olx) & (ia < nxp - r.olx)
            assert own[interior].all(), (fld, t, "interior point changed")
            assert np.all(d["comp"] == (2 if fld.endswith("_v") else 1)), (fld, t)
            info = {tt: ds.tiles_info[tt] for tt in np.unique(d["tile"])}
            for tt in info:
                assert 1 <= tt <= ntl
            src_face = np.vectorize(lambda x: info[x][0])(d["tile"])
            src_ig = np.vectorize(lambda x: info[x][1])(d["tile"]) + d["ia"] - r.olx
            src_jg = np.vectorize(lambda x: info[x][2])(d["tile"]) + d["ja"] - r.oly
            ny, nx = shapes[r.face]
            m = ~own
            assert np.all(src_face[m] == r.face), (fld, t, "copy across faces in a one-face layout")
            assert np.all((src_ig[m] - (r.tbx + ia[m] - r.olx)) % nx == 0), (fld, t, "x source not periodic image")
            assert np.all((src_jg[m] - (r.tby + ja[m] - r.oly)) % ny == 0), (fld, t, "y source not periodic image")
            sinfo = np.array([info[x][3:] for x in d["tile"][m]]).reshape(-1, 4)  # snx, sny, olx, oly of source
            s_int = ((d["ia"][m] >= sinfo[:, 2]) & (d["ia"][m] < sinfo[:, 0] + sinfo[:, 2])
                     & (d["ja"][m] >= sinfo[:, 3]) & (d["ja"][m] < sinfo[:, 1] + sinfo[:, 3]))
            changed += int(m.sum())
            halo_src += int((~s_int).sum())
            flips += int((d["sign"] < 0).sum())
        out[fld] = (changed, halo_src, flips)
    zeros = {}
    for fld in [k[2] for k in ds.keys(it) if k[1] == "X00_exch_probe" and k[2][:2] in ("zp", "zm")]:
        v = ds.field(it, "X00_exch_probe", fld)
        zeros[fld] = (int(np.count_nonzero(v)), int(np.signbit(v).sum()), v.size)
    return out, zeros


def test_exchange_probe_all_layouts():
    report = {}
    for exp, inp in LAYOUTS.items():
        d = paths.REFERENCE_RUNS / exp / inp / f"{RUN_JOB}-jdon" / "dumps"
        assert d.is_dir(), f"missing probe dumps {d}"
        ds = DumpSet(d)
        out, zeros = probe_summary(ds)
        report[exp] = out
        for fld in ("xT", "x3D", "xUVn_u", "xUVn_v"):
            assert out[fld][0] > 0, (exp, fld, "exchange changed no halo point")
        # no cube-sphere layout in M1: no sign flips anywhere; +0 stays +0, -0 stays -0 (copies only)
        assert all(v[2] == 0 for v in out.values()), (exp, {k: v for k, v in out.items() if v[2]})
        for fld, (nonzero, neg, size) in zeros.items():
            assert nonzero == 0, (exp, fld)
            assert neg == (size if fld.startswith("zm") else 0), (exp, fld, neg, size)
    # measured counts (run job 27826873): every exchange fills every halo point, except EXCH_S3D_RL (halo width 1;
    # its 4 ring corners per tile stay unfilled, except on advect_xz, sNy=1) and the exch2 vector exchanges of
    # global_ocean (6 halo points unchanged); no copy comes from a halo point in any M1 layout
    for exp, (n_halo, n_s3d, n_vec) in PROBE_CHANGED.items():
        out = report[exp]
        for fld, (changed, halo_src, _) in out.items():
            want = n_s3d if fld == "xS3D" else n_vec if fld in VECTOR_FIELDS else n_halo
            assert (changed, halo_src) == (want, 0), (exp, fld, changed, halo_src, want)


def test_probe_decoder_negative_control():
    with pytest.raises(ValueError, match="not exchange-probe codes"):
        probe_decode(np.array([1e4 + 0.5]), 1e4, 10, 10)
    with pytest.raises(ValueError):
        probe_decode(np.array([3e4 + 5]), 1e4, 10, 10)  # component 3 does not exist
    d = probe_decode(np.array([-(2e4 + 1 + (1 * 10 + 3) * 10 + 4)]), 1e4, 10, 10)  # tile 2, ja 3, ia 4, v, flipped
    assert (int(d["comp"][0]), int(d["tile"][0]), int(d["ja"][0]), int(d["ia"][0]), int(d["sign"][0])) == \
        (2, 2, 3, 4, -1)


# ---- tools/diffdump.py and tools/step_vs_dump.py: negative controls on synthetic dump sets ---------------------------
sys.path.insert(0, str(REPO))
from mitjax.io.dump import write_records  # noqa: E402
from tools import diffdump, step_vs_dump  # noqa: E402

NR, OL, SN = 2, 1, 3          # Nr, halo, tile size (sNx = sNy)


def _arr(nz, fill=None, seed=0):
    a = np.random.default_rng(seed).standard_normal((nz, SN + 2 * OL, SN + 2 * OL)) + 3.0
    if fill is not None:
        a[:] = fill
    return a


def _dumpset(d, fields, with_masks=True, masks=("hFacC", "hFacW", "hFacS")):
    """One tile; fields: list of (stage, field, kind, values). hFac: level 2 dry at interior point (1, 1)."""
    d.mkdir(parents=True)
    recs, seq = [], 0
    if with_masks:
        h = np.ones((NR, SN + 2 * OL, SN + 2 * OL))
        h[1, OL + 1, OL + 1] = 0.0
        for name in masks:
            seq += 1
            recs.append(dict(iter=0, seq=seq, stage="G00_geometry", field=name, kind=name[-1], tile=1, face=0,
                             tbx=0, tby=0, olx=OL, oly=OL, values=h))
    for stage, fld, kind, v in fields:
        seq += 1
        recs.append(dict(iter=0, seq=seq, stage=stage, field=fld, kind=kind, tile=1, face=0, tbx=0, tby=0,
                         olx=OL, oly=OL, values=v))
    write_records(d / "jd_0000000000_t0001.bin", recs)
    return DumpSet(d)


def _base():
    return [("S00_begin", "theta", "C", _arr(NR, seed=1)), ("S01_dyn", "etaN", "C", _arr(1, seed=2)),
            ("S01_dyn", "vort_k002", "Z", _arr(1, seed=3)), ("C02_cg2d", "numIters", "N", _arr(1, fill=37.0))]


def _with(fields, stage, fld, fn):
    out = []
    for s, f, k, v in fields:
        v = v.copy()
        if (s, f) == (stage, fld):
            fn(v)
        out.append((s, f, k, v))
    return out


def test_diffdump_identical_and_first_difference(tmp_path):
    ref = _dumpset(tmp_path / "ref", _base())
    rows = diffdump.compare(ref, _dumpset(tmp_path / "same", _base()))
    assert {r[2].status for r in rows} == {"same"}
    planted = _with(_base(), "S01_dyn", "etaN", lambda v: v.__setitem__((0, OL, OL), v[0, OL, OL] * (1 + 1e-9)))
    rows = diffdump.compare(ref, _dumpset(tmp_path / "bad", planted))
    bad = [r for r in rows if r[2].status in diffdump.BAD]
    assert bad[0][0] == (0, "S01_dyn", "etaN") and bad[0][2].n_over == 1      # first difference, one point over
    assert diffdump.main([str(tmp_path / "ref"), str(tmp_path / "bad")]) == 1


def test_diffdump_compares_dry_points(tmp_path):
    """Masks are for reporting, not for hiding points (PORTING_RULES §5): a change at a dry point fails and is
    reported as dry. (Before the M0 hardening this test asserted the opposite: a dry change was "same".)"""
    ref = _dumpset(tmp_path / "ref", _base())
    dry = _with(_base(), "S00_begin", "theta", lambda v: v.__setitem__((1, OL + 1, OL + 1), 99.0))  # level 2 dry
    r = [x[2] for x in diffdump.compare(ref, _dumpset(tmp_path / "dry", dry)) if x[0][2] == "theta"][0]
    assert (r.status, r.n_over, r.over) == ("FAIL", 1, {"wet": 0, "dry": 1, "halo": 0})
    wet = _with(_base(), "S00_begin", "theta", lambda v: v.__setitem__((0, OL + 1, OL + 1), 99.0))  # level 1 wet
    r = [x[2] for x in diffdump.compare(ref, _dumpset(tmp_path / "wet", wet)) if x[0][2] == "theta"][0]
    assert (r.status, r.over) == ("FAIL", {"wet": 1, "dry": 0, "halo": 0})
    # every point is compared and counted: 2 levels x 5 x 5 = 50 = 40 interior (1 dry) + ... halo
    assert r.n == NR * (SN + 2 * OL) ** 2 and r.points == {"wet": 17, "dry": 1, "halo": 32}


def test_diffdump_halo_and_dry_planted_cases(tmp_path, capsys):
    """The review's three planted cases (REVIEW_M0 #2), each a failure by default: NaN at a dry interior point, NaN
    at a wet halo point, 1e6 at a wet halo point; the number of points over is gated (two points -> 2)."""
    ref = _dumpset(tmp_path / "ref", _base())
    cases = {"drynan": (1, OL + 1, OL + 1, np.nan, "NAN", "dry 1"),
             "halonan": (0, 0, OL + 1, np.nan, "NAN", "halo 1"),
             "halobig": (0, 0, OL + 1, 1e6, "FAIL", "halo 1")}
    for name, (k, j, i, val, want, where) in cases.items():
        planted = _with(_base(), "S00_begin", "theta", lambda v: v.__setitem__((k, j, i), val))
        r = [x[2] for x in diffdump.compare(ref, _dumpset(tmp_path / name, planted)) if x[0][2] == "theta"][0]
        assert r.status == want and where in r.where(), (name, r)
        assert diffdump.main([str(tmp_path / "ref"), str(tmp_path / name)]) == 1, name
        assert f"FIRST DIFFERENCE: iter 0, stage S00_begin, field theta" in capsys.readouterr().out
    two = _with(_base(), "S00_begin", "theta", lambda v: (v.__setitem__((0, 0, 1), 1e6), v.__setitem__((1, 1, 1), 7.)))
    r = [x[2] for x in diffdump.compare(ref, _dumpset(tmp_path / "two", two)) if x[0][2] == "theta"][0]
    assert r.n_over == 2 and r.over == {"wet": 1, "dry": 0, "halo": 1}
    # tolerance relative to the field's max abs over the compared points: 1e-14 of max|ref| passes at rtol 1e-13
    scale = float(np.abs(ref.field(0, "S00_begin", "theta")).max())
    tiny = _with(_base(), "S00_begin", "theta", lambda v: v.__setitem__((0, 0, 0), v[0, 0, 0] + 1e-14 * scale))
    r = [x[2] for x in diffdump.compare(ref, _dumpset(tmp_path / "tiny", tiny)) if x[0][2] == "theta"][0]
    assert r.status == "ok" and r.n_over == 0


def test_diffdump_interior_only_is_labelled_opt_in(tmp_path, capsys):
    ref = _dumpset(tmp_path / "ref", _base())
    halobig = _with(_base(), "S00_begin", "theta", lambda v: v.__setitem__((0, 0, OL + 1), 1e6))
    _dumpset(tmp_path / "hb", halobig)
    assert diffdump.main([str(tmp_path / "ref"), str(tmp_path / "hb")]) == 1                # default: halos compared
    capsys.readouterr()
    assert diffdump.main([str(tmp_path / "ref"), str(tmp_path / "hb"), "--interior-only", "--all"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("INTERIOR-ONLY") and all("INTERIOR-ONLY" in ln for ln in out.splitlines() if ln.strip())
    # a non-finite candidate value is a failure anywhere, also where an interior-only comparison does not look
    halonan = _with(_base(), "S00_begin", "theta", lambda v: v.__setitem__((0, 0, OL + 1), np.inf))
    _dumpset(tmp_path / "hn", halonan)
    assert diffdump.main([str(tmp_path / "ref"), str(tmp_path / "hn"), "--interior-only"]) == 1


def test_diffdump_nan_controls(tmp_path, capsys):
    ref = _dumpset(tmp_path / "ref", _base())
    nan_test = _with(_base(), "S00_begin", "theta", lambda v: v.__setitem__((0, OL, OL), np.nan))
    rows = diffdump.compare(ref, _dumpset(tmp_path / "t", nan_test))
    assert [r[2].status for r in rows if r[0][2] == "theta"] == ["NAN"]
    # a NaN in the REFERENCE must not pass either (Python's max() never picks a NaN)
    ref_nan = _dumpset(tmp_path / "rn", nan_test)
    assert [r[2].status for r in diffdump.compare(ref_nan, _dumpset(tmp_path / "t2", _base()))
            if r[0][2] == "theta"] == ["NAN"]
    assert diffdump.main([str(tmp_path / "rn"), str(tmp_path / "rn")]) == 1


def test_diffdump_zero_and_missing(tmp_path):
    zero = _with(_base(), "S01_dyn", "etaN", lambda v: v.__setitem__(slice(None), 0.0))
    rows = diffdump.compare(_dumpset(tmp_path / "z1", zero), _dumpset(tmp_path / "z2", zero))
    assert [r[2].status for r in rows if r[0][2] == "etaN"] == ["ZERO"]
    assert diffdump.main([str(tmp_path / "z1"), str(tmp_path / "z2")]) == 0      # ZERO alone is a warning
    # a zero reference: any nonzero candidate value is over (no division by a zero scale)
    rows = diffdump.compare(_dumpset(tmp_path / "z3", zero), _dumpset(tmp_path / "nz", _base()))
    assert [r[2].status for r in rows if r[0][2] == "etaN"] == ["FAIL"]
    short = [f for f in _base() if f[1] != "numIters"]
    rows = diffdump.compare(_dumpset(tmp_path / "r", _base()), _dumpset(tmp_path / "s", short))
    assert [r[2].status for r in rows if r[0][2] == "numIters"] == ["MISSING"]


def test_diffdump_refuses_without_masks(tmp_path, capsys):
    _dumpset(tmp_path / "nomask", _base(), with_masks=False)
    assert diffdump.main([str(tmp_path / "nomask"), str(tmp_path / "nomask")]) == 2
    assert "REFUSED" in capsys.readouterr().out
    # hFacC only: the C records could be masked, the Z record cannot
    only_c = _dumpset(tmp_path / "onlyc", _base(), masks=("hFacC",))
    with pytest.raises(diffdump.MaskError):
        diffdump.compare(only_c, only_c)
    odd = [("S09", "x", "C", _arr(NR + 3))]                                          # nz matching no rule
    with pytest.raises(diffdump.MaskError, match="matches no mask rule"):
        diffdump.compare(_dumpset(tmp_path / "odd", odd), _dumpset(tmp_path / "odd2", odd))


def test_step_vs_dump_first_differing_stage(tmp_path):
    oracle = _dumpset(tmp_path / "o", _base())
    good = {(s, f): oracle.field(0, s, f) for _, s, f in oracle.keys(0) if s != "G00_geometry"}
    per, first, missing, declared, n = step_vs_dump.run(oracle, step_vs_dump.CandidateSet(oracle, 0, good), 0,
                                                        not_ported=["G00_geometry/hFac?"])
    assert first is None and missing == [] and n == 4
    assert [p[0] for p in per] == ["S00_begin", "S01_dyn", "C02_cg2d"]
    assert {k[2] for k in declared} == {"hFacC", "hFacW", "hFacS"}
    bad = dict(good)
    bad[("S01_dyn", "vort_k002")] = good[("S01_dyn", "vort_k002")] + 1e-6
    bad[("C02_cg2d", "numIters")] = good[("C02_cg2d", "numIters")] + 1
    _, first, _, _, _ = step_vs_dump.run(oracle, step_vs_dump.CandidateSet(oracle, 0, bad), 0,
                                         not_ported=["G00_geometry/*"])
    assert first[0] == "S01_dyn" and first[1][1] == "vort_k002"
    with pytest.raises(KeyError, match="never dumped"):
        step_vs_dump.CandidateSet(oracle, 0, {("S05", "x"): good[("S01_dyn", "etaN")]})
    with pytest.raises(ValueError, match="candidate shape"):
        step_vs_dump.CandidateSet(oracle, 0, {("S01_dyn", "etaN"): good[("S00_begin", "theta")]})


def test_step_vs_dump_fails_on_keys_not_computed(tmp_path, capsys):
    """REVIEW_M0 #6: a candidate that does not provide an oracle key it is asked to compare fails (exit 1, no
    ALL STAGES AGREE) unless the key is declared not ported; a run that compares nothing fails."""
    oracle = _dumpset(tmp_path / "o", _base())
    _, first, missing, _, n = step_vs_dump.run(oracle, step_vs_dump.CandidateSet(oracle, 0, {}), 0)
    assert first is not None and first[1][0] == step_vs_dump.NOT_COMPUTED and n == 0 and len(missing) == 7
    only_masks = tmp_path / "c_masks"
    _dumpset(only_masks, [])                                           # the hFac records, nothing else
    o = str(tmp_path / "o")
    assert step_vs_dump.main([o, "--it", "0", "--candidate-dir", str(only_masks)]) == 1
    out = capsys.readouterr().out
    assert "NOT COMPUTED by the candidate" in out and "ALL STAGES AGREE" not in out
    # declared not ported: those keys are listed, the rest is compared and agrees
    assert step_vs_dump.main([o, "--it", "0", "--candidate-dir", str(only_masks),
                              "--not-ported", "S0*/*", "C02_cg2d/*"]) == 0
    out = capsys.readouterr().out
    assert "not ported yet (declared, not compared): 4" in out and "ALL STAGES AGREE" in out
    # nothing compared at all
    assert step_vs_dump.main([o, "--it", "0", "--candidate-dir", str(only_masks), "--stage-prefix", "S",
                              "--not-ported", "S*"]) == 1
    out = capsys.readouterr().out
    assert "NOTHING COMPARED" in out and "ALL STAGES AGREE" not in out


# ---- jaxdump3 (build 275a02b, job 27829310): initialisation stages, rStarDhCDt at S12, mixed signed-zero probes --------
rr = _load(REPO / "reference" / "reference_runs.py", "_mjx_reference_runs_jd")
JD3_COMMIT = rr.BUILDS["jaxdump3"][0].split("-")[0]
INIT = ("I00_pickup_read", "I01_read_pickup", "I02_ini_fields")
# variants with select_rStar /= 0 (data PARM01): S12_calc_rstar runs only then (forward_step.F:938-946), although
# advect_xz/code defines NONLIN_FRSURF for all three advect_xz variants
RSTAR = {("advect_xz", "input.nlfs"), ("global_ocean.90x40x15", "input")}
PICKUP_UV = (("uVel", "vVel"), ("guNm1", "gvNm1"))
# halo points where a plain copy of the interior -0 differs from the oracle (+0), measured in job27828737-jdon
PICKUP_COPY_MISMATCH = {"uVel": 3543, "vVel": 4687, "guNm1": 144, "gvNm1": 60}


def _jd3(exp, inp):
    run = rr.find_run(exp, inp, "jdon3")
    d = rr.run_top(run) / "dumps"
    assert d.is_dir(), f"missing dumps {d}"
    return run, DumpSet(d)


def test_jd3_compiled_stages():
    for exp in sorted({e for e, _ in rr.JD3}):
        f = paths.REFERENCE / "bin" / f"{exp}-code-{UP7}-{JD3_COMMIT}-jaxdump" / "jaxdump_stages.txt"
        assert f.exists(), f"missing instrumented build {f.parent}"
        got = {ln.split()[0] for ln in f.read_text().split("\n") if ln.strip()}
        assert CORE | PKG.get(exp, set()) | set(INIT) <= got, (exp, sorted(CORE | set(INIT) - got))


def test_jd3_invisibility_verdicts():
    for (exp, inp), (job, _) in rr.JD3.items():
        f = paths.REFERENCE_RUNS / exp / inp / f"invisibility-job{job}.txt"
        assert f.exists(), f"missing {f}"
        lines = f.read_text().split("\n")
        assert not any(ln.startswith("VISIBLE") for ln in lines), (exp, inp, lines[:3])
        for mode in ("dumps-off", "dumps-on"):
            assert sum(ln.startswith(f"INVISIBLE {exp}/{inp} {mode}") for ln in lines) == 1, (exp, inp, mode)


def test_jd3_steps_init_stages_and_rstar():
    """Iterations = the registered steps; the initialisation stages come first in iteration nIter0 (I00/I01 only in
    the restart); every step reaches S17_monitor; under r* S12_calc_rstar holds rStarDhCDt, equal bitwise to G00 group
    R of the next dumped step (no routine between MONITOR and that G00 writes it, forward_step.F:451-475); without r*
    (select_rStar = 0) S12 does not run."""
    for exp, inp in rr.JD3:
        run, ds = _jd3(exp, inp)
        its = ds.iterations()
        assert tuple(its) == run["steps"], (exp, inp, its)
        n0 = its[0]
        first = ds.stages(n0)
        restart = exp == "global_ocean.90x40x15"
        want = list(INIT) if restart else ["I02_ini_fields"]
        assert first[:len(want)] == want and not (set(INIT) - set(want)) & set(first), (exp, inp, first[:4])
        for it in its:
            assert "S17_monitor" in ds.stages(it), (exp, inp, it)
            if (exp, inp) in RSTAR:
                assert (it, "S12_calc_rstar", "rStarDhCDt") in ds.index, (exp, inp, it)
                if it + 1 in its:
                    a = ds.field(it, "S12_calc_rstar", "rStarDhCDt")
                    b = ds.field(it + 1, "G00_geometry", "rStarDhCDt")
                    assert np.array_equal(a.view(np.int64), b.view(np.int64)), (exp, inp, it)
            else:
                assert (it, "S12_calc_rstar", "rStarDhCDt") not in ds.index


def exch2_put_rx2_halo(ds, it, stage, uname, vname):
    """uVel-like / vVel-like fields after EXCH_UV_3D_RL(withSigns) on an exch2 layout without rotation, from the
    INTERIOR of (it, stage), literally as pkg/exch2/exch2_put_rx2.template:229-230 and 317-318 form the buffers:
    u halo = sa1*u + sa2*v with sa1 = pi(1) = 1, sa2 = pj(1) = 0; v halo = sa1*u + sa2*v with sa1 = pi(2) = 0,
    sa2 = pj(2) = 1, u and v taken at the same SOURCE index (isl, jsl). Source indices: the X00 probe (xUV3s_u/_v).
    Returns (literal, plain copy) for u and v, each [tile, k, j, i]; halo points the exchange does not fill keep
    their (it, stage) value."""
    cb = ds.scalar(it, "X00_exch_probe", "xCbase")
    U, V = ds.field(it, stage, uname), ds.field(it, stage, vname)
    nyp, nxp = U.shape[2:]
    one, zero = np.float64(1.0), np.float64(0.0)
    out = []
    for comp, A in (("u", U), ("v", V)):
        d = probe_decode(ds.field(it, "X00_exch_probe", f"xUV3s_{comp}")[:, 0], cb, nxp, nyp)
        us = np.moveaxis(U[d["tile"] - 1, :, d["ja"], d["ia"]], -1, 1)
        vs = np.moveaxis(V[d["tile"] - 1, :, d["ja"], d["ia"]], -1, 1)
        with np.errstate(invalid="ignore"):
            lit = one * us + zero * vs if comp == "u" else zero * us + one * vs
        out.append((lit, us if comp == "u" else vs))
    return out


def _bits_differ(a, b):
    return np.ascontiguousarray(a, np.float64).view(np.int64) != np.ascontiguousarray(b, np.float64).view(np.int64)


def test_pickup_halo_signed_zeros_from_exch2_buffer():
    """global_ocean (exch2, 36 tiles 10x10, OLx=OLy=3) restart: (a) before READ_PICKUP's exchanges the interior is the
    pickup file and every halo point is +0 (INI_DYNVARS); (b) the exchanges at read_pickup.F:538/556 give the halos
    of the exch2 buffer arithmetic sa1*u + sa2*v bit for bit (every halo point, all levels), and a plain copy
    differs at exactly the measured points (oracle +0, copied -0); (c) nothing after READ_PICKUP changes these fields
    before the first step (I01 == I02 == S00_begin, all points)."""
    exp, inp = "global_ocean.90x40x15", "input"
    run, ds = _jd3(exp, inp)
    it = run["steps"][0]
    raw = np.fromfile(rr.run_top(run) / "rundir" / "pickup.0000036000", ">f8").reshape(-1, 40, 90)
    first = {"uVel": 0, "vVel": 15, "guNm1": 60, "gvNm1": 75}           # fldList of pickup.0000036000.meta, Nr = 15
    for uname, vname in PICKUP_UV:
        for name in (uname, vname):
            a = ds.field(it, "I00_pickup_read", name)
            halo = np.ones(a.shape, bool)
            halo[:, :, 3:-3, 3:-3] = False
            assert np.all(a[halo] == 0) and not np.signbit(a[halo]).any(), name            # (a) halos +0
            g = ds.global_field(it, "I00_pickup_read", name)[1]
            assert not _bits_differ(g, raw[first[name]:first[name] + 15]).any(), name    # (a) interior = file
        (lit_u, copy_u), (lit_v, copy_v) = exch2_put_rx2_halo(ds, it, "I00_pickup_read", uname, vname)
        for name, lit, copy in ((uname, lit_u, copy_u), (vname, lit_v, copy_v)):
            got = ds.field(it, "I01_read_pickup", name)
            halo = np.ones(got.shape, bool)
            halo[:, :, 3:-3, 3:-3] = False
            assert int((_bits_differ(lit, got) & halo).sum()) == 0, name                   # (b) literal: exact
            bad = _bits_differ(copy, got) & halo
            assert int(bad.sum()) == PICKUP_COPY_MISMATCH[name], (name, int(bad.sum()))   # (b) copy: measured
            assert np.all((copy[bad] == 0) & np.signbit(copy[bad]) & (got[bad] == 0) & ~np.signbit(got[bad]))
            for later in ("I02_ini_fields", "S00_begin"):                                   # (c)
                assert not _bits_differ(ds.field(it, later, name), got).any(), (name, later)


def test_mixed_signed_zero_probes():
    """Mixed pairs through every vector exchange with signs (zu*: u=-0, v=+0; zv*: u=+0, v=-0). exch1 layouts copy:
    the -0 component stays -0 at every point. exch2 (global_ocean): the C-grid vector exchanges EXCH_UV_XY_RL,
    EXCH_UV_3D_RL, EXCH_UV_DGRID_3D_RL go through EXCH2_RX2_CUBE, whose buffer is sa1*u + sa2*v
    (exch2_put_rx2.template:229-230, 317-318): every filled halo point of the -0 component becomes +0 (-0 + 0*(+0)),
    only the interior and the 6 halo points the exchange does not fill keep -0; the A- and B-grid exchanges use
    EXCH2_RX1_CUBE (plain copies). The +0 component stays +0 everywhere. (Measured, jaxdump3 runs.)"""
    for exp, inp in rr.JD3:
        _, ds = _jd3(exp, inp)
        it = ds.iterations()[0]
        for g in ("UVs", "As", "Bs", "Ds", "UV3s"):
            zu_u, zu_v = (ds.field(it, "X00_exch_probe", f"zu{g}_{c}") for c in "uv")
            zv_u, zv_v = (ds.field(it, "X00_exch_probe", f"zv{g}_{c}") for c in "uv")
            for a in (zu_u, zu_v, zv_u, zv_v):
                assert not np.any(a), (exp, inp, g)
            assert not np.signbit(zu_v).any() and not np.signbit(zv_u).any(), (exp, inp, g)
            n, nint = zu_u.size, zu_u.shape[0] * (zu_u.shape[-2] - 6) * (zu_u.shape[-1] - 6)
            if exp == "global_ocean.90x40x15" and g in ("UVs", "UV3s", "Ds"):
                assert nint == 3600 and int(np.signbit(zu_u).sum()) == int(np.signbit(zv_v).sum()) == nint + 6, \
                    (exp, g, int(np.signbit(zu_u).sum()))
            else:
                assert int(np.signbit(zu_u).sum()) == int(np.signbit(zv_v).sum()) == n, (exp, inp, g)
