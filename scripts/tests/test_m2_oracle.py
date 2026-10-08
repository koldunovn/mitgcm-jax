"""M2 Fortran oracle (lane A session 6): instrumented builds of the M2 codes, invisibility of the shim, the exchange
probes on the cube layouts (exch2, 6 faces) with their mixed signed-zero records, the ptracers dump group, and the
probe-map tool lane B builds the cube maps from (reference/probe_map.py).

Oracle-dependent: fails (never skips) while a build or run is missing. Builds: plain 06b4417 (job 27832154), jaxdump
8316e95-jaxdump (job 27832225); runs: the invisibility triples of reference/reference_runs.py M2_TRIPLE.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mitjax import paths
from mitjax.io.dump import DumpSet

REPO = Path(__file__).resolve().parents[2]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load(REPO / "reference" / "reference_runs.py", "_mjx_rr_m2")
pm = _load(REPO / "reference" / "probe_map.py", "_mjx_probe_map")
invis = _load(REPO / "reference" / "jaxdump" / "invisibility.py", "_mjx_invis_m2")
UP7, JD = "63cdc0b", "8316e95-jaxdump"
CUBES = [("adjustment.cs-32x32x1", "input"), ("solid-body.cs-32x32x1", "input"), ("advect_cs", "input"),
         ("global_ocean.cs32x15", "input")]
CORE = {"S00_begin", "G00_geometry", "X00_exch_probe", "S02_load_fields", "S04_oceanic_phys", "S06_dynamics",
        "S09_solve_for_pressure", "C01_cg2d_inputs", "C02_cg2d_solution", "S11_integr_continuity",
        "S15_tracers_correction", "S16_blocking_exchanges", "S17_monitor", "D00c_mom_vecinv", "I02_ini_fields"}
PTRACERS = {"tutorial_global_oce_latlon", "tutorial_advection_in_gyre", "tutorial_tracer_adjsens"}


def _stages(exp, code, inp=None):
    jd = JD if inp is None else rr.BUILDS[rr.M2_BUILD.get((exp, inp), (None, "jaxdump_m2"))[1]][0]
    f = paths.REFERENCE / "bin" / f"{exp}-{code}-{UP7}-{jd}" / "jaxdump_stages.txt"
    return {ln.split()[0] for ln in f.read_text().split("\n") if ln.strip()}


def _jdon(exp, inp):
    return DumpSet(rr.run_top(rr.find_run(exp, inp, "jdon")) / "dumps")


def test_m2_compiled_stages():
    """Core stages in every M2 jaxdump build; T04_ptracers_integrate exactly in the ptracers builds; package stages
    exactly where the package is compiled (GGL90: cs32x15; SBO, cost, ctrl: the code_ad builds that have them)."""
    builds = {(e, c): i for e, i, c in rr.M2_VARIANTS}       # one variant per build names its jaxdump binary
    for (exp, code), inp in sorted(builds.items()):
        st = _stages(exp, code, inp)
        assert CORE <= st, (exp, sorted(CORE - st))
        assert ("T04_ptracers_integrate" in st) == (exp in PTRACERS), exp
        assert ("P04_ggl90" in st) == ((exp, code) == ("global_ocean.cs32x15", "code")), exp
        assert ("S19_sbo_calc" in st) == (exp == "global_ocean.90x40x15"), exp
        assert ("S18_cost_tile" in st) == (code == "code_ad"), exp


def test_m2_invisibility_verdicts():
    """Dumps off and on change no output file and no STDOUT number of any M2 variant (named exemptions only)."""
    for exp, inp, _ in rr.M2_VARIANTS:
        f = rr.invisibility_file(exp, inp)
        assert f.is_file(), f"missing {f}"
        lines = f.read_text().split("\n")
        assert not any(ln.startswith("VISIBLE") for ln in lines), (exp, inp, lines[:3])
        for mode in ("dumps-off", "dumps-on"):
            assert sum(ln.startswith(f"INVISIBLE {exp}/{inp} {mode}") for ln in lines) == 1, (exp, inp, mode)


# ---- exchange probes on the cube layouts (exch2, 6 faces of 32x32) ---------------------------------------------------
# Frozen summaries (reference/probe_map.py on the jdon runs; the JSON files in reference/probe_maps/ are the measured
# values, also written with the npz maps for lane B to $MJX_REFERENCE/probe_maps/).
FROZEN = REPO / "reference" / "probe_maps"
SCALAR = ("xT", "x3D", "xZ", "xSMs", "xS3D")
NOSIGN = ("xUVn_u", "xUVn_v", "xAn_u", "xAn_v", "xBn_u", "xBn_v")


def _frozen(exp, inp):
    run = rr.find_run(exp, inp, "jdon")
    return run, json.loads((FROZEN / f"{exp}-{inp}-{run['run_id']}.json").read_text())


@pytest.mark.parametrize("exp,inp", CUBES)
def test_cube_probe_summary_frozen(exp, inp):
    """The probe summary of every cube layout equals the frozen measurement (every value decodes as a probe code)."""
    run, want = _frozen(exp, inp)
    got = pm.summary(rr.run_top(run) / "dumps")
    assert got["fields"] == want["fields"], exp
    assert {k: list(v) for k, v in got["zeros"].items()} == want["zeros"], exp
    assert got["ntiles"] == want["ntiles"] and got["faces"] == {str(f): [32, 32] for f in range(1, 7)}


@pytest.mark.parametrize("exp,inp", CUBES)
def test_cube_probe_properties(exp, inp):
    """What the frozen numbers say about exch2 on the cube (lane B's cube maps must reproduce it):
    * no exchange writes an interior point; every scalar exchange (EXCH_XY_RL, EXCH_3D_RL, EXCH_SM_3D_RL) fills
      every halo point, corner blocks included; scalars never swap components;
    * every vector exchange crosses faces with component swaps (u halo from v of the rotated neighbour face);
      withSigns=.FALSE. never flips a sign, withSigns=.TRUE. flips some; EXCH_SM_3D_RL (signed scalar) flips signs
      without a swap;
    * the C-grid vector exchanges (EXCH_UV_XY_RL, EXCH_UV_3D_RL) leave corner-block points unwritten (6 per
      layout with 2 halo rows, 66 with OLx=4), the B-grid ones and EXCH_Z_3D_RL 6 edge-strip points;
    * only the D-grid exchange (EXCH_UV_DGRID_3D_RL) copies from HALO points of the source tile (a copy of a copy);
    * mixed signed zeros (zu*: u=-0, v=+0): the interior keeps -0 (6144 = 6 faces x 32 x 32 points of u), the
      C-grid exchanges leave -0 only on a few halo points (sa1*u + sa2*v arithmetic turns -0 into +0), the A/B-grid
      ones keep it on most (copies)."""
    _, s = _frozen(exp, inp)
    f = s["fields"]
    for fld, v in f.items():
        assert v["interior_written"] == 0, (exp, fld)
        assert v["cross_face"] > 0, (exp, fld)
        if fld in SCALAR:
            assert v["comp_swap"] == 0, (exp, fld)
        else:
            assert v["comp_swap"] > 0, (exp, fld)
        if fld in NOSIGN:
            assert v["sign_flip"] == 0, (exp, fld)
        elif fld not in ("xT", "x3D", "xZ", "xS3D"):
            assert v["sign_flip"] > 0, (exp, fld)
        assert (v["halo_source"] > 0) == fld.startswith("xDs_"), (exp, fld)
    for fld in ("xT", "x3D", "xSMs"):
        assert f[fld]["written"] == f[fld]["halo"], (exp, fld)
    n_c = f["xUVs_u"]["unwritten_corner"]
    assert n_c in (6, 66) and all(f[g]["unwritten_corner"] == n_c for g in
                                  ("xUVs_u", "xUVs_v", "xUVn_u", "xUVn_v", "xUV3s_u", "xUV3s_v")), exp
    assert all(f[g]["unwritten_edge"] == 6 and f[g]["unwritten_corner"] == 0 for g in
               ("xZ", "xBs_u", "xBs_v", "xBn_u", "xBn_v")), exp
    z = s["zeros"]
    assert all(z[k][0] == 6144 for k in z if k.startswith(("zm", "zu")) and k.endswith("_u")), exp
    assert z["zuUVs_u"][1] < z["zuAs_u"][1] and z["zuUV3s_u"][1] < z["zuBs_u"][1], exp


def test_cube_probe_negative_controls():
    """Planted changes of one decoded value change the summary: a sign flip, a component swap, a copy from a halo
    point, a point left unwritten, a copy from the own face (no longer cross-face)."""
    exp, inp = CUBES[0]
    ds = _jdon(exp, inp)
    it = ds.iterations()[0]
    base = pm.summarize_field(ds, it, "xUVs_u")

    def planted(change):
        dec = pm.decode_field(ds, it, "xUVs_u")
        t = sorted(dec)[0]
        w = np.argwhere(dec[t]["written"] == 1)
        j, i = (int(x) for x in w[0])
        change(dec[t], j, i, t)
        return pm.summarize_field(ds, it, "xUVs_u", dec=dec)

    def flip(d, j, i, t):
        d["sign"][j, i] *= -1

    def swap(d, j, i, t):
        d["comp"][j, i] = 3 - d["comp"][j, i]

    def halo_src(d, j, i, t):
        d["src_ja"][j, i], d["src_ia"][j, i] = 0, 0

    def unwrite(d, j, i, t):
        d["written"][j, i] = 0

    s = planted(flip)
    assert s["sign_flip"] == base["sign_flip"] + 1 or s["sign_flip"] == base["sign_flip"] - 1
    s = planted(swap)
    assert abs(s["comp_swap"] - base["comp_swap"]) == 1
    s = planted(halo_src)
    assert s["halo_source"] == base["halo_source"] + 1
    s = planted(unwrite)
    assert s["written"] == base["written"] - 1
    with pytest.raises(ValueError, match="not exchange-probe codes"):
        pm.probe_decode(np.array([1.5]), 1e6, 10, 10)


# ---- ptracers group P and the vector-invariant momentum stage ------------------------------------------------------
def test_ptracer_records_and_vecinv_stage():
    """Group P is dumped in the ptracers builds only: pTracer_01 at S00_begin of step n+1 equals S16 of step n
    bitwise (nothing touches it between the end of a step and the start of the next) and T04 holds the advanced
    tracer; D00c_mom_vecinv records exist exactly in the vector-invariant runs (VECINV lane)."""
    for exp, inp in (("tutorial_global_oce_latlon", "input"), ("tutorial_advection_in_gyre", "input"),
                     ("tutorial_tracer_adjsens", "input_ad")):
        ds = _jdon(exp, inp)
        it = ds.iterations()
        a = ds.field(it[0], "S16_blocking_exchanges", "pTracer_01")
        b = ds.field(it[1], "S00_begin", "pTracer_01")
        assert np.array_equal(a.view(np.int64), b.view(np.int64)), exp
        assert ("T04_ptracers_integrate" in ds.stages(it[0])), exp
    ds = _jdon("global_ocean.cs32x15", "input")
    assert not any(k[2].startswith("pTracer") for k in ds.keys(ds.iterations()[0]))
    for exp, inp, vec in (("solid-body.cs-32x32x1", "input", True), ("global_ocean.cs32x15", "input", True),
                          ("global_ocean.90x40x15", "input_ad", True), ("tutorial_global_oce_latlon", "input", False)):
        ds = _jdon(exp, inp)
        assert ("D00c_mom_vecinv" in ds.stages(ds.iterations()[0])) == vec, exp


# ---- coverage worklists of the M2 variants (lane E's tools, reference/coverage/CURRENT_M2) -----------------------------
def _cov_current():
    out = []
    for ln in (REPO / "reference" / "coverage" / "CURRENT_M2").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            doc, rep = ln.split()
            out.append((REPO / "docs" / "coverage" / doc, paths.REFERENCE / "coverage" / rep))
    return out


def test_m2_coverage_worklists():
    """docs/coverage/<experiment>.md of every M2 experiment is its gcov report; every M2 variant is in a report, ran
    the_model_main once (the minimal case input_min: never, its main.F skips it; EEBOOT's w2_eeboot once), printed the
    same %MON/cg2d/cost lines as the plain oracle run, and is physics-active; every report is titled M2."""
    import gzip
    problems, seen = [], set()
    from tools.coverage import neutral_paths    # the page names $MJX_* locations, the report absolute ones
    for doc, md in _cov_current():
        if not doc.is_file() or doc.read_text() != neutral_paths(md.read_text()):
            problems.append(f"{doc} is not the report {md}")
        rep = json.load(gzip.open(str(md)[:-3] + ".json.gz", "rt"))
        problems += [f"{md.name}: mapping problem {p}" for p in rep["problems"]]
        if not md.read_text().split("\n")[0].endswith("— M2 porting worklist"):
            problems.append(f"{md}: title {md.read_text().split(chr(10))[0]!r}")
        for v in rep["variants"]:
            seen.add((rep["meta"]["experiment"], v["input_dir"]))
            if v["compare_plain"]["equal"] is not True:
                problems.append(f"{md}: {v['input_dir']} gcov run vs plain {v['compare_plain']}")
            main = [r for r in v["routines"] if r["routine"] == "the_model_main"]
            if (rep["meta"]["experiment"], v["input_dir"]) in {(e, i) for e, i, _ in rr.M2_MIN}:
                boot = [r for r in v["routines"] if r["routine"] == "w2_eeboot"]
                if any(r["calls"] for r in main) or not boot or boot[0]["calls"] != 1:
                    problems.append(f"{md}: {v['input_dir']} the_model_main executed or w2_eeboot not once")
            elif not main or main[0]["calls"] != 1:
                problems.append(f"{md}: {v['input_dir']} the_model_main not executed once")
            problems += [f"{md}: {v['input_dir']} inactive {c}" for c in v["physics"] if not c["ok"]]
    missing = {(e, i) for e, i, _ in rr.M2_VARIANTS + rr.M2_MIN} - seen
    assert not problems and not missing, "\n".join(problems + [f"not in a report: {m}" for m in sorted(missing)])


def test_diagstats_unset_regions_exemption(tmp_path):
    """DIAGSTATS_UNSET_REGIONS (reference/jaxdump/invisibility.py): the plain and jdoff dynStDiag files of
    global_ocean.cs32x15/input differ only in regions >= 1; a planted change in region 0 is still a difference, one in
    region 1 is exempt; a data.diagnostics with stat_region switches the exemption off."""
    run = rr.find_run("global_ocean.cs32x15", "input", "yardstick")
    rel = "mnc_test_0001/dynStDiag.0000072000.t001.nc"
    a = (rr.run_top(run) / "rundir" / rel).read_bytes()
    b = (rr.run_top(rr.find_run("global_ocean.cs32x15", "input", "jdoff")) / "rundir" / rel).read_bytes()
    assert invis.blank_nc_stamps(a) != invis.blank_nc_stamps(b)
    blank = lambda x: invis.blank_unset_regions(invis.blank_nc_stamps(x))  # noqa: E731
    assert blank(a) == blank(b)
    dims, var, recsize = invis._nc_header(a)
    name, ids, typ, vsize, begin = next(v for v in var if v[0] == "THETA_lv_ave")
    nlev = dims[ids[2]][1]
    for off, exempt in ((begin + 8 * 3, False), (begin + nlev * 8 + 8 * 3, True)):   # region 0 / region 1, level 4
        c = bytearray(a)
        c[off] ^= 0x01
        assert (blank(bytes(c)) == blank(a)) == exempt, (off, exempt)
    d = tmp_path / "rundir"
    d.mkdir()
    assert not invis.stat_region_set(d)
    (d / "data.diagnostics").write_text(" &DIAG_STATIS_PARMS\n stat_region(1,1) = 1,\n &\n")
    assert invis.stat_region_set(d)
    (d / "data.diagnostics").write_text(" &DIAG_STATIS_PARMS\n# stat_region(1,1) = 1,\n &\n")
    assert not invis.stat_region_set(d)


def _milestone_mismatches(cv):
    """Registered variants whose coverage rung (tools/coverage.py RUNG_MILESTONE via rung_key) names another milestone
    than the registry group they belong to (reference_runs.py VARIANTS = M1, M2_VARIANTS = M2, M3_VARIANTS = M3, M4_VARIANTS =
    M4), or
    has none."""
    out = []
    for group, ms in ((rr.VARIANTS, "M1"), (rr.M2_VARIANTS + rr.M2_MIN, "M2"), (rr.M3_VARIANTS, "M3"),
                      (rr.M4_VARIANTS, "M4")):
        for e, i, _ in group:
            got = cv.RUNG_MILESTONE.get(cv.rung_key(e, i))
            if got != ms:
                out.append((e, i, got, ms))
    return out


def test_coverage_report_title_by_milestone(monkeypatch):
    """tools/coverage.py titles a report by the milestone of its variants' rungs (lane A session 7; it said "M1 porting
    worklist" for every report): every rung has a milestone, each registered variant's rung names the milestone of its
    registry group, and the markdown title is report_title(); negative control: a planted wrong milestone is reported
    and changes the title."""
    from tools import coverage as cv
    assert set(cv.RUNG_MILESTONE) == set(cv.RUNG_CHECKS) and set(cv.RUNG_MILESTONE.values()) <= {"M1", "M2", "M3",
                                                                                                    "M4"}
    assert _milestone_mismatches(cv) == []
    assert cv.report_title("tutorial_barotropic_gyre", "code", ["input"]) == \
        "# Coverage: tutorial_barotropic_gyre (code) — M1 porting worklist"
    assert cv.report_title("global_ocean.90x40x15", "code_ad", ["input_ad", "input_ad.kapgm"]).endswith(
        "— M2 porting worklist")
    assert cv.report_title("global_ocean.90x40x15", "code", ["input"]).endswith("— M1 porting worklist")
    v = {"input_dir": "input", "run_top": "/x/job1", "mon_blocks": 0, "exit": 0, "normal_end": 1, "routines": [],
         "physics": [], "compare_plain": {"equal": None}, "collect": "/x/c",
         "switches": {"keys_error": None, "on": [], "off": [], "selectors": []}}
    mapper = type("M", (), {"where": {}, "problems": []})()
    md = cv.markdown("advect_cs", "code", "b", [v], mapper, {"commit": "c"})
    assert md.split("\n")[0] == "# Coverage: advect_cs (code) — M2 porting worklist"
    monkeypatch.setitem(cv.RUNG_MILESTONE, "advect_cs", "M1")              # planted
    assert _milestone_mismatches(cv) == [("advect_cs", "input", "M1", "M2")]
    assert cv.markdown("advect_cs", "code", "b", [v], mapper, {"commit": "c"}).split("\n")[0].endswith(
        "— M1 porting worklist")


def test_adm_monitor_parser_and_taf_reference():
    """reference_runs.adm_monitor_blocks/summary (the TAF adjoint monitor table of docs/YARDSTICK.md): fabricated
    blocks parse into tsnumbers, groups and all-zero groups; a planted extra field in one block is reported as
    differing field sets; on global_ocean.cs32x15's results/output_adm.txt: 6 blocks 72005..72000, one per step,
    adeta zero in every block."""
    def blk(ts, theta, extra=()):
        return [f"(PID.TID 0000.0001) %MON ad_time_tsnumber             =                 {ts}",
                "(PID.TID 0000.0001) %MON ad_time_secondsf             =   1.0000000000000E+00",
                "(PID.TID 0000.0001) %MON ad_dynstat_adeta_max         =   0.0000000000000E+00",
                f"(PID.TID 0000.0001) %MON ad_dynstat_adtheta_max       =   {theta}",
                "(PID.TID 0000.0001) %MON advcfl_uvel_max              =   1.0000000000000E+00", *extra]
    lines = blk(2, "1.0E+00") + ["noise"] + blk(1, "0.0E+00") + blk(0, "-2.0E+00")
    b = rr.adm_monitor_blocks(lines)
    assert [t for t, _ in b] == [2, 1, 0] and len(b[0][1]) == 3
    s = rr.adm_monitor_summary(b)
    assert s["tsnumbers"] == [2, 1, 0] and s["records_per_step"] == [1] and s["same_fields"]
    assert s["fields"] == [("ad_dynstat_adeta", ["max"]), ("ad_dynstat_adtheta", ["max"])]
    assert s["all_zero_groups"] == ["ad_dynstat_adeta"]
    planted = blk(2, "1.0E+00") + blk(1, "0.0E+00", ["(PID.TID 0000.0001) %MON ad_dynstat_adsalt_max = 1.0E+00"])
    assert not rr.adm_monitor_summary(rr.adm_monitor_blocks(planted))["same_fields"]
    assert rr.adm_monitor_summary(rr.adm_monitor_blocks(blk(1, "0.0E+00") + blk(1, "0.0E+00")))["records_per_step"] \
        == [2]
    ref = paths.UPSTREAM / "verification" / "global_ocean.cs32x15" / "results" / "output_adm.txt"
    s = rr.adm_monitor_summary(rr.adm_monitor_blocks(rr.trj.read_lines(ref)))
    assert s["tsnumbers"] == list(range(72005, 71999, -1)) and s["records_per_step"] == [1] and s["same_fields"]
    assert s["all_zero_groups"] == ["ad_dynstat_adeta"] and len(s["fields"]) == 6


def test_cs32x15_input_ad_stop_and_minimal_run(tmp_path):
    """global_ocean.cs32x15/input_ad (session 7): the declared grdchk stop matches the three triple runs and the FD
    runs end normally; a copy of the plain run with one changed run.tr_log line is a FAIL (negative control).
    adjustment.cs-32x32x1/input_min: the registered minimal run ends normally, prints no %MON line, and its W2 log
    differs from the code build's only by the CUMSUM tile matrix."""
    import shutil
    verdict = _load(REPO / "reference" / "run_verdict.py", "_mjx_run_verdict_s7")
    for kind in ("yardstick", "jdoff", "jdon"):
        top = rr.run_top(rr.find_run("global_ocean.cs32x15", "input_ad", kind))
        assert verdict.classify(top, 0)[0] == "EXPECTED-STOP", kind
    for kind in ("fdzero", "fdrandom"):
        assert verdict.classify(rr.run_top(rr.find_run("global_ocean.cs32x15", "input_ad", kind)), 0)[0] == "OK"
    top = rr.run_top(rr.find_run("global_ocean.cs32x15", "input_ad", "yardstick"))
    copy = tmp_path / "run"
    (copy / "rundir").mkdir(parents=True)
    shutil.copy(top / "MANIFEST.json", copy / "MANIFEST.json")
    shutil.copy(top / "rundir" / "output.txt", copy / "rundir" / "output.txt")
    tr = (top / "rundir" / "run.tr_log").read_text(encoding="latin-1")
    (copy / "rundir" / "run.tr_log").write_text(tr.replace("mom_StartAB =         1", "mom_StartAB =         2"),
                                                encoding="latin-1")
    assert verdict.classify(copy, 0)[0] == "FAIL"
    run = rr.find_run("adjustment.cs-32x32x1", "input_min", "minimal")
    rd = rr.run_top(run) / "rundir"
    out = (rd / "output.txt").read_text()
    assert "%MON" not in out and "Skip THE_MODEL_MAIN" in (rd / "run.tr_log").read_text() + out
    assert verdict.classify(rr.run_top(run), 0)[0] == "OK"
    a = (rd / "w2_tile_topology.0000.log").read_text().split("\n")
    b = (rr.run_top(rr.find_run("adjustment.cs-32x32x1", "input", "yardstick")) / "rundir" /
         "w2_tile_topology.0000.log").read_text().split("\n")
    i = next(k for k, (x, y) in enumerate(zip(a, b)) if x != y)
    n = len(b) - len(a) + 1
    assert "skip Tile Matrix setting" in a[i] and "Tile Matrix for CUMUL-SUM" in b[i]
    assert a[:i] + a[i + 1:] == b[:i] + b[i + n:]


def _output_off_runs():
    """[(exp, inp, label)] of every registered output-request overlay run (M1 OUTPUT_OFF, M2 OUTPUT_OFF_M2)."""
    out = [(e, i, lab) for (e, i), labs in rr.OUTPUT_OFF.items() for lab in labs]
    return out + [(e, i, lab) for (e, i), (labs, _) in rr.OUTPUT_OFF_M2.items() for lab in labs]


def test_output_request_runs_change_no_model_number(tmp_path):
    """Diagnostics and mnc are not ported (brainstorm section 3); the gate is that the Fortran's model numbers do not
    depend on them: for every registered diagoff/mncoff run (M1 job 27827363; M2 lane A session 8) the STDOUT model
    lines (%MON incl. trcstat, %SBO, cg2d, cost fc and objf terms, ADM) equal the plain run's line for line, and every
    MDS pickup both runs wrote is byte-identical. Negative controls: one changed %MON value in a copy of an overlay
    run's STDOUT, and one flipped byte in a copied pickup, are each reported."""
    import shutil
    problems, seen_pickups = [], 0
    for e, i, lab in _output_off_runs():
        base = rr.run_top(rr.find_run(e, i, "yardstick"))
        probs, n, common = rr.output_request_problems(base, rr.run_top(rr.find_run(e, i, lab)))
        problems += [f"{e}/{i} {lab}: {p}" for p in probs]
        if n < 10:
            problems.append(f"{e}/{i} {lab}: only {n} model lines")
        seen_pickups += len(common)
    assert not problems, "\n".join(problems)
    # negative controls on a copy (made in tmp_path; nothing is removed)
    e, i, lab = next((e, i, lab) for e, i, lab in _output_off_runs() if rr.mds_pickups(rr.run_top(rr.find_run(e, i, lab))))
    base, test = rr.run_top(rr.find_run(e, i, "yardstick")), rr.run_top(rr.find_run(e, i, lab))
    copy = tmp_path / "copy"
    (copy / "rundir").mkdir(parents=True)
    lines = (test / "rundir" / "output.txt").read_text(errors="replace").split("\n")
    k = next(n for n, ln in enumerate(lines) if "%MON dynstat_theta_mean" in ln or "%MON dynstat_eta_mean" in ln)
    lines[k] = lines[k][:-2] + ("1" if lines[k][-2] != "1" else "2") + lines[k][-1]
    (copy / "rundir" / "output.txt").write_text("\n".join(lines))
    for n, f in rr.mds_pickups(test).items():
        shutil.copyfile(f, copy / "rundir" / n)
    probs, _, _ = rr.output_request_problems(base, copy)
    assert len(probs) == 1 and probs[0].startswith("model lines differ"), probs
    shutil.copyfile(test / "rundir" / "output.txt", copy / "rundir" / "output.txt")
    assert rr.output_request_problems(base, copy)[0] == []
    name = sorted(n for n in rr.mds_pickups(test) if n.endswith(".data"))[0]
    raw = bytearray((copy / "rundir" / name).read_bytes())
    raw[len(raw) // 2] ^= 0x01
    (copy / "rundir" / name).write_bytes(bytes(raw))
    assert rr.output_request_problems(base, copy)[0] == [f"pickup {name} differs"]


# MAX/MIN site tables of the M2 oracle builds (lane A session 8, reference/jobs/minmax_sites.sbatch, job 27833815)
M1_MINMAX = ("advect_xy-code-63cdc0b-f48c062", "advect_xz-code-63cdc0b-f48c062",
             "global_ocean.90x40x15-code-63cdc0b-f48c062", "tutorial_baroclinic_gyre-code-63cdc0b-f48c062",
             "tutorial_barotropic_gyre-code-63cdc0b-f48c062", "tutorial_global_oce_optim-code_ad-63cdc0b-f48c062")
M2_MINMAX = tuple(f"{e}-{c}-{UP7}-06b4417" for e, c in sorted({(e, c) for e, i, c in rr.M2_VARIANTS
                                                                if (e, i) not in rr.M2_BUILD})) + \
    ("global_ocean.cs32x15-code_ad-63cdc0b-78ca390", "adjustment.cs-32x32x1-code_min-63cdc0b-78ca390")


def test_m2_minmax_tables(tmp_path):
    """Every M2 oracle build (plain and code_ad, code_min) has its MAX/MIN site table (tools/fortran_minmax_sites.py)
    with the recorded sha256, objects byte-identical to the build's, every step resolved at the machine level; over
    the M1 and M2 tables no site has a different winner in two builds (0 conflicts). Negative control: a copy of one
    M2 table with one site's p flipped conflicts with the original."""
    fms = _load(REPO / "tools" / "fortran_minmax_sites.py", "_mjx_fms_m2")
    root = paths.REFERENCE / "minmax_sites"
    assert len(M2_MINMAX) == 10
    for b in M2_MINMAX:
        text = (root / f"{b}.json").read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == (root / f"{b}.json.sha256").read_text().split()[0], b
        t = json.loads(text)
        m = t["meta"]
        assert m["objects_differ"] == [] and m["objects_identical"] == m["n_files"] > 100, b
        assert m["problems"] == [] and m["n_sites"] > 0, b
        assert all(st["winner"] in ("acc", "new") for s in t["sites"] for st in s["steps"]), b
        assert m["compiler_version"] == "GNU Fortran (Spack GCC) 11.2.0" and "-O0" in m["flags"], b
    all_tables = [root / f"{b}.json" for b in M1_MINMAX + M2_MINMAX]
    differ, n, counts = fms.compare(all_tables)
    assert differ == {} and n > 400 and len(counts) == 16
    # PTRACERS lane: CG2D_NSA's rhsMax site in tutorial_tracer_adjsens/code_ad's table
    ta = json.loads((root / f"tutorial_tracer_adjsens-code_ad-{UP7}-06b4417.json").read_text())
    assert any(s["F_file"].endswith("cg2d_nsa.F") for s in ta["sites"])
    # negative control: one flipped winner in a copy (written to tmp_path; nothing is removed)
    t = json.loads((root / f"{M2_MINMAX[0]}.json").read_text())
    s = next(s for s in t["sites"] if s["nargs"] == 2)
    s["p"] = "b" if s["p"] == "a" else "a"
    t["meta"]["build"] = "planted"
    (tmp_path / "planted.json").write_text(json.dumps(t))
    differ, _, _ = fms.compare([root / f"{M2_MINMAX[0]}.json", tmp_path / "planted.json"])
    assert len(differ) == 1
