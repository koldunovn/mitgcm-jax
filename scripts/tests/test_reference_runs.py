"""Plan Task 3b: the registered oracle runs, the yardstick and the FD oracle (reference/reference_runs.py).

Oracle-dependent: every test that reads a run fails (never skips) when its data is missing. Negative controls build
planted copies in tmp_path (a missing file, a changed byte, a wrong binary, a yardstick table without one variable)
and require the checks to report them.
"""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load("_t3b_reference_runs", "reference/reference_runs.py")
rdocs = _load("_t3b_reference_docs", "reference/reference_docs.py")
adxx = _load("_t3b_grdchk_adxx", "reference/grdchk_adxx.py")


def variant_rows(text, exp, inp):
    """{(section, variable)} of the yardstick table of one variant in docs/YARDSTICK.md."""
    m = re.search(rf"^### {re.escape(exp)}/{re.escape(inp)}\n(.*?)(?=^### |^## |\Z)", text, re.S | re.M)
    if not m:
        return set()
    rows = re.findall(r"^\| ([^|]+?) \| (\w+) \| `", m.group(1), re.M)
    return {(s, v) for s, v in rows}


def missing_checklist_rows(text):
    """[(variant, variable)] of testreport's check list without a yardstick row (forward or adm section)."""
    out = []
    for exp, inp, _ in rr.VARIANTS + rr.M2_VARIANTS + rr.M3_VARIANTS:
        have = {v for s, v in variant_rows(text, exp, inp) if not s.startswith("forward list")}
        out += [(f"{exp}/{inp}", n) for n in rr.checklist_names(exp, inp) if n not in have]
    return out


def test_yardstick_covers_every_checklist_variable():
    text = (REPO / "docs" / "YARDSTICK.md").read_text()
    assert missing_checklist_rows(text) == []
    # negative control: the table of one variant without its deciding variable
    planted = re.sub(r"^\| forward vs results/output\.ab3_c4\.txt \| Tsd \|.*\n", "", text, flags=re.M)
    assert planted != text
    assert missing_checklist_rows(planted) == [("advect_xy/input.ab3_c4", "Tsd")]
    # M2: the same for a cube variant (only inside its own section: advect_xz/input.nlfs has the same row)
    head = "### adjustment.cs-32x32x1/input.nlfs\n"
    a, b = text.split(head, 1)
    planted = a + head + re.sub(r"^\| forward vs results/output\.nlfs\.txt \| Vav \|.*\n", "", b, count=1,
                                flags=re.M)
    assert planted != text and missing_checklist_rows(planted) == [("adjustment.cs-32x32x1/input.nlfs", "Vav")]


def test_docs_are_the_measured_values():
    """docs/YARDSTICK.md and docs/REFERENCE_RUNS.md equal a fresh rendering from the runs (oracle-dependent)."""
    for name, text in rdocs.render().items():
        assert (REPO / "docs" / name).read_text() == text, f"docs/{name} is not the rendering of the runs now"


def test_registered_runs_pass_checks():
    lock = rr.read_lock()
    assert set(lock) == {rr.run_key(r) for r in rr.RUNS}
    problems = {rr.run_key(r): p for r in rr.RUNS if (p := rr.check_run(r, lock))}
    assert problems == {}
    # every M1 variant has a yardstick, jaxdump (off, on), repeat, debug and retile run, and the invisibility triple
    # of the jaxdump2 build; the variants of rr.JD3 that of the jaxdump3 build with their registered steps
    for exp, inp, _ in rr.VARIANTS:
        kinds = {r["kind"] for r in rr.RUNS if (r["exp"], r["input"]) == (exp, inp)}
        assert {"yardstick", "jdoff", "jdon", "repeat", "debug", "retile", "jdplain2", "jdoff2", "jdon2"} <= kinds, \
            (exp, inp)
        assert ({"jdplain3", "jdoff3", "jdon3"} <= kinds) == ((exp, inp) in rr.JD3), (exp, inp)
        # one run per dumps-on kind (the gates address the oracle by kind: grid_gate.oracle needs exactly one jdon)
        for k in rr.DUMP_KINDS:
            assert sum(r["kind"] == k for r in rr.RUNS if (r["exp"], r["input"]) == (exp, inp)) <= 1, (exp, inp, k)
    assert {rr.find_run(e, i, "jdon3")["steps"] for e, i in rr.JD3} == {(0, 1, 2, 9), (0, 1, 2, 15),
                                                                         (36000, 36001, 36002)}
    # the CTRL lane's runs: zero control and the planted fixture (overlays + copied xx_qnet), never a yardstick
    assert rr.find_run(*rr.OPTIM, "ctrlzero")["steps"] == rr.find_run(*rr.OPTIM, "ctrlxx")["steps"] == tuple(range(10))
    assert rr.find_run(*rr.OPTIM, "ctrlxx")["purpose"].startswith("GATE FIXTURE, NEVER A YARDSTICK")
    assert set(rr.FIXTURE_KINDS) <= set(rr.OVERLAYS) and not set(rr.FIXTURE_KINDS) & {"yardstick", "jdon", "repeat"}
    # M2 (lane A session 6): every variant has its invisibility triple (yardstick = the plain run, jdoff, jdon with
    # the default steps nIter0..nIter0+2); the forward-only code_ad variants their zero and random adxx FD runs
    for exp, inp, code in rr.M2_VARIANTS:
        runs = [r for r in rr.RUNS if (r["exp"], r["input"]) == (exp, inp)]
        kinds = sorted(r["kind"] for r in runs)
        want = ["jdoff", "jdon", "yardstick"] + (["fdrandom", "fdzero"] if code == "code_ad" else [])
        want += list(rr.OUTPUT_OFF_M2.get((exp, inp), ((), None))[0])          # session 8: output-request runs
        want += ["ctrlxx"] if (exp, inp) in rr.M2_CTRLXX else []                # session 8: planted controls
        assert kinds == sorted(want), (exp, inp, kinds)
        n0 = rr.NITER0.get((exp, inp), 0)
        assert rr.find_run(exp, inp, "jdon")["steps"] == (n0, n0 + 1, n0 + 2)
        if (exp, inp) in rr.M2_CTRLXX:
            assert rr.find_run(exp, inp, "ctrlxx")["steps"] == rr.CTRLXX_STEPS
            assert rr.find_run(exp, inp, "ctrlxx")["purpose"].startswith("GATE FIXTURE, NEVER A YARDSTICK")
        assert all(r["build"] in rr.M2_BUILD.get((exp, inp), ("plain_m2", "jaxdump_m2")) for r in runs)
        assert all(r["ends_normally"] == (code == "code" or r["kind"].startswith("fd")) for r in runs), (exp, inp)
    # M3 (lane A session 9): every variant has its invisibility triple, all ending normally
    for exp, inp, code in rr.M3_VARIANTS:
        runs = [r for r in rr.RUNS if (r["exp"], r["input"]) == (exp, inp)]
        assert sorted(r["kind"] for r in runs) == ["jdoff", "jdon", "yardstick"], (exp, inp)
        n0 = rr.NITER0.get((exp, inp), 0)
        assert rr.find_run(exp, inp, "jdon")["steps"] == (n0, n0 + 1, n0 + 2)
        assert all(r["build"] in rr.M3_BUILD[(exp, inp)] and r["ends_normally"] for r in runs), (exp, inp)
    # every reference file has its sha256 recorded, the dumps-on runs their dump listing and steps
    for r in rr.RUNS:
        rec = lock[rr.run_key(r)]
        assert set(rec["sha256"]) == set(rr.reference_files(r)) and all(len(h) == 64 for h in rec["sha256"].values())
        assert (rec["dumps"] is not None) == (r["kind"] in rr.DUMP_KINDS)
        assert (r["steps"] is not None) == (r["kind"] in rr.DUMP_KINDS)
        assert rec["dumps"] is None or len(rec["dumps"]) > 0


def _copy_run(run, root):
    """A planted copy of the small files of a run under root/<exp>/<input>/<run id> (STDOUT copied, not linked)."""
    src = rr.run_top(run)
    dst = Path(root) / run["exp"] / run["input"] / run["run_id"]
    (dst / "rundir").mkdir(parents=True)
    for f in ("READY", "MANIFEST.json", "run_provenance.txt"):
        shutil.copyfile(src / f, dst / f)
    shutil.copyfile(src / "rundir" / "output.txt", dst / "rundir" / "output.txt")
    return dst


def test_check_run_negative_controls(tmp_path):
    lock = rr.read_lock()
    run = rr.find_run("tutorial_barotropic_gyre", "input", "yardstick")
    # positive: an exact copy passes
    good = _copy_run(run, tmp_path / "a")
    assert rr.check_run(run, lock, runs_root=tmp_path / "a") == []
    # a missing run directory
    gone = dict(run, run_id="job0-missing")
    assert any(p.startswith("missing") for p in rr.check_run(gone, lock))
    # a changed byte in the STDOUT
    bad = _copy_run(run, tmp_path / "b")
    out = bad / "rundir" / "output.txt"
    data = bytearray(out.read_bytes())
    i = data.index(b"cg2d_init_res") + 40
    data[i] = ord("9") if data[i] != ord("9") else ord("8")
    out.write_bytes(bytes(data))
    assert any("sha256" in p for p in rr.check_run(run, lock, runs_root=tmp_path / "b"))
    # the STDOUT missing
    miss = _copy_run(run, tmp_path / "c")
    (miss / "rundir" / "output.txt").rename(miss / "rundir" / "output.txt.planted")
    assert any(p.startswith("missing") for p in rr.check_run(run, lock, runs_root=tmp_path / "c"))
    # a MANIFEST naming another binary (the jaxdump build)
    other = _copy_run(run, tmp_path / "d")
    man = json.loads((other / "MANIFEST.json").read_text())
    man["binary"]["path"] = man["binary"]["path"].replace("-f48c062/", "-5f16129-jaxdump/")
    (other / "MANIFEST.json").write_text(json.dumps(man))
    assert any("is not" in p for p in rr.check_run(run, lock, runs_root=tmp_path / "d"))
    # a run that did not end normally although it should
    stop = _copy_run(run, tmp_path / "e")
    prov = stop / "run_provenance.txt"
    prov.write_text(prov.read_text().replace("normal_end 1", "normal_end 0"))
    assert "run_provenance.txt: no normal end" in rr.check_run(run, lock, runs_root=tmp_path / "e")
    assert good.is_dir()


def test_check_run_dump_steps_negative_controls(tmp_path):
    """The dumped iterations (dumps/jaxdump_info.txt) must be the registered JAXDUMP_STEPS: a planted info file with a
    step missing, one with a step changed, and a missing info file are each reported."""
    run = rr.find_run("advect_xy", "input", "jdon3")
    src = rr.run_top(run) / "dumps" / "jaxdump_info.txt"
    assert run["steps"] == (0, 1, 2, 15) and rr.check_run(run) == []

    def planted(name, text):
        top = _copy_run(run, tmp_path / name)
        (top / "dumps").mkdir()
        if text is not None:
            (top / "dumps" / "jaxdump_info.txt").write_text(text)
        return rr.check_run(run, runs_root=tmp_path / name)

    good = src.read_text()
    assert planted("same", good) == []
    assert "step           15" in good
    for name, text in (("short", good.replace("step           15\n", "")),
                       ("moved", good.replace("step           15", "step           16")), ("gone", None)):
        assert any("not the registered steps" in p for p in planted(name, text)), name


def test_check_run_overlay_and_copy_negative_controls(tmp_path):
    """A run's namelist overlays and copied inputs must be the registered ones: planted copies of the ctrlxx fixture
    without one overlay, of a yardstick run with an overlay added, and of the fixture with one copied byte changed are
    each reported (the fixture can never pass as a plain run)."""
    fx = rr.find_run(*rr.OPTIM, "ctrlxx")
    assert rr.check_run(fx) == []

    def planted(run, name, edit_manifest=None, edit_copy=False):
        top = _copy_run(run, tmp_path / name)
        info = rr.run_top(run) / "dumps" / "jaxdump_info.txt"
        if info.is_file():
            (top / "dumps").mkdir()
            shutil.copyfile(info, top / "dumps" / "jaxdump_info.txt")
        if edit_manifest:
            man = json.loads((top / "MANIFEST.json").read_text())
            edit_manifest(man)
            (top / "MANIFEST.json").write_text(json.dumps(man))
        for c in json.loads((top / "MANIFEST.json").read_text()).get("copies") or []:
            shutil.copyfile(rr.run_top(run) / "rundir" / c["name"], top / "rundir" / c["name"])
            if edit_copy:
                f = top / "rundir" / c["name"]
                f.write_bytes(b"\x7f" + f.read_bytes()[1:])
                edit_copy = False
        return rr.check_run(run, runs_root=tmp_path / name)

    assert planted(fx, "same") == []
    assert any("overlays" in p for p in planted(fx, "one_overlay", lambda m: m["overlays"].pop()))
    assert any("not the copied bytes" in p for p in planted(fx, "byte", edit_copy=True))
    ys = rr.find_run(*rr.OPTIM, "yardstick")
    fx_ov = json.loads((rr.run_top(fx) / "MANIFEST.json").read_text())["overlays"]
    assert any("overlays" in p for p in planted(ys, "ys_overlay", lambda m: m.__setitem__("overlays", fx_ov)))
    assert any("copies" in p for p in planted(dict(fx, kind="yardstick"), "fx_as_yardstick"))


def _adm(top):
    return [ln.strip() for ln in (top / "rundir" / "output.txt").read_text().split("\n") if " ADM " in ln]


def test_fd_oracle_zero_vs_random():
    """Option (i): the cost and FD lines do not depend on the adxx file; the adjoint-gradient lines do (the file is
    read: the random file bites); the FD values reach the recorded digits against results/output_adm.txt."""
    zero = rr.run_top(rr.find_run(*rr.OPTIM, "fdzero"))
    rnd = rr.run_top(rr.find_run(*rr.OPTIM, "fdrandom"))
    lz, lr = _adm(zero), _adm(rnd)
    assert len(lz) == len(lr) == 9      # 3 grdchk points x (ref_cost_function, adjoint_gradient, finite-diff_grad)
    for a, b in zip(lz, lr):
        if "adjoint_gradient" in a:
            assert a != b and a.endswith("0.00000000000000E+00")
        else:
            assert a == b
    rows = {(s.split()[0], n): d for s, n, d, _, _ in rr.yardstick_rows(*rr.OPTIM)}
    assert rows[("adm", "admCst")] == 16 and rows[("adm", "admFwd")] == 16
    # the zero files are what the forward-only run itself writes for xx_qnet through the same routine
    for f in sorted((zero / "rundir").resolve().glob("adxx_qnet.0000000000.*")):
        twin = f.with_name(f.name[2:])
        assert twin.is_file() and f.read_bytes() == twin.read_bytes(), f.name
    rfiles = sorted((rnd / "rundir").resolve().glob("adxx_qnet.0000000000.*.data"))
    assert len(rfiles) == 4 and all(f.read_bytes() != f.with_name(f.name[2:]).read_bytes() for f in rfiles)


def test_grdchk_adxx_layout_and_refusals(tmp_path):
    src = rr.run_top(rr.find_run(*rr.OPTIM, "yardstick"))
    assert adxx.main([str(src), str(tmp_path / "z")]) == 0
    made = sorted(p.name for p in (tmp_path / "z").glob("adxx_qnet.*"))
    assert made == sorted("ad" + p.name for p in (src / "rundir").resolve().glob("xx_qnet.0000000000.*"))
    for n in made:
        assert (tmp_path / "z" / n).read_bytes() == ((src / "rundir").resolve() / n[2:]).read_bytes()
    with pytest.raises(SystemExit, match="exists"):
        adxx.main([str(src), str(tmp_path / "z")])
    # planted: a source whose control file is not all zero (not the initial file) is refused
    fake = tmp_path / "fake" / "rundir"
    fake.mkdir(parents=True)
    for p in (src / "rundir").resolve().glob("xx_qnet.0000000000.*"):
        shutil.copyfile(p, fake / p.name)
    d = fake / "xx_qnet.0000000000.001.001.data"
    d.write_bytes(b"\x3f\xf0" + d.read_bytes()[2:])
    with pytest.raises(SystemExit, match="not all zero"):
        adxx.main([str(tmp_path / "fake"), str(tmp_path / "z2")])
