"""scripts/tier1x_shards.py: the parallel tier-1x shards cover every tier1x file of the manifest exactly once, the
split is deterministic, and measured times balance it (longest-first greedy)."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tier1x_shards", REPO / "scripts/tier1x_shards.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)


def test_every_tier1x_file_in_exactly_one_shard():
    files = ts.tier_files("tier1x")
    assert files
    for n in (1, 3, 4):
        shards, _ = ts.split(files, {}, n)
        flat = [f for s in shards for f in s]
        assert sorted(flat) == files and len(flat) == len(set(flat))
        assert ts.split(files, {}, n) == (shards, _)                         # deterministic
    # measured times balance the split (longest first, greedy)
    files = ["a.py", "b.py", "c.py", "d.py"]
    shards, load = ts.split(files, {"a.py": 100.0, "b.py": 60.0, "c.py": 50.0, "d.py": 10.0}, 2)
    assert shards == [["a.py", "d.py"], ["b.py", "c.py"]] and load == [110.0, 110.0]
    # an unmeasured file weighs the largest measured time: scheduled first and alone (the median rule would have put
    # b.py next to it: [["a.py"], ["new.py", "b.py"]], the negative control of this assertion)
    shards, load = ts.split(["a.py", "b.py", "new.py"], {"a.py": 100.0, "b.py": 60.0}, 2)
    assert shards == [["a.py", "b.py"], ["new.py"]] and load == [160.0, 100.0]
    # negative control: a dropped file or a duplicated one is caught by the coverage check above
    bad = [["a.py"], ["b.py", "c.py", "c.py"]]
    flat = [f for s in bad for f in s]
    assert sorted(flat) != files or len(flat) != len(set(flat))
    # JUnit classname -> test file
    assert ts.file_of("scripts.tests.test_tier1x_shards") == "scripts/tests/test_tier1x_shards.py"
    assert ts.file_of("mitjax.tests.test_manifest") == "mitjax/tests/test_manifest.py"
    assert ts.file_of("no.such.module") is None


def _report(path, cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f'<testcase classname="{c}" name="t{i}" time="{t}"/>' for i, (c, t) in enumerate(cases))
    path.write_text(f"<testsuite>{body}</testsuite>")


def test_measured_times_take_each_file_from_its_newest_run(tmp_path):
    """A run cut off by its time limit still measures the files it finished (tier 1x 27951742)."""
    a, b, c = "mitjax.tests.test_manifest", "scripts.tests.test_tier1x_shards", "mitjax.tests.test_scan_k"
    # older complete run: a, b; newer run that timed out: a (re-measured) and c (new), b never finished
    _report(tmp_path / "27941838" / "report_0_1.xml", [(a, 10.0), (a, 5.0)])
    _report(tmp_path / "27941838" / "report_1_1.xml", [(b, 7.0)])
    _report(tmp_path / "27951742" / "report_0_1.xml", [(a, 20.0)])
    _report(tmp_path / "27951742" / "report_1_1.xml", [(c, 3.0)])
    (tmp_path / "27951742" / "report_2_1.xml").write_text("<testsuite")           # killed mid-write: ignored
    _report(tmp_path / "9999999" / "report.xml", [(c, 99.0)])                    # older job id, fewer digits
    times = ts.measured_times(tmp_path)
    # negative control: the earlier rule (one report set, the one covering the most files) loses b (checked)
    assert times == {"mitjax/tests/test_manifest.py": 20.0, "scripts/tests/test_tier1x_shards.py": 7.0,
                     "mitjax/tests/test_scan_k.py": 3.0}
    assert ts.measured_times(tmp_path / "missing") == {}
