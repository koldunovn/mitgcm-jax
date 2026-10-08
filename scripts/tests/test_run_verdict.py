"""reference/run_verdict.py (REVIEW_M0 #12): the forward-only optim run's grdchk stop is recognised by its exact
signature and reported as RUN EXPECTED-STOP (exit 3, which the job scripts accept), so it no longer fails every
invisibility job; any other abnormal end stays RUN FAIL. Reads one real run directory (read-only) and synthetic
copies of it; fails, not skips, when the run is missing."""

import importlib.util
import json
import shutil
from pathlib import Path

from mitjax import paths

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("_mjx_run_verdict", REPO / "reference" / "run_verdict.py")
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)

OPTIM = paths.REFERENCE_RUNS / "tutorial_global_oce_optim" / "input_ad" / "job27826998-plain"   # exit 0, grdchk stop


def _copy(src, dst, experiment=None, tr_log=None, stdout=None):
    (dst / "rundir").mkdir(parents=True)
    man = json.loads((src / "MANIFEST.json").read_text())
    if experiment:
        man["experiment"] = experiment
    (dst / "MANIFEST.json").write_text(json.dumps(man))
    (dst / "rundir" / "output.txt").write_text(stdout if stdout is not None else
                                               (src / "rundir" / "output.txt").read_text(encoding="latin-1"))
    if tr_log is None:
        shutil.copyfile(src / "rundir" / "run.tr_log", dst / "rundir" / "run.tr_log")
    else:
        (dst / "rundir" / "run.tr_log").write_text(tr_log)
    return dst


def test_expected_stop_recognised_and_real_failures_not(tmp_path, capsys):
    assert (OPTIM / "rundir" / "output.txt").is_file(), f"missing oracle run {OPTIM}"
    # the real forward-only optim run: the old run.sh verdict (exit 0 but no normal end) was RUN FAIL, exit 1
    old_fail = "Execution ended Normally" not in "".join(
        (OPTIM / "rundir" / "output.txt").read_text(encoding="latin-1").split("\n")[-10:])
    assert old_fail
    assert rv.classify(OPTIM, 0)[0] == "EXPECTED-STOP"
    assert rv.main([str(OPTIM), "0"]) == 3 and "RUN EXPECTED-STOP" in capsys.readouterr().out
    # planted real failures: another STOP, the same stop with a non-zero exit, the same stop in another experiment,
    # a STDOUT that ends elsewhere
    other = _copy(OPTIM, tmp_path / "other", tr_log="STOP ABNORMAL END: S/R INI_PARMS\n")
    assert rv.classify(other, 0)[0] == "FAIL" and rv.main([str(other), "0"]) == 1
    assert rv.classify(OPTIM, 2)[0] == "FAIL"
    elsewhere = _copy(OPTIM, tmp_path / "baro", experiment="tutorial_barotropic_gyre")
    assert rv.classify(elsewhere, 0)[0] == "FAIL"
    text = (OPTIM / "rundir" / "output.txt").read_text(encoding="latin-1")
    cut = _copy(OPTIM, tmp_path / "cut", stdout=text[:text.index("// CONFIG_CHECK : Normal End")])
    assert rv.classify(cut, 0)[0] == "FAIL"
    # a normal end is RUN OK
    ok = _copy(OPTIM, tmp_path / "ok", stdout=text + "PROGRAM MAIN: Execution ended Normally\n", tr_log="")
    assert rv.classify(ok, 0)[0] == "OK" and rv.main([str(ok), "0"]) == 0
    # the job scripts accept exit 3 and only 3 besides 0
    for job in ("jaxdump_runs.sbatch", "oracle_runs.sbatch", "run.sbatch"):
        t = (REPO / "reference" / "jobs" / job).read_text()
        assert '-eq 3' in t and 'run.sh" "$TOP" ||' not in t and '"$RUNSH" "$TOP" ||' not in t, job
