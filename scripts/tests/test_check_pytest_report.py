"""The tier runner's pass/fail judge: it must pass a clean report and fail every broken one."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_pytest_report", Path(__file__).resolve().parents[1] / "check_pytest_report.py")
cpr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpr)


def _report(tmp_path, cases, **totals):
    body = "".join(
        f'<testcase classname="m" name="t{i}">{inner}</testcase>' for i, inner in enumerate(cases))
    attrs = " ".join(f'{k}="{v}"' for k, v in totals.items())
    p = tmp_path / "r.xml"
    p.write_text(f'<?xml version="1.0"?><testsuites><testsuite name="pytest" {attrs}>{body}'
                 "</testsuite></testsuites>")
    return str(p)


def test_clean_report_passes(tmp_path):
    r = _report(tmp_path, ["", "", '<skipped message="no fixture"/>'],
                tests=3, failures=0, errors=0, skipped=1)
    assert cpr.main([r]) == 0


def test_negative_controls(tmp_path):
    fail = _report(tmp_path, ['<failure message="x"/>'], tests=1, failures=1, errors=0, skipped=0)
    assert cpr.main([fail]) == 1
    err = _report(tmp_path, ['<error message="x"/>'], tests=1, failures=0, errors=1, skipped=0)
    assert cpr.main([err]) == 1
    empty = _report(tmp_path, [], tests=0, failures=0, errors=0, skipped=0)
    assert cpr.main([empty]) == 1
    # totals that under-count a failing testcase
    lying = _report(tmp_path, ['<failure message="x"/>'], tests=1, failures=0, errors=0, skipped=0)
    assert cpr.main([lying]) == 1
    over = _report(tmp_path, ["", "", ""], tests=3, failures=0, errors=0, skipped=0)
    assert cpr.main([over, "--max-tests", "2"]) == 1
    assert cpr.main([str(tmp_path / "absent.xml")]) == 1
    garbage = tmp_path / "g.xml"
    garbage.write_text("<testsuites><testsuite")
    assert cpr.main([str(garbage)]) == 1
