#!/usr/bin/env python3
"""Decide pass/fail of a suite run from pytest's JUnit XML report, not from the exit code alone.

A suite "passes" only if the report exists and parses, at least one test ran, none failed or errored,
and the test count is within the tier budget. Skips are listed, never hidden. Exit 0 = pass, 1 = fail.
(Earlier projects accepted runs that were COMPLETED with failed tests, or green because nothing ran.)

    check_pytest_report.py REPORT.xml [REPORT.xml ...] [--max-tests N]

Several reports (the parallel shards of one tier-1x job) are summed into one verdict; a missing or unreadable
report fails the run.
"""

import argparse
import sys
import xml.etree.ElementTree as ET


def summarize(path):
    """Return dict(tests, failures, errors, skipped, skipped_ids) summed over all <testsuite>s."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    out = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "skipped_ids": [], "failed_ids": []}
    for s in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            out[key] += int(s.get(key, 0))
        for case in s.iter("testcase"):
            cid = f"{case.get('classname')}::{case.get('name')}"
            if case.find("skipped") is not None:
                out["skipped_ids"].append((cid, case.find("skipped").get("message", "")))
            if case.find("failure") is not None or case.find("error") is not None:
                out["failed_ids"].append(cid)
    return out


def verdict(summary, max_tests=None):
    """Return a list of reasons the run fails (empty = pass)."""
    reasons = []
    if summary["tests"] == 0:
        reasons.append("no tests ran")
    if summary["failures"]:
        reasons.append(f"{summary['failures']} failed")
    if summary["errors"]:
        reasons.append(f"{summary['errors']} errors")
    if summary["failed_ids"] and not (summary["failures"] or summary["errors"]):
        reasons.append("testcases carry failures that the suite totals do not count")
    if max_tests is not None and summary["tests"] > max_tests:
        reasons.append(f"{summary['tests']} tests exceed the tier budget of {max_tests}")
    return reasons


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", nargs="+")
    ap.add_argument("--max-tests", type=int, default=None)
    args = ap.parse_args(argv)
    s, unreadable = None, []
    for path in args.report:
        try:
            one = summarize(path)
        except (OSError, ET.ParseError) as e:
            unreadable.append(f"{path}: {e}")
            continue
        if s is None:
            s = one
        else:
            for key in ("tests", "failures", "errors", "skipped"):
                s[key] += one[key]
            s["skipped_ids"] += one["skipped_ids"]
            s["failed_ids"] += one["failed_ids"]
    if s is None:
        s = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "skipped_ids": [], "failed_ids": []}
    passed = s["tests"] - s["failures"] - s["errors"] - s["skipped"]
    print(f"tests {s['tests']}: {passed} passed, {s['failures']} failed, {s['errors']} errors, "
          f"{s['skipped']} skipped")
    for cid in s["failed_ids"]:
        print(f"  FAILED  {cid}")
    for cid, msg in s["skipped_ids"]:
        print(f"  SKIPPED {cid}: {msg}")
    for u in unreadable:
        print(f"  UNREADABLE {u}")
    reasons = verdict(s, args.max_tests)
    if unreadable:
        reasons.append(f"{len(unreadable)} report(s) missing or unreadable (a test process died)")
    print("FAIL: " + "; ".join(reasons) if reasons else "PASS")
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
