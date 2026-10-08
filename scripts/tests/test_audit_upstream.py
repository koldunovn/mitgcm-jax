"""scripts/audit_upstream.py (plan Task 7a): the committed audit table is current and complete, and the gate fails when
a changed file is missing from it, when an override misclassifies a hunk, and when the line rules misread a hunk.

Needs the upstream MITgcm clone ($MJX_UPSTREAM, else mitjax/paths.py); git only, no jax. One full audit (~5 s) is
shared by the module.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "audit_upstream.py"
DOC = REPO / "docs" / "AUDIT_C66G_MASTER.md"


def _load():
    spec = importlib.util.spec_from_file_location("_mjx_audit_upstream", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


au = _load()


@pytest.fixture(scope="module")
def full():
    up, audits, problems = au.run_audit()
    return up, audits, problems, DOC.read_text()


def _hunk(audits, hid):
    return next(h for fa in audits for h in fa.hunks if h.hid == hid)


def test_check_passes_on_committed_table(full):
    """The gate itself: generated section byte-identical, nothing undecided, overrides valid, forward hunks noted;
    and the CLI exits 0 (the same check a reviewer runs)."""
    up, audits, problems, doc = full
    assert au.check(doc, audits, problems) == []
    assert sum(len(fa.hunks) for fa in audits) > 900          # the audit really diffed the files
    assert {fa.status for fa in audits} >= {"modified", "added", "removed", "moved"}
    r = subprocess.run([sys.executable, str(SCRIPT), "--check", str(DOC)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("OK:")


def test_planted_changed_routine_missing_from_table_fails(full):
    """A changed file the table does not list (calc_gw.F, changed between the tags) makes --check fail twice: the
    generated section is stale and its forward-value hunks have no note. A row deleted from the doc also fails."""
    up, audits, problems, doc = full
    extra = au.audit_pair(up, "model/src/calc_gw.F", "model/src/calc_gw.F")
    assert extra.status == "modified" and extra.counts()[au.FWD] > 0
    extra.modules = ["planted"]
    probs = au.check(doc, audits + [extra], problems)
    assert any("stale" in p for p in probs), probs
    assert any("model/src/calc_gw.F#" in p and "no hand-written note" in p for p in probs), probs
    # the committed doc with one hunk row removed
    row = next(line for line in doc.splitlines() if line.startswith("| `model/src/cg2d.F#9`"))
    assert any("stale" in p for p in au.check(doc.replace(row + "\n", ""), audits, problems))
    # a forward-value hunk whose note is removed
    noted = doc.replace("`ini_cg2d.F#7`", "`ini_cg2d.F`")
    assert any("model/src/ini_cg2d.F#7" in p for p in au.check(noted, audits, problems))


def test_override_misclassification_is_caught(full):
    """Overrides are checked against the hunk they name: below the rules' floor, stale content sha, a hunk that does
    not exist, a class the rules already give, and an unknown class are all errors."""
    up, audits, problems, doc = full
    real = dict(au.OVERRIDES)
    cg = _hunk(audits, "model/src/cg2d.F#14")           # debug block: floor diagnostics (WRITE, PRINT_MESSAGE)
    tadv = _hunk(audits, "model/src/temp_integrate.F#5")  # CADJ lines: floor AD-only
    fwd = _hunk(audits, "model/src/cg2d.F#9")            # plain forward-value hunk
    cases = {
        ("model/src/cg2d.F", 14): ((au.COS, cg.sha, "planted: hide a debug block"), "below the rules' floor"),
        ("model/src/temp_integrate.F", 5): ((au.DIAG, tadv.sha, "planted"), "below the rules' floor"),
        ("model/src/cg2d.F", 9): ((au.DIAG, "0123456789", "planted"), "stale content sha"),
        ("model/src/cg2d.F", 99): ((au.DIAG, "0123456789", "planted"), "no such hunk"),
        ("model/src/cg2d.F", 7): ((au.FWD, _hunk(audits, "model/src/cg2d.F#7").sha, "planted"), "repeats"),
        ("model/src/ini_cg2d.F", 7): (("harmless", _hunk(audits, "model/src/ini_cg2d.F#7").sha, "x"), "unknown class"),
    }
    for key, (entry, expect) in cases.items():
        planted = dict(real)
        planted[key] = entry
        probs = []
        fa = au.audit_items([key[0]], up, overrides=planted, problems=probs)
        assert any(expect in p for p in probs), (key, probs)
        if key[1] in (14, 5):    # the bad override is not applied: the hunk keeps the rules' class
            assert _hunk(fa, f"{key[0]}#{key[1]}").cls == au.FWD
    assert fwd.cls == au.FWD
    # the committed overrides are all valid and all applied
    assert not [p for p in problems if p.startswith("override")]
    assert all(_hunk(audits, f"{p}#{n}").how.startswith("override") for p, n in real)


SRC = """\
#include "CPP_OPTIONS.h"
C $Header: /u/gcmpack/MITgcm/model/src/x.F,v 1.1 $
      SUBROUTINE X( a, b, myThid )
      _RL a, b
      INTEGER myThid
      CHARACTER*(MAX_LEN_MBUF) msgBuf
C     a comment
      a = b + 1. _d 0
      IF ( a .GT. 0. ) THEN
        b = 2. _d 0*a
      ENDIF
      WRITE(msgBuf,'(A,E12.4)') 'X: a =', a
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit, SQUEEZE_RIGHT, myThid )
      OPEN( 10, file='x.data', status='old' )
#ifdef ALLOW_AUTODIFF
      b = b*a
#endif /* ALLOW_AUTODIFF */
#ifdef ALLOW_AUTODIFF_TAMC
CADJ STORE a = comlev1, key = ikey_dynamics, kind = isbyte
#endif
      a = a - b
      RETURN
      END
"""


def _classes(old, new, path="model/src/x.F"):
    return [h.cls for h in au.classify_texts(old, new, path)]


# (name, old text in SRC, new text, expected class of the single hunk)
CASES = [
    ("operator", "      a = b + 1. _d 0\n", "      a = b - 1. _d 0\n", au.FWD),
    ("constant", "      a = b + 1. _d 0\n", "      a = b + 2. _d 0\n", au.FWD),
    ("inside_ifdef_AD", "      b = b*a\n#endif /* ALLOW_AUTODIFF */\n", "      b = b*a*a\n#endif /* ALLOW_AUTODIFF */\n",
     au.AD),
    ("CADJ_only (a TAF directive is not a comment)", "CADJ STORE a = comlev1, key = ikey_dynamics, kind = isbyte\n",
     "CADJ STORE a,b = comlev1, key = ikey_dynamics, kind = isbyte\n", au.AD),
    ("comment", "C     a comment\n", "C     another comment\n", au.COS),
    ("cvs_header", "C $Header: /u/gcmpack/MITgcm/model/src/x.F,v 1.1 $\n", "", au.COS),
    ("layout_case", "      a = b + 1. _d 0\n", "        A = B+1. _d 0\n", au.COS),
    ("cpp_comment", "#endif /* ALLOW_AUTODIFF */\n", "#endif\n", au.COS),
    ("write_format", "      WRITE(msgBuf,'(A,E12.4)') 'X: a =', a\n", "      WRITE(msgBuf,'(A,E22.14)') 'X: a =', a\n",
     au.DIAG),
    ("open_action", "      OPEN( 10, file='x.data', status='old' )\n",
     "      OPEN( 10, file='x.data', status='old', ACTION='read' )\n", au.IO),
    ("wrap existing code in #ifdef AD (plain builds lose it)", "      a = a - b\n      RETURN\n",
     "#ifdef ALLOW_AUTODIFF\n      a = a - b\n#endif\n      RETURN\n", au.FWD),
    ("wrap existing code in #ifndef AD (AD builds lose it)", "      a = a - b\n      RETURN\n",
     "#ifndef ALLOW_AUTODIFF\n      a = a - b\n#endif\n      RETURN\n", au.AD),
    ("new AD-only block", "      a = a - b\n      RETURN\n",
     "#ifdef ALLOW_AUTODIFF\n      b = 0.\n#endif\n      a = a - b\n      RETURN\n", au.AD),
    ("new declaration", "      INTEGER myThid\n", "      INTEGER myThid, k\n", au.COS),
    ("_RL -> _RS (both real*8)", "      _RL a, b\n", "      _RS a\n      _RL b\n", au.COS),
    ("a variable changes type: undecided, needs an override", "      INTEGER myThid\n", "      _RL myThid\n",
     au.UNDECIDED),
    ("debug-only IF", "      a = a - b\n      RETURN\n",
     "      a = a - b\n      IF ( debugLevel.GE.debLevB ) CALL DEBUG_MSG('x', myThid)\n      RETURN\n", au.DIAG),
]


def test_rules_classify_synthetic_hunks():
    """One test for all cases (tier 1 counts tests against a budget of 100); the message lists every mismatch."""
    wrong = []
    for name, old, new, expect in CASES:
        assert old in SRC, name
        got = _classes(SRC, SRC.replace(old, new, 1))
        if got != [expect]:
            wrong.append(f"{name}: got {got}, expected [{expect!r}]")
    assert not wrong, "\n".join(wrong)


def test_rules_whole_file_cases():
    """File-level rules and moves: a flow file is AD-only; a #define in an options header is forward value; moving a
    statement across a context line is not cosmetic."""
    flow_old = "CADJ SUBROUTINE cg2d    ADNAME  = cg2d\nCADJ SUBROUTINE cg2d    DEPEND  =   8\n"
    flow_new = "CADJ SUBROUTINE cg2d    ADNAME  = cg2d_mad\nCADJ SUBROUTINE cg2d    DEPEND  =   8\n"
    assert _classes(flow_old, flow_new, "pkg/autodiff/cg2d.flow") == [au.AD]
    opt_old = "#ifndef CPP_OPTIONS_H\n#define CPP_OPTIONS_H\n#undef ALLOW_SRCG\n#endif\n"
    assert _classes(opt_old, opt_old.replace("#undef ALLOW_SRCG", "#define ALLOW_SRCG"),
                    "model/inc/CPP_OPTIONS.h") == [au.FWD]
    moved = SRC.replace("      a = b + 1. _d 0\n      IF", "      IF").replace(
        "      ENDIF\n", "      ENDIF\n      a = b + 1. _d 0\n")
    assert set(_classes(SRC, moved)) == {au.FWD}


def test_routine_interface_for_task4(full):
    """Task 4 passes routine names: they resolve to the defining file at both tags (AD-tool stubs excluded), a file
    given twice is audited once, and a name defined nowhere is an error."""
    up = full[0]
    fas = au.audit_items(["ADAMS_BASHFORTH3", "model/src/adams_bashforth3.F", "global_sum_tile_rl"], up)
    assert [fa.path for fa in fas] == ["model/src/adams_bashforth3.F", "eesupp/src/global_sum_tile.F"]
    assert [h.cls for h in fas[0].hunks].count(au.FWD) == 1
    with pytest.raises(SystemExit):
        au.audit_items(["NO_SUCH_ROUTINE_XYZ"], up)
