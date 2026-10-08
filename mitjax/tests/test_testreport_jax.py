"""Plan Task 2: tools/testreport_jax.py (literal port of MITgcm's testreport comparison) and mitjax/io/stdout.py.

The port is checked against testreport itself, not against expectations written here: tools/testreport_upstream.sh
runs testreport's own functions (testoutput_run, testoutput_var, formatresults, linkdata, extracted from
verification/testreport @63cdc0b) with the tr_cmpnum that testreport's createcodelet compiled into
$MJX_REFERENCE/tools/63cdc0b/ (BUILD_INFO.txt there: compiler and sha256). Reference pairs: results/ files of
different checkpoints of the same variant (global_ocean.90x40x15 checkpoint65z vs 67v; its dwnslp variant 65z vs 67v,
whose physics changed in between; tutorial_global_oce_optim checkpoint68c vs 68x, an adjoint run) and two planted
copies that exercise the quirks (NaN, Inf, D and 3-digit exponents, a second '=', a second field, unequal line counts,
the default check list with ptracer pruning, a check list inherited through linkdata). Each gate's negative control is
a planted error in the port that must make the gate fail (in the tests, and in mutation runs on scratch
copies, 2026-10-01: 17 of 18 planted errors caught, the survivor an equivalent mutant).

Oracle-dependent tests (upstream clone, compiled tr_cmpnum) fail when their data is missing; they never skip.
"""

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mitjax import paths
from mitjax.io import stdout as so
from mitjax import testreport_jax as trj

REPO = Path(__file__).resolve().parents[2]
PINNED = "63cdc0b"
VERIF = paths.UPSTREAM / "verification"
TR_DIR = paths.REFERENCE / "tools" / PINNED
HARNESS = REPO / "tools" / "testreport_upstream.sh"

# every results/output*.txt of the M1 experiments (plan Task 2), with the input dir testreport runs it from
M1_FILES = {
    "tutorial_barotropic_gyre/results/output.txt": "input",
    "tutorial_baroclinic_gyre/results/output.txt": "input",
    "advect_xy/results/output.txt": "input",
    "advect_xy/results/output.ab3_c4.txt": "input.ab3_c4",
    "advect_xz/results/output.txt": "input",
    "advect_xz/results/output.nlfs.txt": "input.nlfs",
    "advect_xz/results/output.pqm.txt": "input.pqm",
    "global_ocean.90x40x15/results/output.txt": "input",
    "global_ocean.90x40x15/results/output.dwnslp.txt": "input.dwnslp",
    "global_ocean.90x40x15/results/output.idemix.txt": "input.idemix",
    "global_ocean.90x40x15/results/output_adm.txt": "input_ad",
    "global_ocean.90x40x15/results/output_adm.bottomdrag.txt": "input_ad.bottomdrag",
    "global_ocean.90x40x15/results/output_adm.kapgm.txt": "input_ad.kapgm",
    "global_ocean.90x40x15/results/output_adm.kapredi.txt": "input_ad.kapredi",
    "tutorial_global_oce_optim/results/output_adm.txt": "input_ad",
}


def _require_upstream():
    head = subprocess.run(["git", "-C", str(paths.UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True)
    assert head.returncode == 0 and head.stdout.startswith(PINNED), \
        f"MJX_UPSTREAM={paths.UPSTREAM} is not a MITgcm clone at {PINNED}: {head.stdout} {head.stderr}"


def _require_tr_cmpnum():
    assert (TR_DIR / "tr_cmpnum").is_file(), \
        f"no {TR_DIR / 'tr_cmpnum'}: build it once with `bash tools/testreport_upstream.sh build`"


def _git_show(rev, rel, dest):
    _require_upstream()
    out = subprocess.run(["git", "-C", str(paths.UPSTREAM), "show", f"{rev}:verification/{rel}"],
                         capture_output=True, check=True).stdout
    dest.write_bytes(out)
    return dest


def _plant(src_lines, pattern, occurrence, new_value):
    """Copy of `src_lines` with the value text after the last '=' of the `occurrence`-th line containing `pattern`
    replaced by new_value(old_text); returns (lines, old value text)."""
    out, seen, old = list(src_lines), 0, None
    for i, line in enumerate(out):
        if pattern in line:
            seen += 1
            if seen == occurrence:
                head, val = line.rsplit("=", 1)
                old = val.strip()
                out[i] = f"{head}=   {new_value(old)}"
                break
    assert old is not None, f"{pattern} occurrence {occurrence} not found"
    return out, old


def _digit(k):
    """A function changing the k-th significant digit of a 1PE value by one (down for a 9, else up)."""
    def change(text):
        mant, exp = text.split("E")
        sign = mant[0] if mant[0] in "+-" else ""
        digits = mant.lstrip("+-").replace(".", "")
        assert len(digits) > k and digits[0] != "0", text
        d = int(digits[k - 1])
        digits = digits[:k - 1] + str(d - 1 if d == 9 else d + 1) + digits[k:]
        return f"{sign}{digits[0]}.{digits[1:]}E{exp}"
    return change


_ninth_digit = _digit(9)


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


def _upstream_run(kind, exp, variant, output, reference, work):
    """testreport's own testoutput_run + formatresults on (output, reference): {LISTVAR, RESULTS, SUMMARY}."""
    _require_upstream()
    _require_tr_cmpnum()
    env = {"PATH": "/usr/bin:/bin", "MJX_PYTHON": sys.executable,
           **{k: str(v) for k, v in paths.ALL.items() if k != "MJX_PYTHON"}}
    r = subprocess.run(["bash", str(HARNESS), "run", str(TR_DIR), str(kind), str(VERIF / exp), variant,
                        str(output), str(reference), str(work)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    out = dict(line.split("=", 1) for line in r.stdout.splitlines() if line.split("=", 1)[0] in
               ("LISTVAR", "RESULTS", "SUMMARY"))
    assert set(out) == {"LISTVAR", "RESULTS", "SUMMARY"}, r.stdout + r.stderr
    return out


def _ours(exp, variant, output, reference):
    rep = trj.compare(output, exp, variant, reference=reference, upstream=paths.UPSTREAM)
    return {"LISTVAR": rep.run.list_var, "RESULTS": rep.run.results, "SUMMARY": rep.summary}, rep


# ---------------------------------------------------------------------------------------------------------------
# tr_cmpnum: the scanf-driven C program, case by case against the compiled binary

TR_CASES = [
    "1 1.0E+00 1.0E+00\n-1\n",                               # equal -> 16
    "1 0 0\n2 0 0\n-1\n",                                    # zeros only -> 22
    "1 0 0\n2 1.0 1.0\n-1\n",
    "1 1.0 1.000000000025\n-1\n",                            # log10(relerr) = -10.6: rint -11, truncation -10
    "1 1.0 1.00000000005\n-1\n",                             # relerr just below 5e-11
    "1 1.0 1.0000000001\n2 3.0 3.0\n-1\n",
    "1 1 2\n-1\n", "1 1 -1\n-1\n", "1 -0.0 0.0\n-1\n",
    "1 2.0962566239063-116 2.0962566239063-116\n2 1.0E+00 1.0E+00\n-1\n",   # Fortran 3-digit exponent
    "1 1.0D+00 1.0D+00\n-1\n",                               # 'D' stops scanf -> 99
    "1 1.0 nan\n-1\n", "1 nan(123) 1\n-1\n", "1 1.0 NaN\n-1\n",
    "1 inf inf\n-1\n", "1 infinity 1\n-1\n", "1 -inf 1\n-1\n", "1 Infinity -Infinity\n-1\n",
    "1 1 1\n2 inx 3\n-1\n", "1 1 1\n2 nax 3\n-1\n", "1 1 1\n2 infinx 3\n-1\n",   # failed word: consumed
    "1 1 1\n2 -x 3\n-1\n", "1 2 2\n2x 3 3\n-1\n",
    "1 1e400 1e400\n-1\n", "1 1e-400 1e-400\n-1\n",
    "1 abc 1\n-1\n", "1 1.0E+00\n-1\n", "1 .5 .5\n2 +1. 1.\n-1\n",
    "1 1.5E+ 1.5E+00\n-1\n", "1 1.5E 1.5\n-1\n", "1 100ergs 100\n-1\n", "1 . 1\n2 1 1\n",
    "+1 1 1\n-1\n", "1 1 1\n- 1\n-1\n", "", "1 1 1\n", "1 1 1\n-1", "1 1 1\n-1x\n",
    "1 1.0 1.0 7\n-1\n",
    "".join(f"{i} 1.0 1.0\n" for i in range(1, 998)) + "-1\n",     # 997 lines: compared
    "".join(f"{i} 1.0 1.0\n" for i in range(1, 999)) + "-1\n",     # 998 lines: loop limit -> 99
]


def _binary(text):
    r = subprocess.run([str(TR_DIR / "tr_cmpnum")], input=text, capture_output=True, text=True, check=True)
    return int(r.stdout)


def test_tr_cmpnum_matches_compiled_binary(tmp_path, monkeypatch):
    """Our tr_cmpnum (with its scanf emulation) prints what the compiled program prints, on edge cases and on the
    real inputs of a reference comparison. Negative control: truncation instead of rint is caught."""
    _require_tr_cmpnum()
    real = []
    go = VERIF / "global_ocean.90x40x15/results/output.txt"
    for out in (_git_show("a618cb669", "global_ocean.90x40x15/results/output.txt", tmp_path / "go_65z.txt"),
                VERIF / "global_ocean.90x40x15/results/output.dwnslp.txt"):
        rep = trj.compare(out, "global_ocean.90x40x15", reference=go, upstream=paths.UPSTREAM)
        real += [v.c_txt for v in rep.run.variables if v.c_txt]
    assert len(real) >= 30
    cases = TR_CASES + real
    want = [_binary(c) for c in cases]
    got = [trj.tr_cmpnum(c) for c in cases]
    assert got == want, [(c[:60], g, w) for c, g, w in zip(cases, got, want) if g != w]
    assert {16, 22, 99, 0, 10, 11}.issubset(want)          # the cases cover the outcomes they are meant to
    with pytest.raises(ValueError, match="uninitialised"):  # C would compute with garbage: refused, not guessed
        trj.tr_cmpnum("1 . 1\n-1\n")
    with pytest.raises(ValueError, match="uninitialised"):
        trj.tr_cmpnum("1 inx 1\n-1\n")

    monkeypatch.setattr(trj, "_rint_int", lambda x: int(x) if math.isfinite(x) else -2**31)
    planted = [trj.tr_cmpnum(c) for c in cases]
    assert planted != want


# ---------------------------------------------------------------------------------------------------------------
# testreport's digits, check lists and verdicts on four reference pairs

def _pairs(tmp_path):
    """(kind, exp, variant, output, reference) for the five reference pairs."""
    pairs = []
    go = "global_ocean.90x40x15/results/output.txt"
    pairs.append((0, "global_ocean.90x40x15", "input", _git_show("a618cb669", go, tmp_path / "go_65z.txt"),
                  VERIF / go))
    op = "tutorial_global_oce_optim/results/output_adm.txt"
    pairs.append((2, "tutorial_global_oce_optim", "input_ad", _git_show("fae4a9197", op, tmp_path / "optim_68c.txt"),
                  VERIF / op))

    # tutorial_baroclinic_gyre: default check list (no tr_checklist), ptracers pruned; quirks planted
    ref = trj.read_lines(VERIF / "tutorial_baroclinic_gyre/results/output.txt")
    out, _ = _plant(ref, "cg2d_init_res", 3, _ninth_digit)                           # PS: 8-9 digits -> FAIL
    out, _ = _plant(out, "dynstat_theta_mean", 5, _digit(12))                         # Tav: 12th digit
    out, _ = _plant(out, "dynstat_salt_sd", 2, lambda v: "NaN")                       # Ssd: NaN in the output
    out, _ = _plant(out, "dynstat_vvel_min", 4, lambda v: "-Infinity")                # Vmn: Inf in the output
    out, _ = _plant(out, "dynstat_salt_mean", 3, lambda v: f"x= {v}")                 # Sav: sed keeps after the last =
    out, _ = _plant(out, "dynstat_uvel_sd", 3, lambda v: f"{v} 7.0")                  # Usd: awk keeps 3 join fields
    rp, _ = _plant(ref, "dynstat_uvel_max", 6, lambda v: "2.0962566239063-116")       # Umx: 3-digit exponent
    rp, _ = _plant(rp, "dynstat_salt_max", 2, lambda v: v.replace("E", "D"))          # Smx: D exponent
    rp, _ = _plant(rp, "dynstat_vvel_sd", 3, lambda v: "NaN")                         # Vsd: NaN in the reference
    pairs.append((0, "tutorial_baroclinic_gyre", "input", _write(tmp_path / "baroc_out.txt", out),
                  _write(tmp_path / "baroc_ref.txt", rp)))

    # advect_xy.ab3_c4: check list inherited from input/tr_checklist (linkdata), first variable Tsd
    ref = trj.read_lines(VERIF / "advect_xy/results/output.ab3_c4.txt")
    out, _ = _plant(ref, "dynstat_theta_sd", 4, _digit(11))                           # Tsd: 11th digit
    drop = [i for i, x in enumerate(out) if "dynstat_uvel_max" in x][2]
    out = out[:drop] + out[drop + 1:]                                                 # Umx: unequal line counts
    pairs.append((0, "advect_xy", "input.ab3_c4", _write(tmp_path / "axy_out.txt", out),
                  VERIF / "advect_xy/results/output.ab3_c4.txt"))

    # global_ocean.90x40x15.dwnslp checkpoint65z vs 67v: the variant's own check list (ptracers, not the base dir's
    # SBO list), ptracer 1 kept and 2-5 pruned
    dw = "global_ocean.90x40x15/results/output.dwnslp.txt"
    pairs.append((0, "global_ocean.90x40x15", "input.dwnslp", _git_show("a618cb669", dw, tmp_path / "dw_65z.txt"),
                  VERIF / dw))
    return pairs


def test_digits_equal_testreport_on_reference_pairs(tmp_path, monkeypatch):
    """listVar, every variable's digits, the summary line and the verdict equal testreport's own on five pairs.
    Negative controls: truncation instead of rint, and an exact-name grep (no `sbo_mass` in `sbo_mass_fw`), each
    make at least one pair disagree."""
    pairs = _pairs(tmp_path)
    upstream, ours, reps = [], [], []
    for n, (kind, exp, variant, out, ref) in enumerate(pairs):
        upstream.append(_upstream_run(kind, exp, variant, out, ref, tmp_path / f"work{n}"))
        o, rep = _ours(exp, variant, out, ref)
        ours.append(o)
        reps.append(rep)
    assert ours == upstream

    # the pairs are not trivial: digits below 16 and every outcome class occurs
    digits = [v.digits for rep in reps for v in rep.run.variables]
    assert {8, 9, 10, 11, 12, 13}.intersection(digits) and 16 in digits and 99 in digits
    verdicts = [rep.verdict for rep in reps]
    assert verdicts == ["pass", "pass", "FAIL", "pass", "FAIL"], verdicts   # dwnslp: physics changed (d4fc20add)
    by = {v.name: v.digits for v in reps[2].run.variables}
    assert by["PS"] in (8, 9) and by["Tav"] in (11, 12) and by["Ssd"] == by["Vmn"] == by["Smx"] == by["Vsd"] == 99
    assert by["Sav"] == 16 and by["Usd"] < 10
    assert reps[0].run.list_var.split()[-4:] == ["sbo_M", "sboFW", "sboAc", "sboAp"]
    assert reps[3].checklist_files["tr_checklist"] == VERIF / "advect_xy/input/tr_checklist"
    assert reps[4].checklist_files["tr_checklist"] == VERIF / "global_ocean.90x40x15/input.dwnslp/tr_checklist"
    assert reps[4].run.list_var.split()[-4:] == ["pt1mn", "pt1mx", "pt1av", "pt1sd"]

    def ours_all():
        return [_ours(exp, variant, out, ref)[0] for kind, exp, variant, out, ref in pairs]

    with monkeypatch.context() as m:
        m.setattr(trj, "_rint_int", lambda x: int(x) if math.isfinite(x) else -2**31)
        assert ours_all() != upstream
    with monkeypatch.context() as m:
        m.setattr(trj, "_grep", lambda p, lines: [x for x in lines if re.search(re.escape(p) + r"\s*=", x)])
        assert ours_all() != upstream


# ---------------------------------------------------------------------------------------------------------------
# self comparison, planted 9th digit, NaN

def _n_lines(path, pattern):
    return sum(pattern in line for line in trj.read_lines(path))


def test_self_comparison_full_digits():
    """Every M1 reference file compared with itself: each variable with at least 2 lines matches to all digits
    (16; 22 when all values are zero), variables with fewer lines are not compared (99), verdict pass."""
    _require_upstream()
    for rel, variant in M1_FILES.items():
        exp = rel.split("/")[0]
        path = VERIF / rel
        rep = trj.compare(path, exp, variant, reference=path, upstream=paths.UPSTREAM)
        assert rep.verdict == "pass", (rel, rep.summary)
        for v in rep.run.variables:
            n = _n_lines(path, v.pattern)
            assert v.digits in ({99} if n < 2 else {16, 22}), (rel, v.name, n, v.digits)
        assert sum(v.digits in (16, 22) for v in rep.run.variables) >= 3, rel


def test_planted_ninth_digit_fails_that_variable_only(tmp_path, capsys):
    """A change in the 9th significant digit of one value: that variable drops below MATCH_CRIT, every other variable
    keeps its digits. In the first check-list variable it makes the verdict FAIL (exit code 1); elsewhere testreport
    still passes (exit code 0)."""
    ref = VERIF / "global_ocean.90x40x15/results/output.txt"
    base = trj.compare(ref, "global_ocean.90x40x15", reference=ref, upstream=paths.UPSTREAM)
    full = {v.name: v.digits for v in base.run.variables}
    for pattern, name, verdict, code in [("cg2d_init_res", "PS", "FAIL", 1),
                                         ("dynstat_theta_mean", "Tav", "pass", 0),
                                         ("sbo_zoamc", "sboAc", "pass", 0)]:
        lines, old = _plant(trj.read_lines(ref), pattern, 4, _ninth_digit)
        out = _write(tmp_path / f"{name}.txt", lines)
        rep = trj.compare(out, "global_ocean.90x40x15", reference=ref, upstream=paths.UPSTREAM)
        got = {v.name: v.digits for v in rep.run.variables}
        assert got[name] in (8, 9) and got[name] < trj.MATCH_CRIT, (name, old, got[name])
        assert {k: d for k, d in got.items() if k != name} == {k: d for k, d in full.items() if k != name}
        assert rep.verdict == verdict
        assert trj.main([str(out), "--exp", "global_ocean.90x40x15", "--reference", str(ref),
                         "--upstream", str(paths.UPSTREAM)]) == code
        assert rep.summary in capsys.readouterr().out
    # testreport's grep for sbo_M ("sbo_mass") also takes the sbo_mass_fw lines: a plant there hits both variables
    lines, _ = _plant(trj.read_lines(ref), "sbo_mass_fw", 4, _ninth_digit)
    rep = trj.compare(_write(tmp_path / "fw.txt", lines), "global_ocean.90x40x15", reference=ref,
                      upstream=paths.UPSTREAM)
    got = {v.name: v.digits for v in rep.run.variables}
    assert got["sboFW"] < trj.MATCH_CRIT and got["sbo_M"] < trj.MATCH_CRIT and got["sboAc"] == full["sboAc"]


def test_nan_in_output(tmp_path):
    """NaN (or Inf) in the first variable: no comparison (99), verdict N/O, exit code 3 -- not a pass. In another
    variable testreport marks it '--' and still decides on the first variable; the CLI names it."""
    ref = VERIF / "tutorial_global_oce_optim/results/output_adm.txt"
    exp = ("tutorial_global_oce_optim", "input_ad")
    for value in ("NaN", "nan", "Infinity"):
        lines, _ = _plant(trj.read_lines(ref), "ADM  adjoint_gradient", 2, lambda v: value)
        out = _write(tmp_path / f"grd_{value}.txt", lines)
        rep = trj.compare(out, *exp, reference=ref, upstream=paths.UPSTREAM)
        assert rep.run.variables[1].name == "admGrd" and rep.run.variables[1].digits == 99
        assert rep.verdict == "N/O" and " N/O " in rep.summary
        assert trj.main([str(out), "--exp", exp[0], "--variant", exp[1], "--reference", str(ref),
                         "--upstream", str(paths.UPSTREAM)]) == 3
    lines, _ = _plant(trj.read_lines(ref), "ADM  finite-diff_grad", 3, lambda v: "NaN")
    rep = trj.compare(_write(tmp_path / "fd_nan.txt", lines), *exp, reference=ref, upstream=paths.UPSTREAM)
    assert {v.name: v.digits for v in rep.run.variables}["admFwd"] == 99 and rep.verdict == "pass"


# ---------------------------------------------------------------------------------------------------------------
# stdout.py parsers on the M1 reference files (counts measured with grep -c on the same files)

PARSE_EXPECT = {   # file: (forward MONITOR blocks, AD_MONITOR blocks, cg2d_init_res, SBO records, grdchk points,
                   #        ';' terminated parameter blocks, first and last time_tsnumber)
    "tutorial_barotropic_gyre/results/output.txt": (11, 0, 10, 0, 0, 250, 0, 10),
    "tutorial_baroclinic_gyre/results/output.txt": (11, 0, 10, 0, 0, 252, 0, 10),
    "advect_xy/results/output.txt": (6, 0, 0, 0, 0, 209, 0, 80),
    "advect_xz/results/output.txt": (21, 0, 0, 0, 0, 234, 0, 200),
    "global_ocean.90x40x15/results/output.txt": (11, 0, 10, 11, 0, 282, 36000, 36010),
    "global_ocean.90x40x15/results/output_adm.txt": (11, 11, 10, 0, 4, 311, 0, 10),
    "tutorial_global_oce_optim/results/output_adm.txt": (11, 0, 10, 0, 3, 288, 0, 10),
}


def test_stdout_parsers_on_m1_references(tmp_path):
    _require_upstream()
    for rel in M1_FILES:                      # every M1 file parses without an unknown-format error
        lines = so.read_stdout(VERIF / rel)
        so.monitor_blocks(lines), so.cg2d_records(lines), so.sbo_records(lines), so.grdchk(lines)
        so.parameter_dict(so.parameters(lines))
    for rel, (mon, admon, cg, sbo, npts, nblk, t0, t1) in PARSE_EXPECT.items():
        lines = so.read_stdout(VERIF / rel)
        blocks = so.monitor_blocks(lines)
        fwd = [b for b in blocks if b.kind == so.FORWARD_DYNAMICS]
        assert (len(fwd), sum(b.kind == so.ADJOINT_DYNAMICS for b in blocks)) == (mon, admon), rel
        assert so.monitor_series(blocks, "time_tsnumber")[::len(fwd) - 1] == [t0, t1], rel
        assert [b.kind for b in blocks[:2]] == [None, None] and "XC_max" in blocks[0].text, rel   # grid statistics
        assert (len(so.cg2d_records(lines)), len(so.sbo_records(lines)), len(so.grdchk(lines).points)) == \
            (cg, sbo, npts), rel
        params = so.parameters(lines)
        assert sum(p.form == "block" for p in params) == nblk, rel

    # values, checked against the text of the file
    go = so.read_stdout(VERIF / "global_ocean.90x40x15/results/output.txt")
    blk = [b for b in so.monitor_blocks(go) if b.kind == so.FORWARD_DYNAMICS][1]
    assert blk["time_tsnumber"] == 36001 and blk.text["dynstat_theta_mean"] == "3.9388941899039E+00"
    cg = so.cg2d_records(go)[0]
    assert (cg.init_res, cg.iters, cg.last_res) == (4.15897087642607e-02, (-1, 123), 9.68584296022951e-14)
    sb = so.sbo_records(go)[1]
    assert sorted(sb.text) == ["sbo_mass", "sbo_mass_fw", "sbo_zoamc", "sbo_zoamp"]
    assert sb["sbo_mass_fw"] == -2.3146196133335e13
    pd = so.parameter_dict(so.parameters(go))
    assert pd["tRef"].values() == ["2.000000000000000E+01"] * 15 and pd["buoyancyRelation"].text == "'OCEANIC'"
    assert (pd["sNx"].text, pd["nSx"].text, pd["OLx"].text, pd["Nr"].text) == ("10", "9", "3", "15")
    assert pd["sNx"].form == "inline" and pd["GM_advec*K"].values() == ["0.000000000000000E+00"]
    assert {"dxF /* dxF(:,1,:,1) ( units: m ) */", "dxF /* dxF(1,:,1,:) ( units: m ) */"} <= set(pd)
    assert pd["delX"].values() == ["4.000000000000000E+00"] * 90
    part = [p for p in pd.values() if ". . ." in p.text]
    assert [p.name for p in part] == ["xC"]
    with pytest.raises(ValueError, match="in part"):
        part[0].values()

    g = so.grdchk(so.read_stdout(VERIF / "tutorial_global_oce_optim/results/output_adm.txt"))
    assert g.fcref == 6.20023228182337
    assert [(p.i, p.j, p.k, p.bi, p.bj) for p in g.points] == [(43, 2, 1, 1, 1), (44, 2, 1, 1, 1), (45, 2, 1, 1, 1)]
    assert [p.adm["adjoint_gradient"] for p in g.points] == [-2.70384203444403e-06, -2.77397605795952e-06,
                                                             -2.69091500991181e-06]
    assert g.points[2].adm["finite-diff_grad"] == -2.69091464666360e-06 and g.points[0].fcpertplus == 6.20023202518715
    ga = so.read_stdout(VERIF / "global_ocean.90x40x15/results/output_adm.txt")
    assert [(p.i, p.j, p.bi, p.bj) for p in so.grdchk(ga).points] == [(31, 7, 2, 2), (32, 7, 2, 2), (33, 7, 2, 2),
                                                                     (34, 7, 2, 2)]
    ad = so.monitor_series(so.monitor_blocks(ga), "ad_dynstat_adtheta_min", so.ADJOINT_DYNAMICS)
    assert len(ad) == 11 and ad[0] == -4.8827045212951e-06
    # an older checkpoint closes ptracer blocks with "End MONITOR ptracers field statistics"
    old = _git_show("a618cb669", "global_ocean.90x40x15/results/output.dwnslp.txt", tmp_path / "dw_65z.txt")
    old_blocks = so.monitor_blocks(so.read_stdout(old))
    assert sum(b.kind == "MONITOR ptracer field statistics" for b in old_blocks) == 11
    assert so.fortran_number("2.0962566239063-116") == 2.0962566239063e-116
    assert so.fortran_number("1.0D+00") == 1.0 and math.isnan(so.fortran_number("NaN"))


def test_stdout_parsers_fail_loudly():
    """Negative controls: each malformed input raises (each also parses once repaired, so the plant is what bites)."""
    ok = so.read_stdout(VERIF / "tutorial_global_oce_optim/results/output_adm.txt")
    text = [f"(PID.TID {ln.pid}) {ln.text}" if ln.pid else ln.text for ln in ok]

    def lines_with(fn):
        t = list(text)
        fn(t)
        return so.split_lines("\n".join(t))

    def first(sub):
        return next(i for i, x in enumerate(text) if sub in x)

    def last(sub):
        return max(i for i, x in enumerate(text) if sub in x)

    semi = next(i for i, ln in enumerate(ok) if ln.text.strip() == ";")

    bad = {
        "unknown %MON format": (so.monitor_blocks, lambda t: t.__setitem__(first("%MON dynstat_eta_max"),
                                                                            "(PID.TID 0000.0001) %MON LAND : Area= 1")),
        "End without Begin": (so.monitor_blocks, lambda t: t.__delitem__(first("// Begin MONITOR"))),
        "never closed": (so.monitor_blocks, lambda t: t.__delitem__(last("// End MONITOR"))),
        "Begin inside": (so.monitor_blocks, lambda t: t.__delitem__(first("// End MONITOR"))),
        "before any grdchk pos": (so.grdchk, lambda t: t.__delitem__(first("grdchk pos"))),
        "out of order": (so.grdchk, lambda t: t.__delitem__(first("ADM  ref_cost_function"))),
        "unknown grdchk pos format": (so.grdchk, lambda t: t.__setitem__(
            first("grdchk pos"), "(PID.TID 0000.0001) grdchk pos: i,j,k= 1 2")),
        "parameter header inside": (so.parameters, lambda t: t.__delitem__(semi)),
        "not a Fortran number": (so.cg2d_records, lambda t: t.__setitem__(
            first("cg2d_init_res"), "(PID.TID 0000.0001)      cg2d_init_res = ***************")),
    }
    for what, (parser, plant) in bad.items():
        parser(so.split_lines("\n".join(text)))
        with pytest.raises(ValueError, match=re.escape(what)):
            parser(lines_with(plant))
