"""testreport_jax: MITgcm's testreport comparison of a model output with a verification reference, ported literally.

Ports `verification/testreport` @63cdc0b (MJX_UPSTREAM):
    testoutput_var   testreport:125-202   grep one variable's lines, `sed 's/.*=//'`, `nl`, `join`, NaN/Inf checks
    tr_cmpnum        testreport:950-998   the C program createcodelet compiles; it reads its input with scanf
    testoutput_run   testreport:204-379   check list (tr_checklist, .adm, .tlm, or the default list), `X+` expansion,
                                          ptracer pruning, the first variable decides
    formatresults    testreport:1000-1039 the summary line and the verdict (pass / FAIL / N/O, MATCH_CRIT=10)
    linkdata         testreport:771-832   which input dir a run takes its check list from (variant dir first)
and the globals they read: KIND (testreport:1146-1149), PTRACERS_NUM (1152), MATCH_CRIT (1154), the input dir and
reference name per KIND (1318-1352), the default check lists (1441-1455).

Every step is the shell's, including its quirks: `grep sbo_mass` also takes the `sbo_mass_fw` lines; a value whose
Fortran exponent has three digits (`2.0962566239063-116`) is two numbers to scanf; tr_cmpnum caps the digits at 16
for equal values and at 22 for zeros, answers 99 ("no comparison") for NaN, for fewer than 2 lines, for unequal line
counts and for 998 lines or more; the verdict is decided by the first check-list variable only, so a NaN in another
variable still passes (it shows as `--`). Inputs the shell handles in ways not emulated here (an empty value after
`sed`, nl section delimiters, NUL bytes, hexadecimal floats) raise instead of guessing. The independent check is
tools/testreport_upstream.sh, which runs testreport's own functions and the compiled tr_cmpnum on the same files
(mitjax/tests/test_testreport_jax.py).

CLI (exit code: 0 pass, 1 FAIL, 3 N/O; 2 usage error; `python tools/testreport_jax.py` is a shim for the same):
    python -m mitjax.testreport_jax OUTPUT --exp EXPERIMENT [--variant input[.X]|input_ad[.X]] [--kind fwd|tlm|adm]
                                          [--reference FILE] [--match N] [-v]
OUTPUT is a model output (our STDOUT-like text, or an oracle's STDOUT/output.txt); the reference defaults to
$MJX_UPSTREAM/verification/EXPERIMENT/results/<testreport's name for the variant> (output.txt, output.X.txt,
output_adm.txt, ...; a .gz copy is read when only that exists, as testreport does).

Part of the package since docs plan 20261006 Task 5 (R7). Stdlib only; machine paths come from mitjax/paths.py
(loaded by path when this file is, e.g. by reference/reference_runs.py: it is stdlib only too).
"""

import argparse
import gzip
import importlib.util
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------
# testreport globals
MATCH_CRIT = 10                    # testreport:1154  MATCH_CRIT=10
PTRACERS_NUM = "1 2 3 4 5"         # testreport:1152  PTRACERS_NUM="1 2 3 4 5"
KIND_FWD, KIND_TLM, KIND_ADM = 0, 1, 2   # testreport:1146-1149 (4, 5 Tapenade and 6 OpenAD are not ported)
KIND_NAMES = {"fwd": KIND_FWD, "tlm": KIND_TLM, "adm": KIND_ADM}

# input dir and reference output name per KIND (testreport:1318-1352)
INPUT_DIR = {KIND_FWD: "input", KIND_TLM: "input_ad", KIND_ADM: "input_ad"}
REF_OUTP = {KIND_FWD: "output.txt", KIND_TLM: "output_tlm.txt", KIND_ADM: "output_adm.txt"}


def default_check_list(kind):
    """DEF_CHECK_LIST (testreport:1441-1455)."""
    if kind == 0:
        return "PS PS T+ S+ U+ V+ pt1+ pt2+ pt3+ pt4+ pt5+"       # testreport:1442
    elif kind == 2:
        return "admGrd admCst admGrd admFwd T+ S+ U+ V+"           # testreport:1448
    else:
        return "admGrd admCst admGrd admFwd"                       # testreport:1452


def _expand_plus(s):
    """`sed 's/ [a-zA-Z0-9]*+/&mn &mx &av &sd/g'` (testreport:228, 1444)."""
    return re.sub(r" [a-zA-Z0-9]*\+", lambda m: f"{m[0]}mn {m[0]}mx {m[0]}av {m[0]}sd", s)


def len_check_list(kind):
    """LEN_CHECK_LIST: `echo $DEF_CHECK_LIST | sed <X+ expansion> | awk '{print NF-1}'` (testreport:1444,1450,1454)."""
    return len(_expand_plus(default_check_list(kind)).split()) - 1


def _echo(s):
    """`echo $s` (unquoted): the words of s, joined by single spaces (no word here holds a glob character)."""
    words = s.split()
    if any(set(w) & set("*?[") for w in words):
        raise ValueError(f"glob character in {s!r}: pathname expansion of `echo $x` not emulated")
    return " ".join(words)


# variable -> (grep pattern, label), testreport:286-351; `kd` is '' (forward), 'g_' (TLM) or 'ad' (adjoint)
def var_pattern(xx, kd):
    """The `case $xx in` table of testoutput_run (testreport:286-351); None for an unrecognised name."""
    table = {
        "PS": ("cg2d_init_res", "Press. Solver (cg2d)"),
        "admCst": ("ADM  ref_cost_function", "ADM CostFct"),
        "admGrd": ("ADM  adjoint_gradient", "ADM Ad Grad"),
        "admFwd": ("ADM  finite-diff_grad", "ADM FD Grad"),
        "tlmCst": ("TLM  ref_cost_function", "TLM CostFct"),
        "tlmGrd": ("TLM  tangent-lin_grad", "TLM TL Grad"),
        "tlmFwd": ("TLM  finite-diff_grad", "TLM FD Grad"),
        "Tmn": (f"dynstat_{kd}theta_min", f"{kd}Theta minimum"),
        "Tmx": (f"dynstat_{kd}theta_max", f"{kd}Theta maximum"),
        "Tav": (f"dynstat_{kd}theta_mean", f"{kd}Theta mean"),
        "Tsd": (f"dynstat_{kd}theta_sd", f"{kd}Theta Std.Dev"),
        "Smn": (f"dynstat_{kd}salt_min", f"{kd}Salt minimum"),
        "Smx": (f"dynstat_{kd}salt_max", f"{kd}Salt maximum"),
        "Sav": (f"dynstat_{kd}salt_mean", f"{kd}Salt mean"),
        "Ssd": (f"dynstat_{kd}salt_sd", f"{kd}Salt Std.Dev"),
        "Umn": (f"dynstat_{kd}uvel_min", f"{kd}U minimum"),
        "Umx": (f"dynstat_{kd}uvel_max", f"{kd}U maximum"),
        "Uav": (f"dynstat_{kd}uvel_mean", f"{kd}U mean"),
        "Usd": (f"dynstat_{kd}uvel_sd", f"{kd}U Std.Dev"),
        "Vmn": (f"dynstat_{kd}vvel_min", f"{kd}V minimum"),
        "Vmx": (f"dynstat_{kd}vvel_max", f"{kd}V maximum"),
        "Vav": (f"dynstat_{kd}vvel_mean", f"{kd}V mean"),
        "Vsd": (f"dynstat_{kd}vvel_sd", f"{kd}V Std.Dev"),
        "Etamn": (f"dynstat_{kd}eta_min", f"{kd}Eta minimum"),
        "Etamx": (f"dynstat_{kd}eta_max", f"{kd}Eta maximum"),
        "Etaav": (f"dynstat_{kd}eta_mean", f"{kd}Eta mean"),
        "Etasd": (f"dynstat_{kd}eta_sd", f"{kd}Eta Std.Dev"),
        "Qntmn": ("forcing_qnet_min", "Qnet minimum"),
        "Qntmx": ("forcing_qnet_max", "Qnet maximum"),
        "Qntav": ("forcing_qnet_mean", "Qnet mean"),
        "Qntsd": ("forcing_qnet_sd", "Qnet Std.Dev"),
        "aSImn": (f"seaice_{kd}area_min", f"SIce {kd}Area min"),
        "aSImx": (f"seaice_{kd}area_max", f"SIce {kd}Area max"),
        "aSIav": (f"seaice_{kd}area_mean", f"SIce {kd}Area mean"),
        "aSIsd": (f"seaice_{kd}area_sd", f"SIce {kd}Area StDv"),
        "hSImn": (f"seaice_{kd}heff_min", f"SIce {kd}Heff min"),
        "hSImx": (f"seaice_{kd}heff_max", f"SIce {kd}Heff max"),
        "hSIav": (f"seaice_{kd}heff_mean", f"SIce {kd}Heff mean"),
        "hSIsd": (f"seaice_{kd}heff_sd", f"SIce {kd}Heff StDv"),
        "uSImn": (f"seaice_{kd}uice_min", f"SIce {kd}Uice min"),
        "uSImx": (f"seaice_{kd}uice_max", f"SIce {kd}Uice max"),
        "uSIav": (f"seaice_{kd}uice_mean", f"SIce {kd}Uice mean"),
        "uSIsd": (f"seaice_{kd}uice_sd", f"SIce {kd}Uice StDv"),
        "vSImn": (f"seaice_{kd}vice_min", f"SIce {kd}Vice min"),
        "vSImx": (f"seaice_{kd}vice_max", f"SIce {kd}Vice max"),
        "vSIav": (f"seaice_{kd}vice_mean", f"SIce {kd}Vice mean"),
        "vSIsd": (f"seaice_{kd}vice_sd", f"SIce {kd}Vice StDv"),
        "AthSiG": ("thSI_Ice_Area_G", "thSIc Area Global"),
        "AthSiS": ("thSI_Ice_Area_S", "thSIc Area South"),
        "AthSiN": ("thSI_Ice_Area_N", "thSIc Area North"),
        "HthSiG": ("thSI_IceH_ave_G", "thSIc H Glob-ave"),
        "HthSiS": ("thSI_IceH_ave_S", "thSIc H South-av"),
        "HthSiN": ("thSI_IceH_ave_N", "thSIc H North-av"),
        "HthMxS": ("thSI_IceH_max_S", "thSIc H South-max"),
        "HthMxN": ("thSI_IceH_max_N", "thSIc H North-max"),
        "sbo_M": ("sbo_mass", "SBO mass"),
        "sboFW": ("sbo_mass_fw", "SBO m-FW"),
        "sboAc": ("sbo_zoamc", "SBO AM-C"),
        "sboAp": ("sbo_zoamp", "SBO AM-P"),
        "StrmIc": ("STREAMICE_FP_ERR", "StreamIce Solver"),
    }
    return table.get(xx)


def ptracer_pattern(nn, ii, kd):
    """The ptracer `case $ii in` of testoutput_run (testreport:355-361); None when no case matches."""
    return {
        "mn": (f"trcstat_{kd}ptracer0{nn}_min", f"{kd}pTr0{nn}_min"),
        "mx": (f"trcstat_{kd}ptracer0{nn}_max", f"{kd}pTr0{nn}_max"),
        "av": (f"trcstat_{kd}ptracer0{nn}_mean", f"{kd}pTr0{nn}_mean"),
        "sd": (f"trcstat_{kd}ptracer0{nn}_sd", f"{kd}pTr0{nn}_StDv"),
    }.get(ii)


# ---------------------------------------------------------------------------------------------------------------
# reading files as grep sees them

def read_lines(path):
    """The lines of a text file as grep reads them (bytes kept 1:1 via latin-1; a .gz file is decompressed)."""
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    if b"\0" in data:
        raise ValueError(f"{path}: NUL bytes (grep would answer 'Binary file matches'); not emulated")
    lines = data.decode("latin-1").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _grep(pattern, lines):
    """`grep "$pattern"` for the literal patterns testreport uses (no BRE special characters)."""
    if re.search(r"[.\[\]*^$\\]", pattern):
        raise ValueError(f"grep pattern {pattern!r} has BRE special characters; only literal patterns are ported")
    return [line for line in lines if pattern in line]


def _sed_value(line):
    """`sed 's/.*=//'`: drop everything up to the last '='."""
    return line.rsplit("=", 1)[-1]


_NL_DELIMITERS = ("\\:\\:\\:", "\\:\\:", "\\:")


def _nl(texts):
    """`nl` (GNU defaults: number non-empty lines, width 6, TAB): returns the numbered texts.

    An empty line is left unnumbered by nl and gives `join` an empty key; nl treats `\\:` lines as section
    delimiters. Neither is emulated: both raise."""
    for t in texts:
        if t == "":
            raise ValueError("empty value after `sed 's/.*=//'`: nl/join behaviour on unnumbered lines not emulated")
        if t in _NL_DELIMITERS:
            raise ValueError(f"line {t!r} is an nl section delimiter; not emulated")
    return [(n, t) for n, t in enumerate(texts, start=1)]


def _blank_fields(text):
    """Fields as join and awk split them by default: runs of blanks (space, TAB), leading blanks ignored."""
    return [f for f in re.split(r"[ \t]+", text) if f != ""]


# ---------------------------------------------------------------------------------------------------------------
# tr_cmpnum (testreport:955-985), with its scanf input

_C_SPACE = " \t\n\v\f\r"       # isspace() in the C locale
_DIGITS = "0123456789"


class _Scanf:
    """glibc scanf on one input stream: "%d" and "%lf" conversions, with glibc's consumption rules.

    A conversion that fails leaves its variable unchanged (the C program never checks scanf's return value). Rules
    (glibc vfscanf): leading white space is skipped; %d takes an optional sign and digits, and fails when no digit
    follows (a sign stays consumed); %lf takes an optional sign, then "nan" / "inf" / "infinity" (any case; a partial
    word fails with its characters consumed), else digits with one '.', one 'e'/'E' after a digit and a sign right
    after the 'e'; the collected text is converted by strtod, which uses its longest valid prefix (glibc accepts
    "1.5E+" as 1.5 and "100ergs" as 100 with "rgs" left). Hexadecimal input raises (not emulated)."""

    def __init__(self, text):
        self.s, self.i = text, 0

    def _peek(self, k=0):
        j = self.i + k
        return self.s[j] if j < len(self.s) else ""

    def _skip_space(self):
        while self.i < len(self.s) and self.s[self.i] in _C_SPACE:
            self.i += 1

    def d(self):
        self._skip_space()
        if self.i >= len(self.s):
            return None                                  # EOF
        start = self.i
        if self._peek() in ("+", "-"):
            self.i += 1
        d0 = self.i
        while self._peek() != "" and self._peek() in _DIGITS:
            self.i += 1
        if self.i == d0:
            return None                                  # matching failure; the failing character is not consumed
        v = int(self.s[start:self.i])
        if not -2**31 <= v < 2**31:
            raise ValueError(f"%d overflow on {self.s[start:self.i]!r}; not emulated")
        return v

    def _word(self, word):
        """Match `word` (any case); glibc reads each character before testing it, so a mismatch is consumed too."""
        for ch in word:
            c = self._peek()
            if c == "":
                return False
            self.i += 1
            if c.lower() != ch:
                return False
        return True

    def lf(self):
        self._skip_space()
        if self.i >= len(self.s):
            return None                                  # EOF
        sign = ""
        if self._peek() in ("+", "-"):
            sign = self._peek()
            self.i += 1
        c = self._peek().lower()
        if c == "n":
            return float("nan") if self._word("nan") else None
        if c == "i":
            if not self._word("inf"):
                return None
            if self._peek().lower() == "i" and not self._word("inity"):
                return None
            return -math.inf if sign == "-" else math.inf
        if c == "0" and self._peek(1) in ("x", "X"):
            raise ValueError("hexadecimal float input to scanf; not emulated")
        buf, got_digit, got_dot, got_e = sign, False, False, False
        while True:
            c = self._peek()
            if c != "" and c in _DIGITS:
                buf += c
                got_digit = True
            elif c == "." and not got_dot:
                buf += c
                got_dot = True
            elif c in ("e", "E") and got_digit and not got_e:
                buf += "e"
                got_e = got_dot = True
            elif c in ("+", "-") and got_e and buf[-1] == "e":
                buf += c
            else:
                break
            self.i += 1
        m = re.match(r"[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?", buf)
        return float(m[0]) if m else None


def _rint_int(x):
    """`(int)rint(x)`: round half to even (rint in the default rounding mode); a non-finite x converts to INT_MIN
    on x86-64 (cvttsd2si)."""
    return int(round(x)) if math.isfinite(x) else -2**31


def tr_cmpnum(text):
    """The comparison program of testreport (testreport:955-985), on its whole standard input `text`.

    Returns what it prints: the number of matching digits (-best), i.e. min over lines of -rint(log10(relerr)),
    16 for equal non-zero values, 22 when no line had a non-zero value, 29/99 for 998 lines or more, 99 when a NaN
    stopped it or the input did not end on the line number -1."""
    sc = _Scanf(text)
    linnum = a = b = None              # uninitialised in C
    undefined = False
    best = -22
    lncnt = 0
    while True:
        lncnt += 1
        if not (1 & (lncnt < 999)):
            break
        v = sc.d()
        linnum = linnum if v is None else v
        if linnum == -1:
            break
        v = sc.lf()
        a = a if v is None else v
        v = sc.lf()
        b = b if v is None else v
        if a is None or b is None:     # C would compute with an uninitialised double
            undefined = True
            continue
        abave = 0.5 * (abs(a) + abs(b))
        if abave == abave:
            if abave > 0.0:
                relerr = abs(a - b) / abave
                if relerr > 0.0:
                    cmplin = _rint_int(math.log10(relerr))
                else:
                    cmplin = -16
                best = best if best > cmplin else cmplin
            else:
                cmplin = -22
        else:
            break
    if lncnt == 999:
        best = -29
    if linnum != -1:
        best = -99
    if undefined and -best != 99:
        raise ValueError("tr_cmpnum read an uninitialised value (undefined behaviour in C)")
    return -best


# ---------------------------------------------------------------------------------------------------------------
# testoutput_var (testreport:125-202)

@dataclass
class VarResult:
    name: str                 # check-list name (PS, Tmn, admGrd, ...)
    pattern: str | None       # grep pattern (None: name not recognised)
    label: str | None
    digits: int               # what testoutput_var returns (99 = no comparison)
    message: str = ""         # testreport's verbose message
    c_txt: str = ""           # the input tr_cmpnum read


def testoutput_var(out_lines, pattern, label, ref_lines):
    """Digits of similarity for one variable: (digits, message, c_txt); digits 99 = no comparison."""
    if out_lines is None:
        return 99, "testoutput_var: output file from model run was not readable", ""
    a = _nl([_sed_value(x) for x in _grep(pattern, out_lines)])
    if len(a) < 2:
        return 99, f"Not enough lines of output when searching for {pattern}", ""
    b = _nl([_sed_value(x) for x in _grep(pattern, ref_lines)]) if ref_lines is not None else []
    if len(b) < 2:
        return 99, f"Not enough lines of output when searching for {pattern}", ""
    if len(a) != len(b):
        return 99, f"Not same Nb of lines when searching for {pattern} : {len(a)} {len(b)}", ""
    has_nan = sum("nan" in t.lower() for _, t in a)
    if has_nan > 0:
        return 99, f"testoutput_var: output contains {has_nan} NaN values", ""
    has_inf = sum("inf" in t.lower() for _, t in a)
    if has_inf > 0:
        return 99, f"testoutput_var: output contains {has_inf} Inf values", ""
    # join a b | awk '{print $1 " " $2 " " $3}' | sed -e 's|:||g' ; echo "-1" >> c.txt
    rows = []
    for (n, ta), (_, tb) in zip(a, b):      # equal keys 1..N in both files: join pairs line by line
        f = [str(n)] + _blank_fields(ta) + _blank_fields(tb) + ["", ""]
        rows.append(f"{f[0]} {f[1]} {f[2]}".replace(":", ""))
    c_txt = "\n".join(rows) + "\n-1\n"
    digits = tr_cmpnum(c_txt)
    if digits == 99:
        msg = f'testoutput_var: No comparison was available for "{label}"'
    else:
        msg = f'There were {digits} decimal places of similarity for "{label}"'
    return digits, msg, c_txt


# ---------------------------------------------------------------------------------------------------------------
# testoutput_run (testreport:204-379) and formatresults (testreport:1000-1039)

@dataclass
class RunResult:
    kind: int
    list_chk: str                         # the check list as read (echo-normalised)
    s_var: str                            # first variable: decides pass/FAIL
    list_var: str                         # listVar exactly as testreport writes it to summary.txt
    variables: list = field(default_factory=list)    # VarResult per word of listVar, in order
    results: str = ""                     # what testoutput_run echoes (e.g. "> 16 < 16 16 ... . .")
    warnings: list = field(default_factory=list)


def testoutput_run(out_lines, ref_lines, kind, checklists=None, ptracers_num=PTRACERS_NUM):
    """testoutput_run for one run. `checklists` maps tr_checklist / tr_checklist.adm / tr_checklist.tlm to the text of
    the file the run directory would hold (resolve_checklists); `ref_lines` is the reference output (results/)."""
    checklists = checklists or {}
    if kind not in (KIND_FWD, KIND_TLM, KIND_ADM):
        raise ValueError(f"KIND={kind} not ported (0, 1, 2)")
    list_chk = default_check_list(kind)
    if "tr_checklist" in checklists:
        list_chk = checklists["tr_checklist"]
    if kind in (1, 4):
        kd = "g_"
        list_chk = re.sub(" adm", " tlm", re.sub("^adm", "tlm", _echo(list_chk)))
        if "tr_checklist.tlm" in checklists:
            list_chk = checklists["tr_checklist.tlm"]
    elif kind in (2, 5):
        kd = "ad"
        if "tr_checklist.adm" in checklists:
            list_chk = checklists["tr_checklist.adm"]
    elif kind == 6:
        kd = "ad"
    else:
        kd = ""
    list_chk = _echo(list_chk)
    s_var = list_chk.split()[0] if list_chk else ""
    if not re.fullmatch(r"[A-Za-z0-9]+", s_var):
        raise ValueError(f"first check-list variable {s_var!r}: only alphanumeric names are ported")
    list_var = re.sub("^" + s_var, "", _expand_plus(list_chk).replace("+", ""), count=1)
    warnings = []
    if kind == 0:
        ptr_mon = ["trcstat_ptracerXX_min", "trcstat_ptracerXX_max", "trcstat_ptracerXX_mean", "trcstat_ptracerXX_sd"]
        for ii in ptracers_num.split():
            found = any(_grep(jj.replace("XX", "0" + ii), ref_lines or []) for jj in ptr_mon)
            if not found:
                list_var = re.sub(f" pt{ii}..", "", list_var)
    words = list_var.split()
    tst = sum(w == s_var for w in words)
    if tst == 0:
        # awk prints an empty t, and `test $tst != 1` becomes `test != 1`: a bash error (status 2), so the branch
        # that would warn and put the variable back is never taken (checked with bash 4.4); then no result is
        # bracketed by '>' '<' and formatresults judges the first result token.
        warnings.append("bash: test: !=: unary operator expected (selected var not in the list; not put back)")
    elif tst != 1:
        warnings.append(f"==> WARNING: found selected var >{s_var}< {tst} times")
        warnings.append("==> WARNING: in checked list: " + " ".join(words))
        list_var = f" {s_var} " + re.sub(f" {s_var} ", " ", list_var + " ")
    run = RunResult(kind=kind, list_chk=list_chk, s_var=s_var, list_var=list_var, warnings=warnings)

    allargs = []
    yy = None
    for xx in list_var.split():
        if not re.fullmatch(r"pt[1-9]..", xx):
            pat = var_pattern(xx, kd)
            if pat is None:
                yy = 99
                warnings.append(f"WARNING: asking for var={xx} : not recognized !")
                run.variables.append(VarResult(xx, None, None, 99, "not recognized"))
            else:
                yy, msg, c_txt = testoutput_var(out_lines, pat[0], pat[1], ref_lines)
                run.variables.append(VarResult(xx, pat[0], pat[1], yy, msg, c_txt))
        else:
            nn = re.sub("..$", "", xx.replace("pt", "", 1))
            ii = re.sub("^pt[0-9]*", "", xx)
            pat = ptracer_pattern(nn, ii, kd)
            if pat is None:
                if yy is None:
                    raise ValueError(f"ptracer variable {xx!r} matches no case and no earlier result exists")
                run.variables.append(VarResult(xx, None, None, yy, "no case matched: previous result repeated"))
            else:
                yy, msg, c_txt = testoutput_var(out_lines, pat[0], pat[1], ref_lines)
                run.variables.append(VarResult(xx, pat[0], pat[1], yy, msg, c_txt))
        if xx == s_var:
            allargs += [">", str(yy), "<"]
        else:
            allargs.append(str(yy))
    nb_var = len(list_var.split())
    n_len = len_check_list(kind)
    run.results = " ".join(allargs + ["."] * max(n_len - nb_var, 0))
    return run


def formatresults(name, results, match_crit=MATCH_CRIT, steps=("Y", "Y", "Y", "Y")):
    """(summary line as summary.txt holds it, verdict) for `results` (testoutput_run's string).

    formatresults prints the genmake/depend/make/run flags, each result token right-aligned in 3 columns, the verdict
    of the token between '>' and '<' (99 or '..' or '--': N/O; >= MATCH_CRIT: pass; else FAIL) and the name; the
    caller pipes it through `sed 's/ 99/ --/g' | sed 's/  > />/' | sed 's/  < /</'` (testreport:1779)."""
    toks = results.split()
    xx_s = re.sub("<.*", "", re.sub(".*>", "", " ".join(toks))).split()
    xx = xx_s[0] if xx_s else ""
    line = " ".join(steps) + "".join(f"{t:>3s}" for t in toks)
    if xx in ("..", "--") or xx == "99":
        verdict = "N/O"
        line += " N/O "
    elif int(xx) >= match_crit:
        verdict = "pass"
        line += " pass"
    else:
        verdict = "FAIL"
        line += " FAIL"
    line += f"  {name}"
    summary = line.replace(" 99", " --").replace("  > ", ">", 1).replace("  < ", "<", 1)
    return summary, verdict


# ---------------------------------------------------------------------------------------------------------------
# experiment layout: check lists through linkdata, reference names

def kind_of_variant(variant):
    """KIND implied by an input dir name: input[.X] -> forward, input_ad[.X] -> adjoint (TLM needs --kind tlm)."""
    base = variant.split(".", 1)[0]
    if base == "input":
        return KIND_FWD
    if base == "input_ad":
        return KIND_ADM
    raise ValueError(f"variant {variant!r}: expected input[.X] or input_ad[.X]")


def reference_name(variant, kind):
    """testreport's reference file for a run: ref_outp, or for an extra run X: `sed "s/\\./.X./g"` on it
    (testreport:1714, 1789)."""
    base = INPUT_DIR[kind]
    if variant == base:
        return REF_OUTP[kind]
    if not variant.startswith(base + "."):
        raise ValueError(f"variant {variant!r} does not belong to KIND={kind} (input dir {base})")
    ex = variant[len(base) + 1:]
    return REF_OUTP[kind].replace(".", f".{ex}.")


def resolve_checklists(exp_dir, variant, kind):
    """{check-list file name: path} as testreport's run directory would hold them: linkdata links the variant's
    files first, then the base input dir's files not present yet (testreport:812-827, 1768, 1800)."""
    base = INPUT_DIR[kind]
    dirs = [variant] if variant == base else [variant, base]
    found = {}
    for name in ("tr_checklist", "tr_checklist.adm", "tr_checklist.tlm"):
        for d in dirs:
            p = Path(exp_dir) / d / name
            if p.is_file():
                found[name] = p
                break
    return found


def _upstream():
    if __package__ == "mitjax":
        from mitjax import paths
        return paths.UPSTREAM
    spec = importlib.util.spec_from_file_location("_mjx_paths", Path(__file__).resolve().parent / "paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UPSTREAM


@dataclass
class Report:
    expname: str
    kind: int
    output: Path
    reference: Path
    checklist_files: dict
    run: RunResult
    summary: str
    verdict: str


def compare(output, exp, variant=None, kind=None, reference=None, upstream=None, match_crit=MATCH_CRIT):
    """Compare model output `output` with the reference of experiment `exp`, input dir `variant`, as testreport
    does. The reference defaults to verification/<exp>/results/<testreport's name> in the upstream clone."""
    upstream = Path(upstream) if upstream is not None else _upstream()
    exp_dir = upstream / "verification" / exp
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"no experiment directory {exp_dir}")
    if kind is None:
        kind = kind_of_variant(variant) if variant else KIND_FWD
    variant = variant or INPUT_DIR[kind]
    if not (exp_dir / variant).is_dir():
        raise FileNotFoundError(f"no input dir {exp_dir / variant}")
    if reference is None:
        reference = exp_dir / "results" / reference_name(variant, kind)
        if not reference.is_file() and Path(str(reference) + ".gz").is_file():
            reference = Path(str(reference) + ".gz")
    reference = Path(reference)
    if not reference.is_file():
        raise FileNotFoundError(f"missing reference output {reference}")
    files = resolve_checklists(exp_dir, variant, kind)
    checklists = {k: p.read_text() for k, p in files.items()}
    out_path = Path(output)
    if not out_path.is_file():
        raise FileNotFoundError(f"missing model output {out_path}")
    out_lines = read_lines(out_path)
    run = testoutput_run(out_lines, read_lines(reference), kind, checklists)
    base = INPUT_DIR[kind]
    expname = exp if variant == base else f"{exp}.{variant[len(base) + 1:]}"
    summary, verdict = formatresults(expname, run.results, match_crit)
    return Report(expname, kind, out_path, reference, files, run, summary, verdict)


EXIT = {"pass": 0, "FAIL": 1, "N/O": 3}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="testreport_jax.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", help="model output to check (STDOUT-like text)")
    ap.add_argument("--exp", required=True, help="verification experiment, e.g. global_ocean.90x40x15")
    ap.add_argument("--variant", help="input dir: input, input.X, input_ad, input_ad.X (default: the kind's base)")
    ap.add_argument("--kind", choices=sorted(KIND_NAMES), help="fwd, tlm or adm (default: from the variant)")
    ap.add_argument("--reference", help="reference output (default: results/ file testreport uses)")
    ap.add_argument("--upstream", help="MITgcm clone (default: MJX_UPSTREAM)")
    ap.add_argument("--match", type=int, default=MATCH_CRIT, help=f"MATCH_CRIT (default {MATCH_CRIT})")
    ap.add_argument("-v", "--verbose", action="store_true", help="print testoutput_var's messages")
    args = ap.parse_args(argv)
    kind = KIND_NAMES[args.kind] if args.kind else None
    rep = compare(args.output, args.exp, args.variant, kind, args.reference, args.upstream, args.match)
    names = {0: "forward", 1: "tangent linear", 2: "adjoint"}
    src = {k: str(p) for k, p in rep.checklist_files.items()}
    print(f"experiment {rep.expname} ({names[rep.kind]}), MATCH_CRIT={args.match}")
    print(f"output     {rep.output}")
    print(f"reference  {rep.reference}")
    print(f"check list '{rep.run.list_chk}'  ({src or 'testreport default'})")
    for w in rep.run.warnings:
        print(w)
    print(f"listVar='{rep.run.list_var}'")
    for v in rep.run.variables:
        mark = "first, decides" if v.name == rep.run.s_var else ""
        dig = "--" if v.digits == 99 else str(v.digits)
        print(f"  {v.name:8s} {dig:>3s}  {str(v.pattern):26s} {mark}")
        if args.verbose and v.message:
            print(f"           {v.message}")
    low = [v.name for v in rep.run.variables if v.name != rep.run.s_var and (v.digits == 99 or v.digits < args.match)]
    if low:
        print(f"note: not deciding, below {args.match} digits or not compared: {' '.join(low)}")
    print(rep.summary)
    return EXIT[rep.verdict]


if __name__ == "__main__":
    sys.exit(main())
