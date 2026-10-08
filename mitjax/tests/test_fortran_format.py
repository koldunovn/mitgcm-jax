"""mitjax/io/fortran_format.py against gfortran 11.2.0 itself (plan Task 11, lane MON).

Oracle: mitjax/tests/fortran_format/probe_gfortran11.out, the output of probe.F (make_probe.py there) built with the
oracle's compiler and flags (gfortran 11.2.0, -O0 -ffp-contract=off ...; build line in make_probe.py). Every
(value, edit descriptor) row is compared character for character: signed zeros, subnormals, 3-digit exponents,
exact rounding ties, NaN and infinities. The planted controls show that the comparison bites: a format change and a
rounding change each break rows of the table.
"""

import struct
from pathlib import Path

import pytest

from mitjax.io import fortran_format as ff

HERE = Path(__file__).resolve().parent / "fortran_format"
TABLE = HERE / "probe_gfortran11.out"


def _load_probe():
    spec = __import__("importlib.util").util.spec_from_file_location("_mjx_make_probe", HERE / "make_probe.py")
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_probe()


def _rows():
    """[(kind, fmt, value, field)] of the gfortran table."""
    rows = []
    for ln in TABLE.read_text().splitlines():
        if ln.startswith("F") or ln.startswith("I"):
            head, field = ln.split("|", 1)
            field = field[:-1]          # the trailing '|'
            kind, fid, val = head.split()
            fid = int(fid)
            if kind == "F":
                x = struct.unpack(">d", bytes.fromhex(val))[0]
                rows.append(("F", PROBE.FORMATS[fid - 1], x, field))
            else:
                rows.append(("I", PROBE.IFORMATS[fid - 1], int(val), field))
    return rows


ROWS = _rows()


def _ours(fmt, x):
    if fmt == "(1P2E11.3)":
        return ff.fortran_write(fmt, x, -x)
    return ff.fortran_write(fmt, x)


def _mismatches(write=_ours):
    bad = []
    for kind, fmt, x, field in ROWS:
        got = write(fmt, x).rstrip()      # the probe trims trailing blanks (LEN_TRIM)
        if got != field:
            bad.append((fmt, x, field, got))
    return bad


def test_table_complete():
    """The table holds every value with every descriptor (F formats on moderate values only) and every integer."""
    nvals = len(PROBE.VALUES) + len(PROBE.SPECIAL)
    nf_moderate = sum(1 for v in PROBE.VALUES if abs(v) < PROBE.F_MAX)
    n_f_formats = sum(1 for f in PROBE.FORMATS if f.startswith("(F") or f.startswith("(0PF"))
    assert n_f_formats == 3
    want = nvals * (len(PROBE.FORMATS) - n_f_formats) + nf_moderate * n_f_formats \
        + len(PROBE.INTS) * len(PROBE.IFORMATS)
    assert len(ROWS) == want, (len(ROWS), want)
    assert (HERE / "probe.F").read_text() == PROBE.fortran_source(), "probe.F is not what make_probe.py writes"


def test_matches_gfortran_character_for_character():
    bad = _mismatches()
    assert not bad, f"{len(bad)} of {len(ROWS)} fields differ from gfortran, e.g. {bad[:5]}"


def test_monitor_line_fields():
    """The two descriptors of MON_OUT_ALL (mon_out.F:195,197) on values of the oracle's own %MON lines."""
    assert ff.fortran_write("(1X,1P1E21.13)", 0.96840090317157) == "   9.6840090317157E-01"
    assert ff.fortran_write("(1X,I21)", 36000) == " " + " " * 16 + "36000"
    assert ff.fortran_write("(1X,1P1E21.13)", -0.0) == "  -0.0000000000000E+00"
    assert ff.fortran_write("(1X,1P1E21.13)", 1.0e100) == "   1.0000000000000+100"


def test_planted_format_change_bites():
    """Negative control: 1PE21.13 written as 1PE21.12, and E21.13 without the 1P scale, break the table."""
    def planted_digits(fmt, x):
        return _ours(fmt.replace("E21.13", "E21.12"), x)

    def planted_scale(fmt, x):
        return _ours(fmt.replace("1P1E21.13", "1E21.13"), x)

    n13 = sum(1 for r in ROWS if r[1] == "(1X,1P1E21.13)")
    bad_digits = [b for b in _mismatches(planted_digits) if b[0] == "(1X,1P1E21.13)"]
    bad_scale = [b for b in _mismatches(planted_scale) if b[0] == "(1X,1P1E21.13)"]
    # every finite row changes with d; with k every finite nonzero row (0.0000000000000E+00 reads the same for k=0
    # and k=1); NaN/Infinity rows depend on neither
    nzero = sum(1 for v in PROBE.VALUES if v == 0.0)
    assert len(bad_digits) == n13 - len(PROBE.SPECIAL), (len(bad_digits), n13)
    assert len(bad_scale) == n13 - len(PROBE.SPECIAL) - nzero, (len(bad_scale), n13)


def test_planted_rounding_change_bites(monkeypatch):
    """Negative control: round-half-up instead of the exact value's correct rounding breaks the tie rows."""
    from decimal import ROUND_HALF_UP, Decimal

    def half_up(x, nsig):
        dx = Decimal(abs(x))
        e10 = dx.adjusted()
        q = dx.scaleb(-e10).quantize(Decimal(1).scaleb(-(nsig - 1)), rounding=ROUND_HALF_UP)
        if q >= 10:
            q, e10 = q / 10, e10 + 1
            q = q.quantize(Decimal(1).scaleb(-(nsig - 1)), rounding=ROUND_HALF_UP)
        return str(q).replace(".", "").ljust(nsig, "0")[:nsig], e10

    monkeypatch.setattr(ff, "_digits", half_up)
    bad = _mismatches()
    ties = {v for v in PROBE.VALUES if v in (100000000000005.0, 100000000000025.0, 1.0625, 123456785.0)}
    assert ties <= {b[1] for b in bad}, sorted({b[1] for b in bad})


@pytest.mark.parametrize("fmt", ["(1X,Z16.16)", "1PE21.13", "(E21)", "(1PF8.2)"])
def test_unsupported_formats_raise(fmt):
    with pytest.raises(ff.FortranFormatError):
        ff.fortran_write(fmt, 1.0)
