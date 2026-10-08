"""Lane M4ADLAB (M4 step 7, session 1): the targets of lab_sea/input_ad (code_ad build) and TAF's adjoint mode.

Seconds-scale file checks, each with a planted negative control:
- TAF's reference output (verification/lab_sea/results/output_adm.txt, checkpoint69p, built 2026-08-12) and lane A's
  FD oracle of the 63cdc0b code_ad forward (job 27856073, fdzero run) print the same reference cost and the same
  fc+ / fc- / finite-difference gradient at all five grdchk points (xx_atemp record 1 at (6..10, 8, 1), eps 1e-3) in
  every printed digit: TAF's forward is the 63cdc0b forward, so the oracle's FD column is TAF's, and our forward has
  one target for both. TAF's adjoint gradients (ADM adjoint_gradient) are recorded here as the adjoint's target.
- data.autodiff of input_ad is empty: TAF's ADAUTODIFF_INADMODE_SET (autodiff_inadmode_set_ad.F:53-80) leaves every
  switch as the forward runs it (pkg/autodiff/autodiff_readparms.py adjoint_mode_check).
"""

import re

import pytest

EXP = ("lab_sea", "input_ad")
FD_JOB = "job27856073-fdzero"
# TAF's ADM adjoint_gradient at the five grdchk points (output_adm.txt:4136, 4234, 4332, 4430, 4528)
TAF_ADM_GRAD = ("1.84490010605889E-04", "1.78821436501969E-04", "2.32024652275522E-04", "2.97982702085000E-04",
                "3.74871810303031E-04")
REF_FC = "7.23649445797779E+03"

_PAT = {"fcref": r"grdchk reference fc: fcref\s+=\s+(\S+)",
        "fcplus": r"grdchk perturb\(\+\)fc: fcpertplus\s+=\s+(\S+)",
        "fcminus": r"grdchk perturb\(-\)fc: fcpertminus\s+=\s+(\S+)",
        "refcost": r"ADM\s+ref_cost_function\s+=\s+(\S+)",
        "adgrad": r"ADM\s+adjoint_gradient\s+=\s+(\S+)",
        "fdgrad": r"ADM\s+finite-diff_grad\s+=\s+(\S+)",
        "pos": r"grdchk pos: i,j,k=\s+(\d+)\s+(\d+)\s+(\d+)"}


def grdchk_values(text):
    """{key: [printed values in order]} of the grdchk lines of a STDOUT (strings: compared digit for digit)."""
    out = {}
    for k, p in _PAT.items():
        out[k] = [m.groups() if k == "pos" else m.group(1) for m in re.finditer(p, text)]
    return out


def _texts():
    from mitjax import paths
    taf = (paths.UPSTREAM / "verification" / EXP[0] / "results" / "output_adm.txt").read_text()
    fd = (paths.REFERENCE_RUNS / EXP[0] / EXP[1] / FD_JOB / "rundir" / "output.txt").read_text()
    return taf, fd


def mismatches(taf, fd):
    """The grdchk values on which TAF's output and the FD oracle differ (every key but the adjoint gradient, which
    the forward-only oracle prints as 0)."""
    a, b = grdchk_values(taf), grdchk_values(fd)
    return {k: (a[k], b[k]) for k in _PAT if k != "adgrad" and a[k] != b[k]}


def test_taf_reference_equals_fd_oracle_every_digit():
    taf, fd = _texts()
    v = grdchk_values(taf)
    assert v["pos"] == [("6", "8", "1"), ("7", "8", "1"), ("8", "8", "1"), ("9", "8", "1"), ("10", "8", "1")]
    assert v["fcref"] == [REF_FC] and set(v["refcost"]) == {REF_FC}
    assert tuple(v["adgrad"]) == TAF_ADM_GRAD
    assert len(v["fcplus"]) == len(v["fcminus"]) == len(v["fdgrad"]) == 5
    assert mismatches(taf, fd) == {}
    # TAF's checkpoint and build (staleness check, as lane M4ADCOL found a c69e file for the column): current
    assert "// MITgcmUV version:  checkpoint69p" in taf
    # negative control: one digit planted in the oracle's fc+ of point 3 is found
    planted = fd.replace("fcpertplus  =  7.23649445920993E+03", "fcpertplus  =  7.23649445920994E+03", 1)
    assert planted != fd
    assert set(mismatches(taf, planted)) == {"fcplus"}


def test_adjoint_mode_equals_forward_and_control():
    from mitjax.config.params import load
    from mitjax.drivers.model import with_namelist
    from mitjax.pkg.autodiff.autodiff_readparms import adjoint_mode_check
    e = load(*EXP)
    v = adjoint_mode_check(e)
    # data.autodiff (input_ad) is an empty AUTODIFF_PARM01: the defaults of autodiff_readparms.F:82-98, :145-159
    assert v["inAdExact"] is True and v["useKPPinAdMode"] is True and v["useSEAICEinAdMode"] is True
    assert v["SEAICEapproxLevInAd"] == 0 and v["useApproxAdvectionInAdMode"] is False
    # negative control: SEAICEuseDYNAMICSswitchInAd would switch the sea-ice dynamics off in the adjoint sweep
    bad = with_namelist(e, {("data.autodiff", "AUTODIFF_PARM01", "SEAICEuseDYNAMICSswitchInAd"): (True, bool)})
    with pytest.raises(NotImplementedError, match="SEAICEuseDYNAMICSswitchInAd"):
        adjoint_mode_check(bad)
