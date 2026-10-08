"""REAL*4 values (REVIEW_M0 #4, #15): a Fortran real literal without a `D` exponent or `_d` is a default-kind REAL*4
constant in the oracle build (no -fdefault-real-8: reference/optfile_levante_gfortran:87-88 is commented out), and a
REAL*4 namelist variable stores the READ's binary32 value.

Expected bits: gfortran 11.2.0 with the oracle's FFLAGS/FOPTIM (`-fallow-argument-mismatch -fconvert=big-endian
-fimplicit-none -mcmodel=medium -Wall -Wno-unused-dummy-argument -ffp-contract=off -O0`), each literal assigned to a
REAL*8 and printed as TRANSFER(x, 0_8); probe and outputs in $MJX_RUNS/hardening_scratch/r4probe_1790866643_2997150/
(r4probe.f, out.txt; r4nml.f, out_nml.txt for the namelist READ). The 13 cited lines are every default at 63cdc0b
(set_defaults.F, ini_parms.F, *readparms.F) whose REAL*4 literal is not exact in binary32. The last five literals are
controls: two decimals whose float64 rounding lands exactly on a binary32 midpoint (rounding twice, decimal -> float64
-> float32, gives the wrong neighbour; gfortran rounds once), and three everyday values.
"""

import struct

import numpy as np
import pytest

from mitjax.config import fortran, namelists
from mitjax.io import namelist as nlio
from mitjax.params_io import fortran_literal

GFORTRAN = [
    ("0.075", "3fb3333340000000"),  # pkg/bling/bling_readparms.F:366
    ("0.6", "3fe3333340000000"),  # pkg/bling/bling_readparms.F:370
    ("3.887", "400f189380000000"),  # pkg/bling/bling_readparms.F:371
    ("1.e-6", "3eb0c6f7a0000000"),  # pkg/ctrl/optim_readparms.F:87
    ("1.e-6", "3eb0c6f7a0000000"),  # pkg/ctrl/optim_readparms.F:88
    ("-1.e-6", "beb0c6f7a0000000"),  # pkg/ctrl/optim_readparms.F:89
    ("1.0e-12", "3d71979980000000"),  # pkg/streamice/streamice_readparms.F:248
    ("1.0e-6", "3eb0c6f7a0000000"),  # pkg/streamice/streamice_readparms.F:249
    ("1e-6", "3eb0c6f7a0000000"),  # pkg/streamice/streamice_readparms.F:253
    ("1e-6", "3eb0c6f7a0000000"),  # pkg/streamice/streamice_readparms.F:254
    ("1.e-14", "3d06849b80000000"),  # pkg/streamice/streamice_readparms.F:255
    ("1.e-14", "3d06849b80000000"),  # pkg/streamice/streamice_readparms.F:258
    ("1.e-14", "3d06849b80000000"),  # pkg/streamice/streamice_readparms.F:259
    ("1.000000059604644776257986737988403547205962240695953369140625", "3ff0000020000000"),
    ("1.000000178813934325304513262011596452794037759304046630859375", "3ff0000020000000"),
    ("0.1", "3fb99999a0000000"),
    ("1.1", "3ff19999a0000000"),
    ("3.14159265358979", "400921fb60000000"),
]


def _bits(x):
    return struct.pack(">d", x).hex()


def test_real4_literals_equal_gfortran():
    got = [(t, _bits(fortran_literal(t)), want) for t, want in GFORTRAN]
    assert [g for g in got if g[1] != g[2]] == []
    # negative control: the float64 value of the text (the old fortran_literal) differs on every one of the 13
    # cited defaults, and rounding twice (float32 of the float64) differs on the two midpoint controls
    assert all(_bits(float(t)) != want for t, want in GFORTRAN[:13])
    assert [t[:12] for t, want in GFORTRAN if _bits(float(np.float32(float(t)))) != want] == ["1.0000000596", "1.0000001788"]
    # D exponent and the `_d` marker (tools/set64bitConst.sh: ` _d ` -> D) are REAL*8; integers stay integers
    assert fortran_literal("1. _d -6") == 1e-6 and fortran_literal("1.D-6") == 1e-6 and fortran_literal("3") == 3
    assert fortran.real4(0.1) == fortran.real4("0.1") == float(np.float32(0.1))


def test_real4_namelist_variable_rounds_on_read():
    """w2_readparms.F declares `Real*4 facetEdgeLink(4, namList_NbFacets)` (W2_EXCH2_PARAMS.h, after preprocessing):
    a value read into it is stored as binary32 (0.1 -> 0.10000000149011612, the probe's READ gives the same); a
    REAL*8 variable keeps the float64 of the text; more than REAL4_READ_DIGITS significant digits is not ported."""
    from mitjax.tests.test_config import experiment
    g = experiment("global_ocean.90x40x15", "input")
    rds = namelists.file_readers(g.farm, g.toolchain, "data.exch2", g.cfg.size.env())
    nl = nlio.parse_namelist(" &W2_EXCH2_PARM01\n facetEdgeLink(1:2,1)= 0.1, 1.1,\n &\n")
    vs, _ = namelists.resolve_file("data.exch2", "planted", nl, rds)
    v = vs[("data.exch2", "w2_exch2_parm01", "facetedgelink")].value
    assert v[(1, 1)] == 0.10000000149011612 and v[(2, 1)] == 1.100000023841858
    long = nlio.parse_namelist(" &W2_EXCH2_PARM01\n facetEdgeLink(1,1)= 1.0000000596046448,\n &\n")
    with pytest.raises(NotImplementedError, match="significant digits"):
        namelists.resolve_file("data.exch2", "planted", long, rds)
    b = experiment("tutorial_barotropic_gyre", "input")
    rds = namelists.file_readers(b.farm, b.toolchain, "data", b.cfg.size.env())
    vs, _ = namelists.resolve_file("data", "planted", nlio.parse_namelist(" &PARM01\n viscAh=0.1,\n &\n"), rds)
    assert vs[("data", "parm01", "viscah")].value == 0.1                 # Real*8 (_RL): the float64 of the text
