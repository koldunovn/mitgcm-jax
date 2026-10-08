"""mitjax/eesupp/exchange.py (plan Task 7b): the single-device Exchanger == the Fortran exchanges, bitwise, for every
routine M1 uses on every M1 layout (exch1: barotropic 1 tile, advect_xy 2, advect_xz sNy=1 < OLy, baroclinic 4,
optim 2x2; exch2: global_ocean 36 tiles), from the registered map files ($MJX_REFERENCE/exch_maps, sha256 checked).

One compact tier-1 test (budget); the adjoint identity, the multi-level check and the negative controls (a sign
flipped in a vector map, a wrong source, -0 lost) are in test_exchange_controls.py (tier 1x).
"""

import numpy as np

from mitjax.ad.cg2d_rule import exchange_reads_interior_only
from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp.exchange import Exchanger
from mitjax.tests.exch_gate import probe_dumps, probe_mismatches

# measured (probe run 27826873, lane A): halo points per layout; points EXCH_S3D_RL writes (the width-1 ring without
# its 4 corners, all of the ring on advect_xz); points the vector exchanges write (global_ocean: 6 of 5616 not written,
# tile 36's north-east corner halo beyond i = sNx+1 / j = sNy+1)
EXPECTED = {"tutorial_barotropic_gyre": (512, 248, 512), "advect_xy": (432, 120, 432), "advect_xz": (304, 52, 304),
            "tutorial_baroclinic_gyre": (1056, 496, 1056), "global_ocean.90x40x15": (5616, 1440, 5610),
            "tutorial_global_oce_optim": (1104, 520, 1104)}
VECTOR_MAPS = ("UVs", "UVn", "Ds", "UV3s")


def test_exchange_equals_fortran_probe():
    """Per layout: the registered maps load (sha256), Exchanger (jit, maps as arguments) reproduces every probe output
    and every signed-zero probe bitwise, the written-point counts are the measured ones (no sign flips, no copy from
    the other component), and EXCH_XY_RL reads interior points only and writes every halo point (the condition under
    which the cg2d rule's zero halo tangent is exact, ad/cg2d_rule.py)."""
    verdict = {}
    for exp in EM.M1_PROBES:
        maps = EM.load_maps(exp)
        ds, it = probe_dumps(exp)
        bad = probe_mismatches(Exchanger(maps), ds, it)
        assert len(bad) == 21 + 22, sorted(bad)
        verdict[exp] = sum(bad.values())
        assert verdict[exp] == 0, (exp, {k: v for k, v in bad.items() if v})
        n_halo, n_s3d, n_vec = EXPECTED[exp]
        for key, (written, other, neg, unwritten) in EM.summary(maps).items():
            want = n_s3d if key == "S3D" else n_vec if key.rsplit("_", 1)[0] in VECTOR_MAPS else n_halo
            assert (written, other, neg, written + unwritten) == (want, 0, 0, n_halo), (exp, key)
        assert exchange_reads_interior_only(maps, "XY"), exp
        src, comp, _ = maps.maps["XY"]
        assert np.all(comp[comp > 0] == 1)
    print(f"exchange probe gate (mismatching points per layout): {verdict}")
