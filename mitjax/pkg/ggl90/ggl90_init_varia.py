"""GGL90_INIT_VARIA: pkg/ggl90/ggl90_init_varia.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_XY_RL
from mitjax.farray import loops_kji, loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.pkg.ggl90.ggl90_h import GGL90eps, declare


def _blank(s):
    return str(s).strip() == ""


def ggl90_init_varia(ggl, *, cfg, grid, nIter0, pickupSuff, ex, rw):
    """GGL90_INIT_VARIA( myThid )   @63cdc0b pkg/ggl90/ggl90_init_varia.F:6-156

    C     | SUBROUTINE GGL90_INIT_VARIA
    C     | o Routine to initialize GGL90 parameters and variables.

    Returns `ggl` with GGL90viscArU, GGL90viscArV, GGL90diffKr = 0 and GGL90TKE = GGL90eps*maskC (useIDEMIX) or
    GGL90TKEmin*maskC on every point (:38-61); under ALLOW_GGL90_IDEMIX, IDEMIX_E, IDEMIX_F_B, IDEMIX_F_S = 0
    (:68-87), then with useIDEMIX the tidal and wind forcing read (READ_REC_XY_RL = MDS_READ_FIELD with
    readBinaryPrec, record 1, pkg/rw/read_rec.F:78-131), exchanged (_EXCH_XY_RL) and clipped/scaled (:91-128).
    The k loops write constants per level (independent; vectorised). Raise: a restart (nIter0 /= 0 or
    pickupSuff /= ' ': GGL90_READ_PICKUP, :131-132) and GGL90TKEFile (:135-150) are not ported."""
    sz = cfg.size
    Nr = sz.Nr
    fj, fi = (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx)
    k, j, i = loops_kji((1, Nr), fj, fi)
    out = {}
    for n in ("GGL90viscArU", "GGL90viscArV", "GGL90diffKr"):           # :45-47
        out[n] = declare(n, sz).at[i, j, k].set(0.0)
    if ggl.useIDEMIX:                                                   # :48-53
        out["GGL90TKE"] = declare("GGL90TKE", sz).at[i, j, k].set(GGL90eps*grid.maskC[i, j, k])
    else:
        out["GGL90TKE"] = declare("GGL90TKE", sz).at[i, j, k].set(ggl.GGL90TKEmin*grid.maskC[i, j, k])
    if cfg.cpp.flag("ALLOW_GGL90_IDEMIX", "GGL90_OPTIONS.h"):           # :63-129
        out["IDEMIX_E"] = declare("IDEMIX_E", sz).at[i, j, k].set(0.0)                  # :74
        j, i = loop_j(*fj), loop_i(*fi)
        F_b = declare("IDEMIX_F_B", sz).at[i, j].set(0.0)               # :81
        F_s = declare("IDEMIX_F_S", sz).at[i, j].set(0.0)               # :82
        if ggl.useIDEMIX and not _blank(ggl.IDEMIX_tidal_file):         # :91-109
            F_b = rw.mds_read_field(ggl.IDEMIX_tidal_file, rw.readBinaryPrec, 1, F_b, 1)  # :92 READ_REC_XY_RL
            F_b = EXCH_XY_RL(F_b, ex=ex)                                # :93
            F_b = F_b.at[i, j].set(-MAX(0.0,                            # :99-100
                                        MIN(1.0, F_b[i, j], p="a"), p="b"))
            F_b = F_b.at[i, j].set(ggl.IDEMIX_frac_F_b*                 # :102-104
                                   F_b[i, j]/1024.0)                    # 1024. _d 0
        if ggl.useIDEMIX and not _blank(ggl.IDEMIX_wind_file):          # :111-128
            F_s = rw.mds_read_field(ggl.IDEMIX_wind_file, rw.readBinaryPrec, 1, F_s, 1)   # :112 READ_REC_XY_RL
            F_s = EXCH_XY_RL(F_s, ex=ex)                                # :113
            F_s = F_s.at[i, j].set(MAX(0.0,                             # :118-119
                                       MIN(1.0, F_s[i, j], p="a"), p="b"))
            F_s = F_s.at[i, j].set(ggl.IDEMIX_frac_F_s*                 # :121-123  1024. (REAL*4, exact)
                                   F_s[i, j]/1024.)
        out["IDEMIX_F_B"], out["IDEMIX_F_S"] = F_b, F_s
    if nIter0 != 0 or not _blank(pickupSuff):                           # :131-132
        raise NotImplementedError("GGL90_INIT_VARIA: GGL90_READ_PICKUP (a restart) is not ported")
    if not _blank(ggl.GGL90TKEFile):                                    # :135-150
        raise NotImplementedError("GGL90_INIT_VARIA: GGL90TKEFile (READ_FLD_XYZ_RL) is not ported")
    return ggl.replace(**out)
