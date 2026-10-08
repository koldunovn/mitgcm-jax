#!/usr/bin/env python3
"""Input of the Fortran-order global-sum reference (scripts/global_sum_ref/gsumref.F; plan Task 7b).

    make_input.py OUT_DIR        writes OUT_DIR/input.bin (refuses an existing file)

Cases: the tile shapes (nT, sNy, sNx) of the six M1 layouts plus a small all -0 case. Values: random signs,
magnitudes 10**U(-8, 8) (so the order of the additions matters), 15 % exact +0 and 10 % -0. Seeded: `cases()` returns
the same arrays in every environment (numpy's PCG64), which mitjax/tests/test_global_sum.py checks against the bytes
of input.bin. Stdlib + numpy only.
"""

import sys
from pathlib import Path

import numpy as np

SEED = 20261001
SHAPES = ((1, 62, 62), (2, 10, 20), (2, 1, 10), (4, 31, 31), (4, 20, 45), (36, 10, 10))


def _values(rng, shape):
    mag = 10.0 ** rng.uniform(-8.0, 8.0, shape)
    v = np.where(rng.random(shape) < 0.5, -mag, mag)
    u = rng.random(shape)
    v = np.where(u < 0.15, 0.0, v)
    return np.where((u >= 0.15) & (u < 0.25), -0.0, v)


def cases():
    """[(a, b)] with a, b float64 [nT, sNy, sNx] (tile, j, i)."""
    rng = np.random.default_rng(SEED)
    out = [(_values(rng, s), _values(rng, s)) for s in SHAPES]
    z = np.full((2, 3, 3), -0.0)
    out.append((z, z.copy()))
    return out


def encode(cs):
    """Big-endian stream: nCase; per case nT, sNy, sNx (int32), a and b in Fortran order a(sNx, sNy, nT)."""
    parts = [np.array([len(cs)], ">i4").tobytes()]
    for a, b in cs:
        parts.append(np.array(a.shape, ">i4").tobytes())
        parts.append(a.astype(">f8").tobytes())   # C order [t, j, i] == Fortran order a(i, j, t)
        parts.append(b.astype(">f8").tobytes())
    return b"".join(parts)


def decode_output(raw, cs):
    """output.bin -> [(partA[nT], sumA, partAB[nT], sumAB)] as float64."""
    v = np.frombuffer(raw, ">f8").astype(np.float64)
    out, p = [], 0
    for a, _ in cs:
        n = a.shape[0]
        out.append((v[p:p + n], v[p + n], v[p + n + 1:p + 2 * n + 1], v[p + 2 * n + 1]))
        p += 2 * n + 2
    if p != v.size:
        raise ValueError(f"output.bin has {v.size} values, the cases need {p}")
    return out


def main(argv):
    if len(argv) != 1:
        print(__doc__)
        return 2
    out = Path(argv[0]) / "input.bin"
    if out.exists():
        print(f"FAIL: {out} exists (never overwritten)")
        return 1
    out.write_bytes(encode(cases()))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
