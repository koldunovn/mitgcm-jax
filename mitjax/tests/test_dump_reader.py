"""mitjax/io/dump.py: round trip through jaxdump.F's record format, and planted format errors (plan Task 4)."""

import numpy as np
import pytest

from mitjax.io.dump import HDR, DumpFormatError, DumpSet, read_file, write_records


def _rec(stage, fld, tile, seq, values, it=7, kind="C", olx=2, oly=1, face=0, tbx=None, tby=0):
    return {"iter": it, "seq": seq, "stage": stage, "field": fld, "kind": kind, "tile": tile, "face": face,
            "tbx": (tile - 1) * (values.shape[2] - 2 * olx) if tbx is None else tbx, "tby": tby,
            "olx": olx, "oly": oly, "values": values}


def _fixture(tmp_path):
    """Two exch1 tiles side by side (sNx=4, sNy=3, OLx=2, OLy=1, nz=2), two stages, one repeated key."""
    rng = np.random.default_rng(0)
    vals = {t: rng.standard_normal((2, 5, 8)) for t in (1, 2)}
    vals[1][0, 1, 2] = -0.0  # signed zero must survive
    vals[2][1, 3, 5] = 1e-310  # subnormal
    for t in (1, 2):
        write_records(tmp_path / f"jd_0000000007_t{t:04d}.bin", [
            _rec("S00_begin", "theta", t, 1, vals[t]),
            _rec("S01_x", "etaN", t, 2, vals[t][:1] * 2),
            _rec("S00_begin", "theta", t, 3, vals[t] + 1),  # repeated key -> occurrence 1
        ])
    return vals


def test_round_trip_bits_layout_and_occurrences(tmp_path):
    vals = _fixture(tmp_path)
    ds = DumpSet(tmp_path)
    assert ds.keys() == [(7, "S00_begin", "theta"), (7, "S01_x", "etaN")]
    assert ds.stages(7) == ["S00_begin", "S01_x"]
    assert ds.repeated_keys() == [(7, "S00_begin", "theta")] and ds.n_occ((7, "S00_begin", "theta")) == 2
    f = ds.field(7, "S00_begin", "theta")
    assert f.shape == (2, 2, 5, 8)
    for t in (1, 2):  # bitwise, including -0 and the subnormal
        assert f[t - 1].tobytes() == vals[t].tobytes()
    assert np.signbit(f[0, 0, 1, 2]) and f[0, 0, 1, 2] == 0.0
    assert np.array_equal(ds.field(7, "S00_begin", "theta", occ=1), np.stack([vals[1] + 1, vals[2] + 1]))
    g = ds.global_field(7, "S00_begin", "theta")
    assert list(g) == [0] and g[0].shape == (2, 3, 8)  # face 0: Ny = 3, Nx = 2 tiles x 4
    assert np.array_equal(g[0][:, :, 0:4], vals[1][:, 1:4, 2:6])
    assert np.array_equal(g[0][:, :, 4:8], vals[2][:, 1:4, 2:6])


def test_scalar_records(tmp_path):
    for t in (1, 2):
        write_records(tmp_path / f"jd_0000000007_t{t:04d}.bin",
                      [_rec("C02", "numIters", t, 1, np.full((1, 5, 8), 37.0), kind="N")])
    assert DumpSet(tmp_path).scalar(7, "C02", "numIters") == 37.0


def _one_file(tmp_path):
    _fixture(tmp_path)
    return tmp_path / "jd_0000000007_t0001.bin"


def test_planted_truncated_record_fails(tmp_path):
    p = _one_file(tmp_path)
    data = p.read_bytes()
    p.write_bytes(data[:-8])  # last value of the last record missing
    with pytest.raises(DumpFormatError, match="truncated"):
        read_file(p, lazy=True)
    with pytest.raises(DumpFormatError, match="truncated"):
        DumpSet(tmp_path)
    p.write_bytes(data[:HDR.itemsize // 2])  # half a header
    with pytest.raises(DumpFormatError, match="truncated header"):
        read_file(p, lazy=True)


def test_planted_garbled_records_fail(tmp_path):
    p = _one_file(tmp_path)
    data = bytearray(p.read_bytes())
    rec_bytes = HDR.itemsize + 8 * 2 * 5 * 8
    for offset, value, msg in ((0, b"\x00\x00\x00\x01", "bad magic"),               # magic of record 1
                               (rec_bytes + 4, b"\x00\x00\x00\x01", "version"),     # version 1 = the ECCO format
                               (rec_bytes + 16, b"S0 1\xff", "garbled names|non-ASCII"),
                               (HDR.fields["nz"][1], b"\xff\xff\xff\xfb", "garbled dimensions")):
        bad = bytearray(data)
        bad[offset:offset + len(value)] = value
        p.write_bytes(bytes(bad))
        with pytest.raises(DumpFormatError, match=msg):
            read_file(p, lazy=True)


def test_record_in_the_wrong_file_fails(tmp_path):
    write_records(tmp_path / "jd_0000000007_t0001.bin",
                  [_rec("S00", "theta", 2, 1, np.zeros((1, 5, 8)))])  # tile 2 record in tile 1's file
    with pytest.raises(DumpFormatError, match="has iter 7 tile 2"):
        DumpSet(tmp_path)


def test_empty_directory_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        DumpSet(tmp_path)
