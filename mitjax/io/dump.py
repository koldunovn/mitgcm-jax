"""Reader (and test writer) for jaxdump per-substep dumps of the Fortran oracle (reference/jaxdump/jaxdump.F).

Copied from the ECCO port's io/dump.py and generalised (plan Task 4): any tile layout (exch1 and exch2), any number
of faces (the ECCO reader assembled the LLC90 facets with a fixed FACET_SHAPE), repeated keys kept per occurrence,
record order from the record sequence number, and a strict validation of every header.

Files: <dir>/jd_<iter %010d>_t<tile %04d>.bin, a stream of big-endian records (jaxdump.F, version 2)
    int32 magic=1245990224, version=2, iter, seq | char32 stage | char32 field | char4 kind |
    int32 nz, sNx, sNy, OLx, OLy, tile, face, tBasex, tBasey | float64 values[nz][sNy+2*OLy][sNx+2*OLx]
Values include halos. A value at array index (k, j, i) of a record sits at global point
(face, jG = tBasey + j - OLy + 1, iG = tBasex + i - OLx + 1) in 1-based model indices; halo points fall outside the
tile's interior (and may fall outside the face). exch1 builds write face 0 and tBasex/y = the tile's offsets in the
one global Nx x Ny domain; exch2 builds the W2 facet and offsets.

Repeated keys: a key (iter, stage, field) written more than once into the same tile file (a stage inside a loop
without a pass suffix, or the forward runs of grdchk, which run FORWARD_STEP again with the same iterations) is kept
as occurrences 0, 1, ... in file order; the ECCO reader silently kept only the last one.

Values are read lazily (headers first, in parallel threads) and kept once loaded (`Record.data`); a DumpSet is meant
to be built once per directory and reused.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 1245990224
VERSION = 2
HDR = np.dtype([("magic", ">i4"), ("version", ">i4"), ("iter", ">i4"), ("seq", ">i4"),
                ("stage", "S32"), ("field", "S32"), ("kind", "S4"),
                ("nz", ">i4"), ("snx", ">i4"), ("sny", ">i4"), ("olx", ">i4"), ("oly", ">i4"),
                ("tile", ">i4"), ("face", ">i4"), ("tbx", ">i4"), ("tby", ">i4")])
KINDS = ("C", "W", "S", "Z", "N", "V")
_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_FILE = re.compile(r"jd_(\d{10})_t(\d{4})\.bin")


class DumpFormatError(ValueError):
    """A dump file that does not follow the jaxdump.F record format (truncated, garbled, wrong version)."""


@dataclass
class Record:
    iter: int
    seq: int
    stage: str
    field: str
    kind: str
    nz: int
    snx: int
    sny: int
    olx: int
    oly: int
    tile: int
    face: int
    tbx: int
    tby: int
    path: str
    offset: int
    occ: int = 0
    _data: np.ndarray = field(default=None, repr=False)

    @property
    def shape(self):
        return (self.nz, self.sny + 2 * self.oly, self.snx + 2 * self.olx)

    @property
    def data(self):
        """(nz, sNy+2*OLy, sNx+2*OLx) float64, halos included; read on first access, then kept."""
        if self._data is None:
            n = int(np.prod(self.shape))
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                buf = fh.read(8 * n)
            if len(buf) != 8 * n:
                raise DumpFormatError(f"{self.path}: record at byte {self.offset} truncated")
            self._data = np.frombuffer(buf, ">f8", count=n).reshape(self.shape).astype(np.float64)
        return self._data

    @property
    def interior(self):
        d = self.data
        return d[:, self.oly:d.shape[1] - self.oly, self.olx:d.shape[2] - self.olx]


def _check_header(h, path, pos):
    where = f"{path}: record at byte {pos}"
    if int(h["magic"]) != MAGIC:
        raise DumpFormatError(f"{where}: bad magic {int(h['magic'])} (garbled or not a jaxdump file)")
    if int(h["version"]) != VERSION:
        raise DumpFormatError(f"{where}: version {int(h['version'])}, this reader reads version {VERSION}")
    try:
        names = [h[k].decode("ascii").strip() for k in ("stage", "field", "kind")]
    except UnicodeDecodeError as e:
        raise DumpFormatError(f"{where}: non-ASCII name ({e})") from None
    stage, fld, kind = names
    if not (_NAME.match(stage) and _NAME.match(fld)) or kind not in KINDS:
        raise DumpFormatError(f"{where}: garbled names stage={stage!r} field={fld!r} kind={kind!r}")
    nz, snx, sny, olx, oly, tile = (int(h[k]) for k in ("nz", "snx", "sny", "olx", "oly", "tile"))
    if not (1 <= nz <= 100000 and 1 <= snx <= 100000 and 1 <= sny <= 100000 and 0 <= olx <= 100 and 0 <= oly <= 100
            and tile >= 1 and int(h["face"]) >= 0):
        raise DumpFormatError(f"{where}: garbled dimensions nz={nz} sNx={snx} sNy={sny} OLx={olx} OLy={oly} "
                              f"tile={tile} face={int(h['face'])}")
    return stage, fld, kind


def read_file(path, lazy=False):
    """All records of one dump file, validated. lazy=True reads only the headers (values on first access).
    Raises DumpFormatError on a truncated or garbled record."""
    path = str(path)
    size = Path(path).stat().st_size
    out, pos, seen = [], 0, {}
    with open(path, "rb") as fh:
        while pos < size:
            if size - pos < HDR.itemsize:
                raise DumpFormatError(f"{path}: truncated header at byte {pos} ({size - pos} bytes left)")
            fh.seek(pos)
            h = np.frombuffer(fh.read(HDR.itemsize), HDR, count=1)[0]
            stage, fld, kind = _check_header(h, path, pos)
            pos += HDR.itemsize
            r = Record(int(h["iter"]), int(h["seq"]), stage, fld, kind, *(int(h[k]) for k in (
                "nz", "snx", "sny", "olx", "oly", "tile", "face", "tbx", "tby")), path=path, offset=pos)
            nbytes = 8 * int(np.prod(r.shape))
            if pos + nbytes > size:
                raise DumpFormatError(f"{path}: record {stage}/{fld} at byte {pos - HDR.itemsize} truncated "
                                      f"({size - pos} of {nbytes} value bytes)")
            key = (r.iter, stage, fld)
            r.occ = seen.get(key, 0)
            seen[key] = r.occ + 1
            if not lazy:
                r.data  # noqa: B018 (load now)
            out.append(r)
            pos += nbytes
    return out


def write_records(path, records):
    """Append records to a dump file in jaxdump.F's format (tests and synthetic fixtures). records: iterable of
    dicts with iter, seq, stage, field, kind, tile, face, tbx, tby, olx, oly and values (nz, sNy+2*OLy, sNx+2*OLx)."""
    with open(path, "ab") as fh:
        for r in records:
            v = np.asarray(r["values"], dtype=np.float64)
            nz, ny, nx = v.shape
            h = np.zeros(1, HDR)
            h["magic"], h["version"], h["iter"], h["seq"] = MAGIC, r.get("version", VERSION), r["iter"], r["seq"]
            h["stage"], h["field"], h["kind"] = (r["stage"].ljust(32).encode(), r["field"].ljust(32).encode(),
                                                 r["kind"].ljust(4).encode())
            h["nz"], h["snx"], h["sny"], h["olx"], h["oly"] = nz, nx - 2 * r["olx"], ny - 2 * r["oly"], r["olx"], r["oly"]
            h["tile"], h["face"], h["tbx"], h["tby"] = r["tile"], r["face"], r["tbx"], r["tby"]
            fh.write(h.tobytes())
            fh.write(v.astype(">f8").tobytes())


class DumpSet:
    """All records under a dump directory.

    index[(iter, stage, field)] = [ {tile: Record} for occurrence 0, 1, ... ]
    order: the keys in call order (smallest record sequence number of occurrence 0 over all tiles), which also places
    tile-scoped stages (written inside a bi,bj loop) where the Fortran called them first."""

    def __init__(self, directory, threads=16):
        self.dir = Path(directory)
        files = sorted(f for f in self.dir.glob("jd_*_t*.bin") if _FILE.fullmatch(f.name))
        if not files:
            raise FileNotFoundError(f"no jaxdump files in {self.dir}")
        with ThreadPoolExecutor(max(1, min(threads, len(files)))) as pool:
            headers = list(pool.map(lambda f: read_file(f, lazy=True), files))
        self.index, first_seq, self.tiles_info = {}, {}, {}
        for f, recs in zip(files, headers):
            m = _FILE.fullmatch(f.name)
            for r in recs:
                if r.iter != int(m.group(1)) or r.tile != int(m.group(2)):
                    raise DumpFormatError(f"{f}: record {r.stage}/{r.field} has iter {r.iter} tile {r.tile}")
                key = (r.iter, r.stage, r.field)
                occs = self.index.setdefault(key, [])
                while len(occs) <= r.occ:
                    occs.append({})
                occs[r.occ][r.tile] = r
                if r.occ == 0:
                    first_seq[key] = min(first_seq.get(key, r.seq), r.seq)
                info = (r.face, r.tbx, r.tby, r.snx, r.sny, r.olx, r.oly)
                if self.tiles_info.setdefault(r.tile, info) != info:
                    raise DumpFormatError(f"{f}: tile {r.tile} layout {info} != {self.tiles_info[r.tile]}")
        self.order = sorted(self.index, key=lambda k: (k[0], first_seq[k], k[1], k[2]))

    # ---- keys -------------------------------------------------------------------------------------------------
    def keys(self, it=None):
        return [k for k in self.order if it is None or k[0] == it]

    def iterations(self):
        return sorted({k[0] for k in self.order})

    def stages(self, it):
        """Stage names of iteration `it` in call order."""
        out = []
        for k in self.keys(it):
            if k[1] not in out:
                out.append(k[1])
        return out

    def n_occ(self, key):
        return len(self.index[key])

    def repeated_keys(self):
        return [k for k in self.order if len(self.index[k]) > 1]

    def tiles(self, it, stage, fld, occ=0):
        """{tile: Record} of one key and occurrence."""
        return self.index[(it, stage, fld)][occ]

    # ---- values -----------------------------------------------------------------------------------------------
    def field(self, it, stage, fld, occ=0):
        """Array [tile, k, j, i] with halos, tiles in increasing tile number (the W2 numbering under exch2, the
        bi + (bj-1)*nSx order under exch1)."""
        recs = self.tiles(it, stage, fld, occ)
        return np.stack([recs[t].data for t in sorted(recs)])

    def scalar(self, it, stage, name, occ=0):
        """Value of a kind-N record (JAXDUMP_SCALAR); it must be the same at every point of every tile."""
        recs = self.tiles(it, stage, name, occ)
        vals = np.concatenate([r.data.ravel() for r in recs.values()])
        if any(r.kind != "N" for r in recs.values()):
            raise ValueError(f"{stage}/{name} is not a scalar record")
        if not (np.all(vals == vals[0]) or np.all(np.isnan(vals))):
            raise ValueError(f"{stage}/{name}: scalar record is not constant")
        return float(vals[0])

    def face_shapes(self):
        """{face: (NyF, NxF)} from the tile offsets and sizes (all tiles that wrote any record)."""
        out = {}
        for face, tbx, tby, snx, sny, _, _ in self.tiles_info.values():
            ny, nx = out.get(face, (0, 0))
            out[face] = (max(ny, tby + sny), max(nx, tbx + snx))
        return out

    def global_field(self, it, stage, fld, occ=0):
        """{face: array (nz, NyF, NxF)} of interior values; points of no dumped tile stay NaN."""
        recs = self.tiles(it, stage, fld, occ)
        nz = next(iter(recs.values())).nz
        out = {f: np.full((nz, *s), np.nan) for f, s in self.face_shapes().items()}
        for r in recs.values():
            v = r.interior
            out[r.face][:, r.tby:r.tby + v.shape[1], r.tbx:r.tbx + v.shape[2]] = v
        return out


def probe_decode(values, cbase, nxp, nyp):
    """Decode exchange-probe values (jaxdump.F JAXDUMP_PROBE_FILL): value = sign * (comp*cbase + idx),
    idx = ((tile-1)*nyp + ja)*nxp + ia + 1 with ja = j+OLy-1, ia = i+OLx-1 the 0-based padded indices of the SOURCE
    point, nxp = sNx+2*OLx, nyp = sNy+2*OLy, cbase from the record xCbase. Returns a dict of int arrays comp, tile,
    ja, ia and sign (+1/-1, from the sign bit); raises ValueError on a value that is not a probe code."""
    v = np.asarray(values, dtype=np.float64)
    a = np.abs(v)
    comp = np.floor(a / cbase)
    idx = a - comp * cbase - 1
    if not (np.all(np.isfinite(v)) and np.all(idx == np.round(idx)) and np.all((comp == 1) | (comp == 2))
            and np.all(idx >= 0)):
        raise ValueError("values are not exchange-probe codes")
    idx = idx.astype(np.int64)
    return {"comp": comp.astype(np.int64), "tile": idx // (nxp * nyp) + 1, "ja": (idx // nxp) % nyp,
            "ia": idx % nxp, "sign": np.where(np.signbit(v), -1, 1)}
