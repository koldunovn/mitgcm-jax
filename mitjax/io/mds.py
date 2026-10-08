"""MDS binary files as master's pkg/mdsio reads model input (plan Task 7b; L-COPY-2 `mds`).

Reading (`read_field`) follows MDS_READ_FIELD (`pkg/mdsio/mdsio_read_field.F` @63cdc0b), the routine behind
READ_FLD_XY_RS/RL, READ_REC_* and the input-file readers, for a single-process build (no MPI, every M1 run):
- the file: the name as given if it exists, else `<name>.data` (a global file, :229-255), else one file per tile
  `<name>.<iG %03d>.<jG %03d>.data` with iG = bi + (myXGlobalLo-1)/sNx, jG = bj + (myYGlobalLo-1)/sNy (:450-453; a
  missing tile file stops the model, :468-481);
- the precision: `filePrec` = readBinaryPrec (32 or 64; precFloat32 = 32, precFloat64 = 64, EEPARAMS.h:62-65), the
  bytes big-endian (the oracle compiles with `-fconvert=big-endian`, reference/optfile_levante_gfortran);
- a global file (multi-CPU-IO path, `useSingleCpuIO` false: :380-441) is read row by row: record length sNx values,
  the row (j, k) of tile (bi,bj) at record `1 + tBx/sNx + (tBy + j-1)*global_nTx + (k-kLo + (irecord-1)*nNz)*
  global_nTx*ySize` with tBx = myXGlobalLo-1 + (bi-1)*sNx, tBy likewise, global_nTx = xSize/sNx and xSize = Nx,
  ySize = Ny (:120-121; under pkg/exch2 with useExch2ioLayOut exch2_global_Nx/Ny and the W2 tile offsets, :125-127,
  :403-425: the same formula for a single facet that fits the global array, as global_ocean.90x40x15 has) -- i.e. the
  global array `[irecord][k][Ny][Nx]` in Fortran order; the single-CPU-IO path (:262-366) reads the same values;
- a tiled file holds one record per irecord of sNx*sNy*nNz values (:464, :483-490);
- MDS_PASS_R4toRL/R8toRL (`mdsio_pass_r4torl.F:71`) copy the values into the interior of the model array: a real*4
  value becomes real*8 exactly; halos are not touched (the callers exchange them).
`read_meta` parses a `.meta` file as MDS_READ_META (`pkg/mdsio/mdsio_read_meta.F`): fixed column positions, and the
record count read as I10 when the line is at least 25 characters long and as I5 otherwise (:301-309; master writes
I10, `mdsio_write_meta.F:165`; c66g wrote I5) -- both widths are accepted.

`write_field` writes the same formats (global or tiled, 32 or 64 bit, meta file as `mdsio_write_meta.F`) for tests
and fixtures.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PREC = {32: ">f4", 64: ">f8"}         # precFloat32 = 32, precFloat64 = 64 (eesupp/inc/EEPARAMS.h:62-65)


@dataclass(frozen=True)
class MdsLayout:
    """Tiling of a single-process build: sNx, sNy, nSx, nSy from SIZE.h (nPx = nPy = 1); Nx = sNx*nSx, Ny = sNy*nSy.
    Tiles in the order bi + (bj-1)*nSx (eesupp/tiles.py)."""
    sNx: int
    sNy: int
    nSx: int
    nSy: int

    @property
    def Nx(self):
        return self.sNx * self.nSx

    @property
    def Ny(self):
        return self.sNy * self.nSy

    @property
    def nTiles(self):
        return self.nSx * self.nSy

    def tiles(self):
        """[(bi, bj)] 1-based, in tile order."""
        return [(bi, bj) for bj in range(1, self.nSy + 1) for bi in range(1, self.nSx + 1)]


def _check_prec(filePrec):
    if filePrec not in PREC:
        raise ValueError(f"filePrec {filePrec}: MDS_READ_FIELD knows precFloat32 (32) and precFloat64 (64) only")
    return np.dtype(PREC[filePrec])


def global_file(fName):
    """The global file MDS_READ_FIELD opens (mdsio_read_field.F:229-255), or None."""
    p = Path(fName)
    if p.exists():
        return p
    q = Path(str(fName) + ".data")
    return q if q.exists() else None


def tile_file(fName, bi, bj):
    """`<fName>.<iG %03d>.<jG %03d>.data` (mdsio_read_field.F:450-453; myXGlobalLo = myYGlobalLo = 1)."""
    return Path(f"{fName}.{bi:03d}.{bj:03d}.data")


@dataclass(frozen=True)
class E2ioLayout:
    """ADVECT lane (M2): MDS_READ_FIELD's exch2 global-file layout (mdsio_read_field.F:120-129, :403-425 with
    W2_useE2ioLayOut): xSize = exch2_global_Nx, ySize = exch2_global_Ny, and per tile in tile order (bi fastest,
    tN = W2_myTileList(bi,bj)) tBx = exch2_txGlobalo(tN)-1, tBy = exch2_tyGlobalo(tN)-1 and the (iGjLoc, jGjLoc)
    of :411-423 (fold / long line / default)."""
    xSize: int
    ySize: int
    tiles: tuple            # ((tBx, tBy, iGjLoc, jGjLoc), ...)

    @classmethod
    def from_w2(cls, w2, size):
        xSize, ySize = w2.exch2_global_Nx, w2.exch2_global_Ny                    # :125-126
        out = []
        for bj in range(1, size.nSy + 1):
            for bi in range(1, size.nSx + 1):
                tN = w2.W2_myTileList[bi, bj]                                    # :405
                tBx = w2.exch2_txGlobalo[tN] - 1                                 # :409
                tBy = w2.exch2_tyGlobalo[tN] - 1                                 # :410
                if w2.exch2_mydNx[tN] > xSize:                                   # :411-414 fold
                    g = (0, w2.exch2_mydNx[tN] // xSize)
                elif w2.exch2_tNy[tN] > ySize:                                   # :415-418 long line
                    g = (w2.exch2_mydNx[tN], 0)
                else:                                                            # :419-422 default
                    g = (0, 1)
                out.append((int(tBx), int(tBy)) + tuple(int(x) for x in g))
        return cls(int(xSize), int(ySize), tuple(out))


def read_field(fName, filePrec, layout, nNz=1, irecord=1, e2io=None):
    """Values MDS_READ_FIELD puts into the interior of the model array: float64 [nTiles, nNz, sNy, sNx] (levels kLo..
    kLo+nNz-1 of the field are the file's levels 1..nNz of record `irecord`, 1-based). `e2io`: the exch2 global-file
    layout (E2ioLayout; ADVECT lane, M2) when the build uses pkg/exch2 with W2_useE2ioLayOut."""
    L = layout
    dt = _check_prec(filePrec)
    if irecord < 1:
        raise ValueError(f"irecord {irecord} < 1 (MDS_READ_FIELD: Invalid value for irecord)")
    g = global_file(fName)
    out = np.empty((L.nTiles, nNz, L.sNy, L.sNx))
    if g is not None and e2io is not None:
        # mdsio_read_field.F:396-441 with useExch2ioLayOut: record irec of sNx values, irec = 1 + (tBx + (j-1)*
        # iGjLoc)/sNx + (tBy + (j-1)*jGjLoc)*global_nTx + (k-kLo + (irecord-1)*nNz)*global_nTx*ySize
        global_nTx = e2io.xSize // L.sNx                                         # :398
        rec = L.sNx * dt.itemsize
        with open(g, "rb") as fh:
            for t, (tBx, tBy, iGjLoc, jGjLoc) in enumerate(e2io.tiles):
                for k in range(nNz):
                    for j in range(1, L.sNy + 1):
                        irec = (1 + (tBx + (j-1)*iGjLoc)//L.sNx
                                + (tBy + (j-1)*jGjLoc)*global_nTx
                                + (k + (irecord-1)*nNz)*global_nTx*e2io.ySize)
                        fh.seek((irec - 1) * rec)
                        buf = fh.read(rec)
                        if len(buf) != rec:
                            raise ValueError(f"{g}: record {irec} beyond the end of the file")
                        out[t, k, j-1] = np.frombuffer(buf, dt).astype(np.float64)
        return out
    if g is not None:
        n = nNz * L.Ny * L.Nx
        with open(g, "rb") as fh:
            fh.seek((irecord - 1) * n * dt.itemsize)
            buf = fh.read(n * dt.itemsize)
        if len(buf) != n * dt.itemsize:
            raise ValueError(f"{g}: record {irecord} of {nNz} x {L.Ny} x {L.Nx} {dt} values beyond the end of the file")
        arr = np.frombuffer(buf, dt).astype(np.float64).reshape(nNz, L.Ny, L.Nx)
        for t, (bi, bj) in enumerate(L.tiles()):
            tBx, tBy = (bi - 1) * L.sNx, (bj - 1) * L.sNy
            out[t] = arr[:, tBy:tBy + L.sNy, tBx:tBx + L.sNx]
        return out
    n = nNz * L.sNy * L.sNx
    for t, (bi, bj) in enumerate(L.tiles()):
        p = tile_file(fName, bi, bj)
        if not p.exists():
            raise FileNotFoundError(f"MDS_READ_FIELD: neither {fName}, {fName}.data nor {p} exists")
        with open(p, "rb") as fh:
            fh.seek((irecord - 1) * n * dt.itemsize)
            buf = fh.read(n * dt.itemsize)
        if len(buf) != n * dt.itemsize:
            raise ValueError(f"{p}: record {irecord} beyond the end of the file")
        out[t] = np.frombuffer(buf, dt).astype(np.float64).reshape(nNz, L.sNy, L.sNx)
    return out


def into_tiles(values, OLx, OLy, base=None):
    """Interior values [nTiles, nNz, sNy, sNx] placed into tile arrays [nTiles, nNz, sNy+2OLy, sNx+2OLx]; halos from
    `base` (the array's previous values, as MDS_PASS_*toRL leaves them) or 0."""
    v = np.asarray(values)
    T, nz, sny, snx = v.shape
    out = np.zeros((T, nz, sny + 2 * OLy, snx + 2 * OLx)) if base is None else np.array(base, np.float64, copy=True)
    out[:, :, OLy:OLy + sny, OLx:OLx + snx] = v
    return out


# ---------------------------------------------------------------------------------------------------------------------
# meta files


def _int(text, fmt_width):
    """Fortran Iw read of a field: blanks ignored, an all-blank field is 0."""
    t = text[:fmt_width].strip()
    return int(t) if t else 0


def read_meta(path):
    """Parse a .meta file the way MDS_READ_META does (mdsio_read_meta.F:205-377 @63cdc0b): returns a dict with
    nDims, dimList [[n, first, last], ...], filePrec (32/64), nRecords, fileIter, timeList, misVal, nFlds, fldList,
    simulName, titleLine (only the keys present in the file)."""
    lines = Path(path).read_text().splitlines()
    out = {}
    it = iter(lines)
    for raw in it:
        line = raw.rstrip()
        iL = len(line)
        if iL >= 22 and line[0:14] == " simulation = ":
            out["simulName"] = line[17:iL - 4]                                     # lineBuf(18:iL-4)
        elif "nDims" not in out and iL >= 15 and line[0:9] == " nDims = ":
            out["nDims"] = _int(line[11:iL], 3)
        elif out.get("nDims", 0) >= 1 and iL >= 11 and line[0:11] == " dimList = ":
            dims = []
            for _ in range(out["nDims"]):
                d = next(it).rstrip()
                ii = len(d)
                if ii < 20:
                    vals = [d[1 + 6 * k:6 + 6 * k] for k in range(3)]               # (3(1X,I5))
                elif ii < 30:
                    vals = [d[10 + 6 * k:15 + 6 * k] for k in range(3)]             # (9X,3(1X,I5))
                else:
                    vals = [d[1 + 11 * k:11 + 11 * k] for k in range(3)]            # (3(1X,I10))
                dims.append([int(v.replace(",", " ").strip() or 0) for v in vals])
            out["dimList"] = dims
            next(it, None)                                                          # the closing line
        elif iL >= 20 and line[0:12] == " dataprec = ":
            out["filePrec"] = _prec_word(line[15:22], path)
        elif "filePrec" not in out and iL >= 18 and line[0:10] == " format = ":
            out["filePrec"] = _prec_word(line[13:20], path)
        elif "nRecords" not in out and iL >= 20 and line[0:12] == " nrecords = ":
            # mdsio_read_meta.F:303-307: I10 when the line is long enough (master), else I5 (older files)
            out["nRecords"] = _int(line[14:iL], 10) if iL >= 25 else _int(line[14:iL], 5)
        elif "fileIter" not in out and iL >= 31 and line[0:18] == " timeStepNumber = ":
            out["fileIter"] = _int(line[20:iL], 10)
        elif "timeList" not in out and iL >= 38 and line[0:16] == " timeInterval = ":
            n = (iL - 17 - 3) // 20
            body = line[17:iL - 3]
            out["timeList"] = [float(body[20 * k:20 * k + 20].replace("D", "E")) for k in range(n)]
        elif iL >= 8 and line[0:4] == " /* " and line[iL - 3:iL] == " */":
            out["titleLine"] = line[4:iL - 3]
        elif "misVal" not in out and iL >= 40 and line[0:16] == " missingValue = ":
            out["misVal"] = float(line[18:iL].split()[0].replace("D", "E"))
        elif "nFlds" not in out and iL >= 16 and line[0:9] == " nFlds = ":
            out["nFlds"] = _int(line[11:iL], 4)
        elif out.get("nFlds", 0) >= 1 and iL >= 11 and line[0:11] == " fldList = ":
            names = []
            while len(names) < out["nFlds"]:
                row = next(it)
                names += [row[2 + 11 * k:10 + 11 * k].strip() for k in range(min(20, out["nFlds"] - len(names)))]
            out["fldList"] = names
            next(it, None)
    return out


def _prec_word(word, path):
    if word == "float32":
        return 32
    if word == "float64":
        return 64
    raise ValueError(f"{path}: MDS_READ_META: invalid dataprec {word!r}")


def write_meta(path, dims, filePrec, nrecords, timeStepNumber=None, nrecords_width=10):
    """A .meta file in mdsio_write_meta.F's format (master: nrecords as I10, :165; nrecords_width=5 writes the older
    I5 form). dims: [[n, first, last], ...] (x first)."""
    lines = [f" nDims = [ {len(dims):3d} ];", " dimList = ["]
    small = max(max(d) for d in dims) < 10000
    for j, d in enumerate(dims):
        w = 5 if small else 10
        end = "," if j < len(dims) - 1 else ""
        lines.append(" " + ",".join(f"{v:{w}d}" for v in d) + end)
    lines.append(" ];")
    lines.append(f" dataprec = [ 'float{filePrec}' ];")
    lines.append(f" nrecords = [ {nrecords:{nrecords_width}d} ];")
    if timeStepNumber is not None:
        lines.append(f" timeStepNumber = [ {timeStepNumber:10d} ];")
    Path(path).write_text("\n".join(lines) + "\n")


def write_field(fName, values, filePrec, layout, tiled=False, nrecords_width=10):
    """Write records of a field: values float64 [nRec, nNz, Ny, Nx] (global array order) as `<fName>.data` + `.meta`
    (tiled=False) or one `<fName>.III.JJJ.data` + `.meta` per tile. Refuses existing files."""
    L = layout
    dt = _check_prec(filePrec)
    v = np.asarray(values, np.float64)
    nrec, nz, ny, nx = v.shape
    if (ny, nx) != (L.Ny, L.Nx):
        raise ValueError(f"values {v.shape} do not match Ny, Nx = {L.Ny}, {L.Nx}")
    targets = []
    if not tiled:
        targets.append((Path(f"{fName}.data"), v, [[L.Nx, 1, L.Nx], [L.Ny, 1, L.Ny]]))
    else:
        for bi, bj in L.tiles():
            x0, y0 = (bi - 1) * L.sNx, (bj - 1) * L.sNy
            targets.append((tile_file(fName, bi, bj), v[:, :, y0:y0 + L.sNy, x0:x0 + L.sNx],
                            [[L.Nx, x0 + 1, x0 + L.sNx], [L.Ny, y0 + 1, y0 + L.sNy]]))
    for p, arr, dims in targets:
        meta = p.with_suffix(".meta")
        for q in (p, meta):
            if q.exists():
                raise FileExistsError(f"{q} exists (never overwritten)")
        if nz > 1:
            dims = dims + [[nz, 1, nz]]
        with open(p, "xb") as fh:
            fh.write(arr.astype(dt).tobytes())
        write_meta(meta, dims, filePrec, nrec, nrecords_width=nrecords_width)
