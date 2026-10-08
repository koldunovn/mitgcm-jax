"""MDS_FACEF_READ_RS (pkg/mdsio/mdsio_facef_read.F @63cdc0b, the ALLOW_EXCH2 branch): read one record of a facet
file (a `tile00N.mitgrid` or `<horizGridFile>.face00N.bin` grid file) into the points i = 1..sNx+1, j = 1..sNy+1
of one tile.

    @63cdc0b pkg/mdsio/mdsio_facef_read.F:9-156 (ALLOW_EXCH2: :70-124)

The file is a direct-access file of records of dNx+1 values (MDS_RECLEN( fPrec, dNx+1 )); field `irec` is the
dNy+1 records jBase+1 .. jBase+dNy+1, jBase = (irec-1)*(dNy+1) (:79-80); for the tile with offsets (tBx, tBy) the
loop reads records jj+jBase, jj = 1+tBy .. sNy+1+tBy, and array(i, j) = ioBuf(i+tBx) for i = 1..sNx+1 (:82-104).
Byte order: big-endian (the oracle builds with -fconvert=big-endian; _BYTESWAPIO undefined). Values are stored as
read (Real*8 -> _RS Real*8; Real*4 -> Real*8 is exact). Host-side numpy (file reading, never traced).
"""

import numpy as np

PRECFLOAT32 = 32         # EEPARAMS.h:63
PRECFLOAT64 = 64         # EEPARAMS.h:65


def mds_facef_read_rs(path, fPrec, irec, tile_values, *, sNx, sNy, OLx, OLy, dNx, dNy, tBx, tBy):
    """MDS_FACEF_READ_RS( fName, fPrec, irec, array, bi, bj, myThid ) for one tile: returns a copy of
    `tile_values` ([ny, nx] storage of array(:,:,bi,bj), padded indices) with points (1..sNx+1, 1..sNy+1) replaced."""
    if fPrec == PRECFLOAT32:
        dt, size = ">f4", 4
    elif fPrec == PRECFLOAT64:
        dt, size = ">f8", 8
    else:
        raise ValueError(f" MDS_FACEF_READ_RS:{fPrec:8d} = illegal value for fPrec; "
                         "ABNORMAL END: S/R MDS_FACEF_READ_RS")                # :108-112
    reclen = (dNx + 1) * size
    out = np.array(tile_values, dtype=np.float64, copy=True)
    jBase = (irec - 1) * (dNy + 1)                                            # :80
    with open(path, "rb") as fh:
        j = 0                                                                 # :79
        for jj in range(1 + tBy, sNy + 1 + tBy + 1):                          # :83 / :95
            fh.seek((jj + jBase - 1) * reclen)                                # READ(dUnit,rec=jj+jBase)
            raw = fh.read(reclen)
            if len(raw) != reclen:
                raise EOFError(f"{path}: record {jj + jBase} beyond the end of the file")
            ioBuf = np.frombuffer(raw, dt).astype(np.float64)                 # ioBuf(1:dNx+1)
            j = j + 1
            for i in range(1, sNx + 1 + 1):                                   # :89-91 / :101-103
                out[j - 1 + OLy, i - 1 + OLx] = ioBuf[i + tBx - 1]            # array(i,j,bi,bj) = ioBuf(i+tBx)
    return out
