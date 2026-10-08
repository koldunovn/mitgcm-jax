"""MDS_WRITE_FIELD: pkg/mdsio/mdsio_write_field.F @63cdc0b, and MDS_WR_METAFILES: pkg/mdsio/mdsio_wr_metafiles.F
(host-side binary output of one process; the binaries are built with -fconvert=big-endian,
reference/optfile_levante_gfortran:80, so records are big-endian)."""

import os

import numpy as np

from mitjax.pkg.mdsio.mdsio_write_meta import mds_write_meta, oneRL, precFloat32, precFloat64

blank8c = " " * 8                            # mdsio_write_field.F  DATA blank8c / '        ' /


class MdsContext:
    """What MDS_WRITE_FIELD reads besides its arguments: SIZE.h, EEPARAMS.h (useSingleCpuIO; myXGlobalLo =
    myYGlobalLo = 1 for one process, ini_procs.F:75-76), PARAMS.h mdsioLocalDir and the_run_name (blank,
    set_defaults.F:398), the run directory files are written to (the model runs there), and the build's ALLOW_EXCH2
    with its W2 common blocks `w2` (W2_useE2ioLayOut, GO lane; an ALLOW_EXCH2 context without them raises)."""

    def __init__(self, rundir, size, *, exch2=False, useSingleCpuIO=False, mdsioLocalDir=" ", the_run_name=" ",
                 useCal=False, w2=None):
        self.rundir, self.size = str(rundir), size
        self.exch2, self.useSingleCpuIO = exch2, useSingleCpuIO
        # GO lane: the W2 common blocks (pkg/exch2/w2_exch2_h.W2Common: W2_useE2ioLayOut, exch2_global_Nx/Ny,
        # W2_myTileList, exch2_txGlobalo/tyGlobalo, exch2_mydNx, exch2_tNy) of an ALLOW_EXCH2 build
        self.w2 = w2
        self.mdsioLocalDir, self.the_run_name, self.useCal = mdsioLocalDir, the_run_name, useCal
        self.myXGlobalLo = self.myYGlobalLo = 1

    def path(self, name):
        return os.path.join(self.rundir, name)


def _dtype(filePrec):
    if filePrec == precFloat32:
        return ">f4"
    if filePrec == precFloat64:
        return ">f8"
    raise ValueError(" MDS_WRITE_FIELD: illegal value for filePrec\nABNORMAL END: S/R MDS_WRITE_FIELD")


def mds_write_field(fName, filePrec, globalFile, useCurrentDir, arrType, kSize, kLo, kHi, fldRL, jrecord, myIter, *,
                    mds):
    """MDS_WRITE_FIELD( fName, filePrec, globalFile, useCurrentDir, arrType, kSize, kLo, kHi, fldRL, fldRS,
    jrecord, myIter, myThid )   @63cdc0b pkg/mdsio/mdsio_write_field.F:6-605

    C Arguments:
    C fName     (string)  :: base name for file to write
    C filePrec  (integer) :: number of bits per word in file (32 or 64)
    C globalFile (logical):: selects between writing a global or tiled file
    C irecord   (integer) :: record number to write
    C  irecord=|jrecord| is the record number to be written and must be >= 1.
    C  NOTE: The file is created only if irecord = 1. The meta file is written only if jrecord > 0.

    fldRL: an FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,kSize,nSx,nSy) (a 2-D field is kSize = 1). Ported: one process,
    the tiled (globalFile = .FALSE.) and the global-file branches (:437-559); the record of a tile file holds the tile
    interior, levels kLo..kHi, in Fortran order (MDS_PASS_R8toRL with oLi = oLj = 0 fills the buffer from the
    interior, :392); the useSingleCpuIO branch (:269-364: GATHER_2D into one global record per level; lane B,
    `_single_cpu_io`); the exch2 I/O layout (W2_useE2ioLayOut, :146-151, :447-469) with mds.w2 (GO lane). Files are opened
    status 'unknown' (MDSIO_OPTIONS.h:27, :31): a record is written in place; an existing file is not truncated."""
    sz = mds.size
    sNx, sNy, OLx, OLy, nSx, nSy = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.nSx, sz.nSy
    Nx, Ny = sNx * nSx * sz.nPx, sNy * nSy * sz.nPy
    useExch2ioLayOut = False                                              # :145
    if mds.exch2:                                                         # :146-151 (#ifdef ALLOW_EXCH2; GO lane)
        if mds.w2 is None:
            raise NotImplementedError("MDS_WRITE_FIELD: an ALLOW_EXCH2 context needs the W2 common blocks (w2=)")
        if mds.w2.W2_useE2ioLayOut:
            useExch2ioLayOut = True
    if arrType != "RL":
        raise NotImplementedError("MDS_WRITE_FIELD: arrType 'RS' is not ported here")
    xSize, ySize = Nx, Ny                                                 # :142-143
    if useExch2ioLayOut:
        xSize, ySize = mds.w2.exch2_global_Nx, mds.w2.exch2_global_Ny     # :147-148
    iGjLoc, jGjLoc = 0, 1                                                 # :154-155
    IL = len(fName.rstrip())
    pIL = len(mds.mdsioLocalDir.rstrip())
    nNz = 1 + kHi - kLo                                                   # :162
    irecord = abs(jrecord)                                                # :163
    writeMetaF = jrecord > 0                                              # :164
    if irecord < 1:                                                       # :189-203
        raise ValueError(f" MDS_WRITE_FIELD: file=\"{fName[:IL]}\" , iter={myIter:10d}\n"
                         " MDS_WRITE_FIELD: invalid value for irecord\nABNORMAL END: S/R MDS_WRITE_FIELD")
    if kLo < 1 or kHi > kSize:                                            # :205-220
        raise ValueError(" MDS_WRITE_FIELD: invalid sub-set of levels\nABNORMAL END: S/R MDS_WRITE_FIELD")
    pfName = fName[:IL] if (useCurrentDir or pIL == 0) else mds.mdsioLocalDir[:pIL] + fName[:IL]   # :246-251
    pIL = len(pfName)
    if mds.useSingleCpuIO:                                                # :269-364 (lane B, Task 25)
        _single_cpu_io(fName[:IL], filePrec, kSize, kLo, kHi, fldRL, irecord, nNz, xSize, ySize, useExch2ioLayOut,
                       mds=mds)
    else:
        _per_tile_io(fName, IL, pfName, filePrec, globalFile, kSize, kLo, kHi, fldRL, irecord, nNz, xSize, ySize,
                     useExch2ioLayOut, iGjLoc, jGjLoc, writeMetaF, myIter, mds=mds)
    if (globalFile or mds.useSingleCpuIO) and writeMetaF:                 # :575-599 (also if useSingleCpuIO)
        dataFName = fName[:IL] + ".data"
        dimList = [[xSize, 1, xSize], [ySize, 1, ySize], [nNz, 1, nNz]]
        nDims = 2 if nNz == 1 else 3
        mds_write_meta(mds.path(fName[:IL] + ".meta"), mds.path(dataFName), mds.the_run_name, " ", filePrec,
                       nDims, dimList, (0, 1), 0, [blank8c], 0, [0.0], oneRL, irecord, myIter, useCal=mds.useCal)


def _single_cpu_io(fNameIL, filePrec, kSize, kLo, kHi, fldRL, irecord, nNz, xSize, ySize, useExch2ioLayOut, *,
                   mds):
    """The useSingleCpuIO branch of MDS_WRITE_FIELD (mdsio_write_field.F:269-364; lane B, Task 25): the global file
    `fName(1:IL)//'.data'` (no mdsioLocalDir prefix, :274), records of xSize*ySize values, opened status _NEW_STATUS =
    'unknown' (SAFE_IO undefined, MDSIO_OPTIONS.h:24-28) for irecord = 1, else _OLD_STATUS = 'old' (no ALLOW_AUTODIFF,
    :30-34: the file must exist); per level k = kLo..kHi MDS_PASS_R8toRL / R4toRL of the tile interiors and
    GATHER_2D_R8 / R4 into the global buffer (eesupp/src/gather_2d_rx.template:56-133: the exch2 layout zeroes the
    buffer at k = kLo (`zeroBuff`) and places tile tN at exch2_tx/tyGlobalo with the iGjLoc / jGjLoc row steps; without
    it tile (bi,bj) at (myXGlobalLo-1+(bi-1)*sNx, myYGlobalLo-1+(bj-1)*sNy)), written as record
    irec = 1 + k-kLo + (irecord-1)*nNz (:337-350). The global buffer is static storage (zero before its first use);
    a REAL*4 file holds the values rounded to REAL*4 by MDS_PASS_R4toRL."""
    sz = mds.size
    sNx, sNy, OLx, OLy, nSx, nSy = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.nSx, sz.nSy
    dt = _dtype(filePrec)
    path = mds.path(fNameIL + ".data")                                    # :274 dataFName = fName(1:IL)//'.data'
    if irecord == 1:                                                      # :276-278  status=_NEW_STATUS ('unknown')
        fh = _open(path)
    else:                                                                 # :279-281  status=_OLD_STATUS ('old')
        if not os.path.exists(path):
            raise FileNotFoundError(f"MDS_WRITE_FIELD: {path} does not exist (OPEN status='old')")
        fh = open(path, "r+b")
    a = np.asarray(fldRL.data, np.float64)
    if a.ndim == 3:
        a = a[:, None]
    if a.shape[1] != kSize:
        raise ValueError(f"MDS_WRITE_FIELD: {fNameIL}: field has {a.shape[1]} levels, kSize = {kSize}")
    gloBuff = _GLOBUFF.setdefault((dt, xSize, ySize), np.zeros((ySize, xSize)))   # xy_buffer_r8 / _r4 (static)
    w2 = mds.w2
    with fh:
        for k in range(kLo, kHi + 1):                                     # :285
            zeroBuff = k == kLo                                           # :286
            if useExch2ioLayOut:                                          # gather_2d_rx.template:59-94
                if zeroBuff:
                    gloBuff[:, :] = 0.                                    # gloBuff(i,j) = 0.
                for bj in range(1, nSy + 1):
                    for bi in range(1, nSx + 1):
                        t = (bi - 1) + (bj - 1) * nSx
                        tN = int(w2.W2_myTileList[bi, bj])
                        if int(w2.exch2_mydNx[tN]) > xSize:
                            iGjLoc, jGjLoc = 0, int(w2.exch2_mydNx[tN]) // xSize
                        elif int(w2.exch2_tNy[tN]) > ySize:
                            iGjLoc, jGjLoc = int(w2.exch2_mydNx[tN]), 0
                        else:
                            iGjLoc, jGjLoc = 0, 1
                        for j in range(1, sNy + 1):
                            iG = int(w2.exch2_txGlobalo[tN]) + iGjLoc * (j - 1) - 1
                            jG = int(w2.exch2_tyGlobalo[tN]) + jGjLoc * (j - 1)
                            gloBuff[jG - 1, iG:iG + sNx] = a[t, k - 1, OLy + j - 1, OLx:OLx + sNx]
            else:                                                         # :95-115
                iBase, jBase = mds.myXGlobalLo - 1, mds.myYGlobalLo - 1
                for bj in range(1, nSy + 1):
                    for bi in range(1, nSx + 1):
                        t = (bi - 1) + (bj - 1) * nSx
                        for j in range(1, sNy + 1):
                            iG = iBase + (bi - 1) * sNx
                            jG = jBase + (bj - 1) * sNy + j
                            gloBuff[jG - 1, iG:iG + sNx] = a[t, k - 1, OLy + j - 1, OLx:OLx + sNx]
            irec = 1 + k - kLo + (irecord - 1) * nNz                      # :337
            rec = gloBuff.astype(dt).tobytes()                            # WRITE(dUnit,rec=irec) xy_buffer(1:x*y)
            fh.seek((irec - 1) * len(rec))
            fh.write(rec)


_GLOBUFF = {}


def _per_tile_io(fName, IL, pfName, filePrec, globalFile, kSize, kLo, kHi, fldRL, irecord, nNz, xSize, ySize,
                 useExch2ioLayOut, iGjLoc, jGjLoc, writeMetaF, myIter, *, mds):
    """The multi-CPU-IO branch of MDS_WRITE_FIELD (:366-562): per-tile records of a global file or tile files."""
    sz = mds.size
    sNx, sNy, OLx, OLy, nSx, nSy = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.nSx, sz.nSy
    dt = _dtype(filePrec)
    # :366-407 MDS_PASS_R8toRL / R4toRL( shared3dBuf, fldRL, oLi=0, oLj=0, nNz, kLo, kSize, 0, 0, .FALSE. ):
    # the interior of levels kLo..kHi of every tile, buffer(i,j,k,bi,bj) Fortran order
    a = np.asarray(fldRL.data, np.float64)
    if a.ndim == 3:
        a = a[:, None]
    if a.shape[1] != kSize:
        raise ValueError(f"MDS_WRITE_FIELD: {fName}: field has {a.shape[1]} levels, kSize = {kSize}")
    buf = a[:, kLo - 1:kHi, OLy:OLy + sNy, OLx:OLx + sNx]                  # [tile, k, j, i]
    reclen = np.dtype(dt).itemsize
    dUnit = None
    if globalFile:                                                        # :424-435
        dataFName = mds.path(fName[:IL] + ".data")
        dUnit = _open(dataFName)
    for bj in range(1, nSy + 1):                                          # :438-559
        for bi in range(1, nSx + 1):
            t = (bi - 1) + (bj - 1) * nSx                                 # bBij / (sNx*sNy*nNz)
            tNx, tNy = sNx, sNy
            global_nTx = xSize // sNx
            tBx = mds.myXGlobalLo - 1 + (bi - 1) * sNx
            tBy = mds.myYGlobalLo - 1 + (bj - 1) * sNy
            if useExch2ioLayOut:                                          # :447-469 (GO lane)
                w2 = mds.w2
                tN = int(w2.W2_myTileList[bi, bj])                        # :449
                tBx = int(w2.exch2_txGlobalo[tN]) - 1                     # :453
                tBy = int(w2.exch2_tyGlobalo[tN]) - 1                     # :454
                if int(w2.exch2_mydNx[tN]) > xSize:                       # :455-458  fold the face
                    iGjLoc, jGjLoc = 0, int(w2.exch2_mydNx[tN]) // xSize
                elif int(w2.exch2_tNy[tN]) > ySize:                       # :459-462  a long line
                    iGjLoc, jGjLoc = int(w2.exch2_mydNx[tN]), 0
                else:                                                     # :463-467  default
                    iGjLoc, jGjLoc = 0, 1
            if globalFile:                                                # :470-489
                for k in range(kLo, kHi + 1):
                    for j in range(1, tNy + 1):
                        irec = (1 + (tBx + (j - 1) * iGjLoc) // tNx + (tBy + (j - 1) * jGjLoc) * global_nTx
                                + (k - kLo + (irecord - 1) * nNz) * global_nTx * ySize)
                        dUnit.seek((irec - 1) * sNx * reclen)
                        dUnit.write(buf[t, k - kLo, j - 1, :].astype(dt).tobytes())
            else:                                                         # :490-524
                iG = bi + (mds.myXGlobalLo - 1) // sNx
                jG = bj + (mds.myYGlobalLo - 1) // sNy
                dataFName = f"{pfName}.{iG:03d}.{jG:03d}.data"
                with _open(mds.path(dataFName)) as fh:
                    irec = irecord
                    fh.seek((irec - 1) * sNx * sNy * nNz * reclen)
                    fh.write(np.ascontiguousarray(buf[t]).astype(dt).tobytes())
            if not globalFile and writeMetaF:                             # :526-556
                iG = bi + (mds.myXGlobalLo - 1) // sNx
                jG = bj + (mds.myYGlobalLo - 1) // sNy
                metaFName = f"{pfName}.{iG:03d}.{jG:03d}.meta"
                dimList = [[xSize, tBx + 1, tBx + tNx], [ySize, tBy + 1, tBy + tNy], [nNz, 1, nNz]]
                nDims = 2 if nNz == 1 else 3
                mds_write_meta(mds.path(metaFName), mds.path(dataFName), mds.the_run_name, " ", filePrec, nDims,
                               dimList, (iGjLoc, jGjLoc), 0, [blank8c], 0, [0.0], oneRL, irecord, myIter,
                               useCal=mds.useCal)
    if dUnit is not None:                                                 # :558-562
        dUnit.close()


def _open(path):
    """OPEN( dUnit, file=..., status='unknown', access='direct' ): create if missing, write records in place."""
    return open(path, "r+b" if os.path.exists(path) else "w+b")


def mds_wr_metafiles(fName, filePrec, globalFile, useCurrentDir, nNx, nNy, nNz, titleLine, nFlds, fldList, nTimRec,
                     timList, misVal, irecord, myIter, *, mds):
    """MDS_WR_METAFILES( fName, filePrec, globalFile, useCurrentDir, nNx, nNy, nNz, titleLine, nFlds, fldList,
    nTimRec, timList, misVal, irecord, myIter, myThid )   @63cdc0b pkg/mdsio/mdsio_wr_metafiles.F:7-222

    C     | o Write the meta files of an MDS file (global or per tile) with a field list and time records

    One process (MASTER_CPU_IO); the exch2 I/O layout (W2_useE2ioLayOut, :106-111, :168-188) with mds.w2 (GO lane)."""
    sz = mds.size
    sNx, sNy, nSx, nSy = sz.sNx, sz.sNy, sz.nSx, sz.nSy
    useE2 = False
    if mds.exch2:                                                         # :106-111 (GO lane)
        if mds.w2 is None:
            raise NotImplementedError("MDS_WR_METAFILES: an ALLOW_EXCH2 context needs the W2 common blocks (w2=)")
        useE2 = bool(mds.w2.W2_useE2ioLayOut)
    xSize = sNx * nSx * sz.nPx                                            # :104-105
    ySize = sNy * nSy * sz.nPy
    if useE2:
        xSize, ySize = mds.w2.exch2_global_Nx, mds.w2.exch2_global_Ny     # :108-109
    if nNx == 1:                                                          # :112-113
        xSize = 1
    if nNy == 1:
        ySize = 1
    IL = len(fName.rstrip())
    if mds.useSingleCpuIO or globalFile:                                  # :118-142
        dataFName, metaFName = fName[:IL] + ".data", fName[:IL] + ".meta"
        dimList = [[xSize, 1, xSize], [ySize, 1, ySize], [nNz, 1, nNz]]
        nDims = 2 if nNz == 1 else 3
        mds_write_meta(mds.path(metaFName), mds.path(dataFName), mds.the_run_name, titleLine, filePrec, nDims,
                       dimList, (0, 1), nFlds, fldList, nTimRec, timList, misVal, irecord, myIter,
                       useCal=mds.useCal)
        return
    pIL = len(mds.mdsioLocalDir.rstrip())                                 # :143-150
    pfName = fName[:IL] if (useCurrentDir or pIL == 0) else mds.mdsioLocalDir[:pIL] + fName[:IL]
    for bj in range(1, nSy + 1):                                          # :154-216
        for bi in range(1, nSx + 1):
            iG = bi + (mds.myXGlobalLo - 1) // sNx
            jG = bj + (mds.myYGlobalLo - 1) // sNy
            dataFName = f"{pfName}.{iG:03d}.{jG:03d}.data"
            metaFName = f"{pfName}.{iG:03d}.{jG:03d}.meta"
            tBx = mds.myXGlobalLo - 1 + (bi - 1) * sNx                    # :164
            tBy = mds.myYGlobalLo - 1 + (bj - 1) * sNy                    # :165
            map2gl = (0, 1)                                               # :166-167
            if useE2:                                                     # :168-188 (GO lane)
                w2 = mds.w2
                tN = int(w2.W2_myTileList[bi, bj])                        # :170
                tBx = int(w2.exch2_txGlobalo[tN]) - 1                     # :171
                tBy = int(w2.exch2_tyGlobalo[tN]) - 1                     # :172
                if nNx == 0 and nNy == 0:                                 # :173
                    if int(w2.exch2_mydNx[tN]) > xSize:                   # :174-177
                        map2gl = (0, int(w2.exch2_mydNx[tN]) // xSize)
                    elif int(w2.exch2_tNy[tN]) > ySize:                   # :178-181
                        map2gl = (int(w2.exch2_mydNx[tN]), 0)
                    else:                                                 # :182-185
                        map2gl = (0, 1)
            dimList = [[xSize, tBx + 1, tBx + sNx], [ySize, tBy + 1, tBy + sNy], [nNz, 1, nNz]]
            nDims = 2 if nNz == 1 else 3
            if nNx == 1:                                                  # :201-204
                dimList[0][1] = dimList[0][2] = 1
            if nNy == 1:
                dimList[1][1] = dimList[1][2] = 1
            mds_write_meta(mds.path(metaFName), mds.path(dataFName), mds.the_run_name, titleLine, filePrec, nDims,
                           dimList, map2gl, nFlds, fldList, nTimRec, timList, misVal, irecord, myIter,
                           useCal=mds.useCal)
