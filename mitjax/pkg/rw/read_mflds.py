"""pkg/rw/read_mflds.F @63cdc0b: reading a multi-field file (a pickup) by field name: READ_MFLDS_SET,
READ_MFLDS_3D_RL, READ_MFLDS_LEV_RL (lane M4COL), READ_MFLDS_CHECK. `MFlds` is the COMMON block of pkg/rw/RW_MFLDS.h
(thirdDim, nFl3D, nFlds, nMissFld, mFldsFile, fldList, fldMiss), returned updated by each routine (host side).

Owner: M1 core lane (Task 11, used by READ_PICKUP).
"""

from dataclasses import dataclass, field, replace

from mitjax.io import mds

sizFldList = 100            # RW_MFLDS.h  PARAMETER( sizFldList = 100 )


@dataclass(frozen=True)
class MFlds:
    """COMMON /RW_MFLDS_I/, /RW_MFLDS_C/ (pkg/rw/RW_MFLDS.h)."""
    thirdDim: int = 0
    nFl3D: int = 0
    nFlds: int = 0
    nMissFld: int = 0
    mFldsFile: str = " "
    fldList: tuple = field(default_factory=tuple)
    fldMiss: tuple = field(default_factory=tuple)


def READ_MFLDS_SET(fName, fileDim3, myIter, *, rw):
    """READ_MFLDS_SET( fName, nbFields, filePrec, fileDim3, myIter, myThid )   @63cdc0b pkg/rw/read_mflds.F:59-232

    C     Read, from a meta file, the list of fields of a multi-fields file and set the file name and the list in
    C     common block (RW_MFLDS.h) for later reading of the fields.

    Returns (MFlds, nbFields, filePrec). :118-119 nbFields = 0, filePrec = 0; :136-144 reset the common block;
    :147-159 MDS_READ_META (mitjax/io/mds.py read_meta; a missing meta file leaves nFlds = 0, as MDS_READ_META does
    when it finds no file: nFlds is the size of the list on input and 0 on output); :163-183 nFl3D: with a 2-D
    file holding 3-D and 2-D fields, the number of 3-D fields; :229 nbFields = nFlds. The debugLevel printout
    (:186-223) writes STDOUT only."""
    import os
    # MDS_READ_META's search (pkg/mdsio/mdsio_read_meta.F:139-166): {fName}.meta (global file), else the tile file
    # {fName}.{iG}.{jG}.meta with iG = jG = 1 for one process (myXGlobalLo = myYGlobalLo = 1), else .001.001.meta
    meta = None
    for cand in (str(fName).strip() + ".meta", str(fName).strip() + f".{1:03d}.{1:03d}.meta"):
        if os.path.exists(rw.path(cand)):
            meta = rw.path(cand)
            break
    info = mds.read_meta(meta) if meta is not None else {}
    filePrec = info.get("filePrec", 0)
    fldList = tuple(f"{n:<8s}"[:8] for n in info.get("fldList", ()))
    nFlds = len(fldList) if fldList else 0
    nRecords = info.get("nRecords", 0)
    nDims = info.get("nDims", 0)
    thirdDim = fileDim3
    nFl3D = 0
    if nFlds >= 1:                                                              # :164-183
        if nDims == 2 and thirdDim > 1 and nFlds < nRecords:
            if (nRecords - nFlds) % (thirdDim - 1) == 0:
                nFl3D = (nRecords - nFlds) // (thirdDim - 1)
        if nFlds != nRecords and nFl3D == 0:
            raise ValueError(f"READ_MFLDS_SET: Pb with Nb of records={nRecords} (3rd-Dim={thirdDim}) does not match "
                             f"Nb of flds={nFlds}\nABNORMAL END: S/R READ_MFLDS_SET (Nb-records Pb)")
    st = MFlds(thirdDim=thirdDim, nFl3D=nFl3D, nFlds=nFlds, nMissFld=0, mFldsFile=str(fName), fldList=fldList,
               fldMiss=())
    return st, nFlds, filePrec


def READ_MFLDS_3D_RL(fldName, field, nj, fPrec, nNz, myIter, *, mf, rw):
    """READ_MFLDS_3D_RL( fldName, field, nj, fPrec, nNz, myIter, myThid )   @63cdc0b pkg/rw/read_mflds.F:238-358

    C     Read, from the file set by READ_MFLDS_SET, the field "fldName" (3-D, nNz levels).

    Returns (field, nj, mf). With a field list (:302-331): nj = position of fldName in the list (first match), 0 if
    missing (then fldMiss gets the name and the field is left unchanged); a 2-D field after the nFl3D 3-D fields is
    at record nj + nFl3D*(thirdDim-1) (:325). Without a list (:332-340): nj = nj + 1. :342-355 MDS_READ_FIELD(
    mFldsFile, fPrec, .FALSE., 'RL', nNz, 1, nNz, field, dummyRS, nj, myThid) when nj >= 1."""
    name = f"{fldName:<8s}"[:8]
    if mf.nFlds >= 1:
        nj = 0
        for j in range(1, mf.nFlds + 1):                                        # :305-307
            if name == mf.fldList[j - 1] and nj == 0:
                nj = j
        if nj == 0:                                                             # :308-321
            mf = replace(mf, nMissFld=mf.nMissFld + 1, fldMiss=mf.fldMiss + (name,))
        else:                                                                   # :322-331
            if nj > mf.nFl3D:
                nj = nj + mf.nFl3D * (mf.thirdDim - 1)
    elif nj >= 0:                                                               # :332-340
        nj = nj + 1
    if nj >= 1:                                                                 # :342-355
        field = rw.mds_read_field(mf.mFldsFile, fPrec, nNz, field, nj)
    return field, nj, mf


def READ_MFLDS_LEV_RL(fldName, field, nj, fPrec, kSiz, kLo, kHi, myIter, *, mf, rw):
    """READ_MFLDS_LEV_RL( fldName, field, nj, fPrec, kSiz, kLo, kHi, myIter, myThid )
    @63cdc0b pkg/rw/read_mflds.F:364-490

    C     Read, from the file set by READ_MFLDS_SET, the field "fldName" and store it in levels kLo:kHi of the
    C     array "field" (kSiz levels).

    Lane M4COL (SEAICE_READ_PICKUP's siTICE, seaice_read_pickup.F:199-203). Returns (field, nj, mf). The field-list
    search (:431-460) and the no-list increment (:461-469) as READ_MFLDS_3D_RL; :471-482 MDS_READ_FIELD( mFldsFile,
    fPrec, .FALSE., 'RL', kSiz, kLo, kHi, field, dummyRS, nj, myThid ): levels kLo..kHi of `field` (an FArray with a
    k axis of kSiz levels) replaced by record nj (units of kHi-kLo+1 levels); the other levels are unchanged. The
    debugLevel printout is STDOUT only."""
    import jax.numpy as jnp
    from mitjax.farray import FArray
    name = f"{fldName:<8s}"[:8]
    if mf.nFlds >= 1:
        nj = 0
        for j in range(1, mf.nFlds + 1):                                        # :434-436
            if name == mf.fldList[j - 1] and nj == 0:
                nj = j
        if nj == 0:                                                             # :437-450
            mf = replace(mf, nMissFld=mf.nMissFld + 1, fldMiss=mf.fldMiss + (name,))
        else:                                                                   # :451-460
            if nj > mf.nFl3D:
                nj = nj + mf.nFl3D * (mf.thirdDim - 1)
    elif nj >= 0:                                                               # :461-469
        nj = nj + 1
    if nj >= 1:                                                                 # :471-482
        (_, ilo, ihi), (_, jlo, jhi), (_, klo, khi) = field.dims
        if khi - klo + 1 != kSiz:
            raise ValueError(f"READ_MFLDS_LEV_RL: {field.name} has {khi - klo + 1} levels, kSiz = {kSiz}")
        lev = FArray(field.data[:, kLo - klo:kHi - klo + 1], field.name, tiled=field.tiled, i=(ilo, ihi),
                     j=(jlo, jhi), k=(kLo, kHi))
        lev = rw.mds_read_field(mf.mFldsFile, fPrec, kHi - kLo + 1, lev, nj)
        field = FArray(jnp.asarray(field.data).at[:, kLo - klo:kHi - klo + 1].set(lev.data), field.name,
                       tiled=field.tiled, _dims=field.dims)
    return field, nj, mf


def READ_MFLDS_CHECK(nMissing, myIter, *, mf):
    """READ_MFLDS_CHECK( missFldList, nMissing, myIter, myThid )   @63cdc0b pkg/rw/read_mflds.F:622-750

    Returns (missFldList, nMissing, mf): the missing-field list (at most nMissing on input), nMissing = the number
    missing, and the common block reset (mFldsFile blank, :733-748); the debugLevel printout is STDOUT only.
    Ported: the list/count and the reset; a list truncated (nMissFld > nMissing on input) returns nMissFld so that
    the caller stops as READ_PICKUP does (read_pickup.F:473-478)."""
    miss = mf.fldMiss[:nMissing]
    n = mf.nMissFld
    return miss, n, MFlds()
