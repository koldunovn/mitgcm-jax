"""pkg/rw/write_rec.F @63cdc0b: WRITE_REC_3D_RL (the front-end WRITE_PICKUP calls), over MDS_WRITE_FIELD
(mitjax/pkg/mdsio/mdsio_write_field.py). Host-side; `mds` is the MdsContext of the run and `globalFile` the
COMMON /RD_WR_REC/ flag that INI_MODEL_IO sets from globalFiles (ini_model_io.F:188, SET_WRITE_GLOBAL_REC)."""

from mitjax.pkg.mdsio.mdsio_write_field import mds_write_field


def WRITE_REC_3D_RL(fName, fPrec, nNz, field, iRec, myIter, *, mds, globalFile):
    """WRITE_REC_3D_RL( fName, fPrec, nNz, field, iRec, myIter, myThid )   @63cdc0b pkg/rw/write_rec.F:399-458

    C WRITE_REC_3D_RL is a "front-end" interface to the low-level I/O routines.

    :449-455: useCurrentDir = .FALSE., fType = 'RL'; MDS_WRITE_FIELD( fName, fPrec, globalFile, useCurrentDir,
    fType, nNz, 1, nNz, field, dummyRS, iRec, myIter, myThid )."""
    useCurrentDir = False
    fType = "RL"
    mds_write_field(fName, fPrec, globalFile, useCurrentDir, fType, nNz, 1, nNz, field, iRec, myIter, mds=mds)
