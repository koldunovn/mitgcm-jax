"""pkg/rw/read_rec.F @63cdc0b: READ_REC_XY_RS, READ_REC_3D_RL (the front-ends the M1 initialisation calls), over
MDS_READ_FIELD (pkg/mdsio/mdsio_read_field.F; mitjax/io/mds.py `read_field`, `into_tiles`).

`RW` is the model-input context these routines need besides their arguments: the run directory the file names are
relative to (the model runs there; mdsioLocalDir is blank in every M1 run, so useCurrentDir = .FALSE. adds no
prefix), PARAMS.h readBinaryPrec and SIZE.h. The routines are host-side (numpy file reads) and return FArrays with
the declaration of `field`: the tile interiors from the file, the halos untouched (MDS_PASS_R8toRL /
MDS_PASS_R4toRL write the interior only, mdsio_pass_r4torl.F:71; the callers exchange).

Owner: M1 core lane (moved from mitjax/model/grid.py `ReadRec`, Task 10 follow-up).
"""

import os

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.io import mds


def _rewrap(data, like):
    return FArray(data, like.name, tiled=like.tiled, _dims=like.dims)


class RW:
    """Run directory, readBinaryPrec and SIZE.h of the run (see the module docstring)."""

    def __init__(self, rundir, readBinaryPrec, size, e2io=None):
        self.rundir, self.readBinaryPrec, self.size = str(rundir), int(readBinaryPrec), size
        self.e2io = e2io        # ADVECT lane (M2): mitjax.io.mds.E2ioLayout of an exch2 build with W2_useE2ioLayOut

    def path(self, fName):
        """MDS_READ_FIELD opens names relative to the run directory (useCurrentDir = .FALSE., mdsioLocalDir blank)."""
        return os.path.join(self.rundir, str(fName).strip())

    def layout(self):
        sz = self.size
        return mds.MdsLayout(sz.sNx, sz.sNy, sz.nSx, sz.nSy)

    def mds_read_field(self, fName, filePrec, nNz, field, irecord):
        """CALL MDS_READ_FIELD( fName, filePrec, useCurrentDir=.FALSE., fType, nNz, kLo=1, kHi=nNz, field, irecord,
        myThid ): `field` with levels 1..nNz of the interior replaced by record `irecord` (units of nNz levels)."""
        sz = self.size
        vals = mds.read_field(self.path(fName), filePrec, self.layout(), nNz=nNz, irecord=irecord, e2io=self.e2io)
        base = np.asarray(field.data)
        three_d = base.ndim == 4
        b = base if three_d else base[:, None]
        if b.shape[1] != nNz:
            raise ValueError(f"MDS_READ_FIELD: {field.name} has {b.shape[1]} levels, nNz = {nNz}")
        new = mds.into_tiles(vals, sz.OLx, sz.OLy, base=b)
        return _rewrap(jnp.asarray(new if three_d else new[:, 0]), field)

    # ------------------------------------------------------------------------------------------------------------
    def READ_REC_XY_RS(self, fName, field, iRec, myIter):
        """READ_REC_XY_RS( fName, field, iRec, myIter, myThid )   @63cdc0b pkg/rw/read_rec.F:22-69

        C READ_REC_XY_RS is a "front-end" interface to the low-level I/O
        C routines.

        :61-68: MDS_READ_FIELD( fName, readBinaryPrec, useCurrentDir=.FALSE., 'RS', nNz=1, 1, nNz, dummyRL, field,
        iRec, myThid ). `myIter` is not used (:50-55 are commented out)."""
        return self.mds_read_field(fName, self.readBinaryPrec, 1, field, iRec)

    def READ_REC_3D_RL(self, fName, fPrec, nNz, field, iRec, myIter):
        """READ_REC_3D_RL( fName, fPrec, nNz, field, iRec, myIter, myThid )   @63cdc0b pkg/rw/read_rec.F:318-376

        C READ_REC_3D_RL is a "front-end" interface to the low-level I/O routines.

        :367-374: MDS_READ_FIELD( fName, fPrec, useCurrentDir=.FALSE., 'RL', nNz, 1, nNz, field, dummyRS, iRec,
        myThid ). `myIter` is not used (commented out)."""
        return self.mds_read_field(fName, fPrec, nNz, field, iRec)


def READ_FLD_XY_RL(pref, suff, field, myIter, *, rw):
    """READ_FLD_XY_RL( pref, suff, field, myIter, myThid )   @63cdc0b pkg/rw/read_fld_xy_rl.F

    fullName = pref(s1Lo:s1Hi) [// suff(s2Lo:s2Hi)] (blanks trimmed, suff ignored if blank); MDS_READ_FIELD( fullName,
    readBinaryPrec, .FALSE., 'RL', nNz=1, 1, 1, field, dummyRS, iRec=1, myThid )."""
    full = str(pref).strip() + ("" if str(suff).strip() == "" else str(suff).strip())
    return rw.mds_read_field(full, rw.readBinaryPrec, 1, field, 1)


def READ_FLD_XYZ_RL(pref, suff, field, myIter, *, rw):
    """READ_FLD_XYZ_RL( pref, suff, field, myIter, myThid )   @63cdc0b pkg/rw/read_fld_xyz_rl.F

    fullName as READ_FLD_XY_RL; MDS_READ_FIELD( fullName, readBinaryPrec, .FALSE., 'RL', nNz=Nr, 1, Nr, field,
    dummyRS, iRec=1, myThid )."""
    full = str(pref).strip() + ("" if str(suff).strip() == "" else str(suff).strip())
    return rw.mds_read_field(full, rw.readBinaryPrec, rw.size.Nr, field, 1)
