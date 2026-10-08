"""RBCS_FIELDS.h (pkg/rbcs/RBCS_FIELDS.h @63cdc0b): the declarations of the relaxation masks, fields and the loaded
records, as Fortran-index arrays (M3 Task 30)."""

from mitjax.model.grid import local


def rbcs_array(name, size):
    """One `(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)` RBCS_FIELDS.h array (RBC_mask(..,irbc), RBC_maskU/V, RBCtemp,
    RBCsalt, RBCuVel, RBCvVel, rbct0/1, rbcs0/1, rbcu0/1, rbcv0/1: :16-60), every point NaN until written."""
    return local(name, size, i=(1 - size.OLx, size.sNx + size.OLx), j=(1 - size.OLy, size.sNy + size.OLy),
                 k=(1, size.Nr))
