"""SALT_PLUME_READPARMS: pkg/salt_plume/salt_plume_readparms.F @63cdc0b, and the SALT_PLUME.h parameters read by
SEAICE_GROWTH (lane M4LAB session 3)."""

import dataclasses

import numpy as np

from mitjax.params_io import params_pytree


@params_pytree
@dataclasses.dataclass(frozen=True)
class SaltPlumeParams:
    """SALT_PLUME.h /SALT_PLUME_PARAMS_R/ SPsalFRAC (:58, :66-67; traced) and /SALT_PLUME_PARAMS_L/
    SaltPlumeSouthernOcean (:9, :13; static): the two parameters SEAICE_GROWTH reads (seaice_growth.F:2020, :2042)."""
    SPsalFRAC: float
    SaltPlumeSouthernOcean: bool = dataclasses.field(metadata=dict(static=True))


def salt_plume_readparms(cfg):
    """SALT_PLUME_READPARMS( myThid )   @63cdc0b pkg/salt_plume/salt_plume_readparms.F:6-125

    Ported for useSALT_PLUME = .FALSE. only (lab_sea/input: pkg/salt_plume compiled, not used): the routine prints
    the PACKAGES_UNUSED_MSG and RETURNs (:46-54) before the defaults (:57-75) and the namelist read, so nothing ever
    writes /SALT_PLUME_PARAMS_R/ or /SALT_PLUME_PARAMS_L/: SPsalFRAC and SaltPlumeSouthernOcean hold the static
    commons' zero-initialised storage of the oracle build (0.0 and .FALSE.), which SEAICE_GROWTH still reads
    (seaice_growth.F:2020, :2042; docs/ISSUES_UPSTREAM.md, "pkg/salt_plume parameters read without being set").
    The message itself is output only (not ported). useSALT_PLUME raises; SALT_PLUME_SPLIT_BASIN (SPsalFRAC(2)),
    which changes the common's shape, raises."""
    if cfg.use_flag("useSALT_PLUME"):
        raise NotImplementedError("SALT_PLUME_READPARMS: useSALT_PLUME (the defaults :57-75 and the namelist) is "
                                  "not ported")
    if cfg.cpp.flag("SALT_PLUME_SPLIT_BASIN", "SALT_PLUME_OPTIONS.h"):
        raise NotImplementedError("SALT_PLUME_READPARMS: SALT_PLUME_SPLIT_BASIN (SPsalFRAC(2)) is not ported")
    return SaltPlumeParams(SPsalFRAC=np.float64(0.0),          # never written: the static common's zero
                           SaltPlumeSouthernOcean=False)       # never written: the static common's .FALSE.
