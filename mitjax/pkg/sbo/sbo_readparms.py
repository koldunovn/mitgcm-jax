"""SBO_READPARMS: pkg/sbo/sbo_readparms.F @63cdc0b (derived settings only; the namelist values come from
mitjax/config), and the run-time checks of SBO_CHECK (pkg/sbo/sbo_check.F), which changes no state."""

from dataclasses import dataclass

from mitjax.params_io import RunParams


@dataclass(frozen=True)
class SboParams:
    """SBO.h /sbo_params_r/ (SBO.h:44-45)."""
    sbo_monFreq: float


def sbo_readparms(exp, fp, *, usingCartesianGrid=False, usingCylindricalGrid=False):
    """SBO_READPARMS( myThid )   @63cdc0b pkg/sbo/sbo_readparms.F:7-126

    C     | SUBROUTINE SBO_READPARMS
    C     | o Routine to initialize SBO parameters and constants.

    :68 sbo_monFreq = monitorFreq (PARAMS.h, `fp.monitorFreq`), then data.sbo SBO_PARM01 may set it (:77);
    sbo_taveFreq is retired (:72, :87-104: a value in data.sbo is a fatal error). useSBO=.FALSE. returns at :45-53
    (the caller does not call SBO then). The PRINT_MESSAGE lines (:59-61, :82-84) belong to the parameter-file
    echo (mitjax/config) and are not reproduced here. SBO_CHECK (sbo_check.F:33-42) STOPs on a Cartesian or
    cylindrical grid: raised here."""
    if fp.monitorFreq < 0.:
        raise NotImplementedError("SBO_READPARMS: monitorFreq < 0 (ini_parms.F:1187-1196 default) is not ported")
    rp = RunParams(exp.run)
    sbo_monFreq = fp.monitorFreq                                                # :68
    if rp.has("data.sbo", "SBO_PARM01", "sbo_monFreq"):                         # :77
        sbo_monFreq = rp.get("data.sbo", "SBO_PARM01", "sbo_monFreq")
    if rp.has("data.sbo", "SBO_PARM01", "sbo_taveFreq"):                        # :87-104
        raise ValueError('S/R SBO_READPARMS: "sbo_taveFreq" is no longer allowed in file "data.sbo"')
    if usingCartesianGrid:                                                      # sbo_check.F:33-37
        raise ValueError("SBO not implemented for Cartesian Grid")
    if usingCylindricalGrid:                                                    # sbo_check.F:38-42
        raise ValueError("SBO not implemented for Cylindrical Grid")
    return SboParams(sbo_monFreq=float(sbo_monFreq))
