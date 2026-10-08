"""SIZE.h of an experiment build (plan Task 6): the PARAMETERs of the SIZE.h the build compiles (the -mods copy where
there is one, else model/inc/SIZE.h, by the flat-directory precedence of mitjax/config/cpp_options.py), read from the
preprocessed text. Literal values only; Nx, Ny, MAX_OLX, MAX_OLY are evaluated from their own PARAMETER
expressions. testreport with MPI=0 uses SIZE.h, not SIZE.h_mpi (testreport:1133-1135, 771-834)."""

from dataclasses import dataclass

from mitjax.config import cpp_options, fortran

NAMES = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "nPx", "nPy", "Nx", "Ny", "Nr", "MAX_OLX", "MAX_OLY")


@dataclass(frozen=True)
class Size:
    sNx: int
    sNy: int
    OLx: int
    OLy: int
    nSx: int
    nSy: int
    nPx: int
    nPy: int
    Nx: int
    Ny: int
    Nr: int
    MAX_OLX: int
    MAX_OLY: int
    source: str          # the SIZE.h file the build links

    def env(self):
        return {k: getattr(self, k) for k in NAMES}


def read_size(farm, toolchain):
    text = cpp_options.preprocess(farm, toolchain, "SIZE.h")
    env = fortran.parameters(fortran.statements(text, "SIZE.h"), source="SIZE.h")
    missing = [k for k in NAMES if k not in env]
    if missing:
        raise ValueError(f"SIZE.h ({farm.manifest['links'].get('SIZE.h')}): no PARAMETER for {missing}")
    return Size(**{k: env[k] for k in NAMES}, source=farm.manifest["links"]["SIZE.h"])
