"""W2_EEBOOT (pkg/exch2/w2_eeboot.F @63cdc0b): set up the WRAPPER2 (exch2) execution environment, i.e. the W2 tile
topology, and `exch2_topology`, the part of it the grid routines read (Exch2Topology of
mitjax/model/src/ini_parms.py).

    @63cdc0b pkg/exch2/w2_eeboot.F:7-122

    w2, io = w2_eeboot(exp)                         # W2Common (all W2 common blocks) and the records written
    io.records(STANDARD_MESSAGE_UNIT)               # the W2 lines of STDOUT
    io.records(w2.W2_oUnit)                         # w2_tile_topology.0000.log
    ex2 = exch2_topology(w2)                        # -> ini_parms_grid(exp, ex2)

Host side, run once (EEBOOT calls W2_EEBOOT before the model, eesupp/src/eeboot.F under ALLOW_EXCH2). The chain as
global_ocean.90x40x15 executes it (docs/coverage/global_ocean.90x40x15.md): W2_READPARMS (no data.exch2: single
facet, W2_printMsg = -1, W2_mapIO = -1), the log file `w2_tile_topology.0000.log` (:85-95), W2_E2SETUP
(W2_SET_SINGLE_FACET, W2_SET_F2F_INDEX, W2_SET_MAP_TILES, W2_SET_MAP_CUMSUM, W2_SET_TILE2TILES), W2_MAP_PROCS,
W2_PRINT_COMM_SEQUENCE (:106-108), the log closed (:111-115). Unported topologies raise in the routines.

The log file's Fortran unit (MDSFINDUNIT, :90) is represented by its file name as the key of the record list.
Pre-conditions checked here: the build compiles pkg/exch2's own W2_EXCH2_SIZE.h (W2_OPTIONS.h may be the
experiment's: its CPP options come from the build's cpp), myProcId = 0, nPx*nPy = 1 (the oracle runs one process).
"""

import math

from mitjax import paths
from mitjax.eesupp.print import (SQUEEZE_BOTH, SQUEEZE_RIGHT, STANDARD_MESSAGE_UNIT, MessageUnits, ilnblnk,
                                 internal_write, print_message)
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.exch2.w2_e2setup import w2_e2setup
from mitjax.pkg.exch2.w2_exch2_h import W2_maxNeighbours, W2Common
from mitjax.pkg.exch2.w2_map_procs import w2_map_procs
from mitjax.pkg.exch2.w2_print_comm_sequence import w2_print_comm_sequence
from mitjax.pkg.exch2.w2_readparms import use_cubed_sphere_exchange, w2_readparms


def _check_build(exp):
    """W2_EXCH2_SIZE.h must be pkg/exch2's own (its PARAMETERs are hard-coded in w2_exch2_h.py). W2_OPTIONS.h may be
    the experiment's (adjustment.cs-32x32x1 defines W2_CUMSUM_USE_MATRIX): its options are read from the build's cpp
    (cfg.cpp) where the set-up tests them."""
    links = exp.farm.manifest.get("links", {})
    for name in ("W2_EXCH2_SIZE.h",):
        want = str((paths.UPSTREAM / "pkg" / "exch2" / name).resolve())
        if links.get(name) != want:
            raise NotImplementedError(f"W2_EEBOOT: this build's {name} is {links.get(name)}, not pkg/exch2's own: "
                                      "an experiment override is not ported")


def w2_eeboot(exp, io=None, myThid=1):
    """W2_EEBOOT on the experiment `exp` (mitjax.config.params.Experiment, ALLOW_EXCH2 build). Returns the filled
    W2Common and the MessageUnits holding what it wrote (STDOUT records and the log file)."""
    cfg = exp.cfg
    if not cfg.cpp.ALLOW_EXCH2:
        raise ValueError("W2_EEBOOT: the build does not compile pkg/exch2")
    _check_build(exp)
    sz = cfg.size
    io = MessageUnits() if io is None else io
    if io.myProcId != 0 or sz.nPx * sz.nPy != 1:
        raise NotImplementedError("W2_EEBOOT: more than one process is not ported (the oracle runs one)")
    useCubedSphereExchange = use_cubed_sphere_exchange(exp)
    w2 = W2Common.declare(sz, cumsum_matrix=bool(cfg.cpp.W2_CUMSUM_USE_MATRIX))

    # Initialise to zero EXCH2_TOPOLOGY common blocks (:45-74)
    w2.exch2_nTiles = 0
    for I in range(1, w2.W2_maxNbTiles + 1):
        for name in ("exch2_tNx", "exch2_tNy", "exch2_tBasex", "exch2_tBasey", "exch2_txGlobalo", "exch2_tyGlobalo",
                     "exch2_isWedge", "exch2_isNedge", "exch2_isEedge", "exch2_isSedge", "exch2_myFace",
                     "exch2_mydNx", "exch2_mydNy", "exch2_nNeighbours"):
            getattr(w2, name)[I] = 0
        for J in range(1, W2_maxNeighbours + 1):
            w2.exch2_neighbourId[J, I] = 0
            w2.exch2_opposingSend[J, I] = 0
            for ii in range(1, 5):
                w2.exch2_pij[ii, J, I] = 0
            for name in ("exch2_oi", "exch2_oj", "exch2_iLo", "exch2_iHi", "exch2_jLo", "exch2_jHi"):
                getattr(w2, name)[J, I] = 0
    w2.W2_oUnit = STANDARD_MESSAGE_UNIT                                       # :75

    # Set W2-EXCH2 parameters
    w2_readparms(w2, exp=exp, io=io, myThid=myThid)                           # :78

    stdUnit = STANDARD_MESSAGE_UNIT                                           # :80
    msgBuf = internal_write("(A)", "===== Start setting W2 TOPOLOGY:")        # :81
    print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)              # :82

    # Open message output-file (if needed) (:85-95)
    if w2.W2_printMsg < 0:
        iTmp = max(4, 1 + int(math.log10(float(sz.nPx * sz.nPy))))           # :86  INT(LOG10(DFLOAT(nPx*nPy))); MINMAX-INT: integer (no tie or NaN case)
        fmtStr = fortran_write("(2(A,I1),A)", "(A,I", iTmp, ".", iTmp, ",A)")  # :87
        fName = fortran_write(fmtStr, "w2_tile_topology.", io.myProcId, ".log")   # :88
        iLen = ilnblnk(fName)                                                 # :89
        w2.W2_oUnit = fName[:iLen]                                            # :90-92 MDSFINDUNIT + OPEN
        io.units[w2.W2_oUnit] = []                                            # status='unknown': a new file
        msgBuf = internal_write("(2A)", " write to log-file: ", fName[:iLen])     # :93
        print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)          # :94

    # Define topology for every tile
    w2_e2setup(w2, cfg=cfg, io=io, useCubedSphereExchange=useCubedSphereExchange, myThid=myThid)   # :98
    # --   Decide which tiles this process handles
    w2_map_procs(w2, cfg=cfg, io=io, myThid=myThid)                           # :103
    # Print out the topology communication schedule
    if w2.W2_printMsg != 0:                                                   # :106-108
        w2_print_comm_sequence(w2, cfg=cfg, io=io, myThid=myThid)

    # Close message output-file (if needed) (:111-115)
    if w2.W2_oUnit != STANDARD_MESSAGE_UNIT:
        msgBuf = internal_write("(A)", "===  End TOPOLOGY report ===")       # :112
        print_message(msgBuf, w2.W2_oUnit, SQUEEZE_BOTH, myThid, io=io)       # :113
    msgBuf = internal_write("(A)", "=====       setting W2 TOPOLOGY: Done")  # :116
    print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)              # :117
    msgBuf = internal_write("(A)", " ")                                       # :118
    print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)              # :119
    return w2, io


def exch2_topology(w2):
    """The W2_EXCH2_TOPOLOGY.h values INI_PARMS, LOAD_GRID_SPACING and INI_LOCAL_GRID read under ALLOW_EXCH2, as the
    core lane's Exch2Topology: exch2_tBasex/y, exch2_mydNx/y per W2 tile (tuple index = tile number - 1, tiles
    1..exch2_nTiles; the readers take exch2_mydNx(1)), W2_myTileList in the order bi + (bj-1)*nSx."""
    from mitjax.model.src.ini_parms import Exch2Topology
    n = w2.exch2_nTiles
    sz = w2.size
    return Exch2Topology(
        exch2_tBasex=w2.exch2_tBasex.values(1, n), exch2_tBasey=w2.exch2_tBasey.values(1, n),
        exch2_mydNx=w2.exch2_mydNx.values(1, n), exch2_mydNy=w2.exch2_mydNy.values(1, n),
        W2_myTileList=tuple(w2.W2_myTileList[bi, bj] for bj in range(1, sz.nSy + 1) for bi in range(1, sz.nSx + 1)),
        source="pkg/exch2 W2 set-up (mitjax/pkg/exch2/w2_eeboot.py, w2_eeboot.F @63cdc0b)",
        exch2_myFace=w2.exch2_myFace.values(1, n))

