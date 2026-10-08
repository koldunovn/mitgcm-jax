"""W2_READPARMS (pkg/exch2/w2_readparms.F @63cdc0b): initialise the W2_EXCH2 parameters (defaults, then the namelist
W2_EXCH2_PARM01 of "data.exch2" if that file exists).

    @63cdc0b pkg/exch2/w2_readparms.F:9-205

Ported branches: the defaults (:60-73; preDefTopol = 3 under useCubedSphereExchange, :66-67, ported as code because
the ELSE arm's :69 reassigns it, else 1 through `fortran_default`); without "data.exch2" (global_ocean.90x40x15,
adjustment.cs-32x32x1, advect_cs, global_ocean.cs32x15: data.exch2.mpi is not opened by a serial run) the
"not found" messages (:130-142); with "data.exch2" (solid-body.cs-32x32x1: `W2_mapIO = 1`) OPEN_COPY_DATA_FILE's
STDOUT echo (eesupp/src/open_copy_data_file.F:66-150, `open_copy_data_file_echo`) and the scalar namelist values
preDefTopol, W2_mapIO, W2_printMsg, W2_useE2ioLayOut (:110-122). The list values dimsFacets, facetEdgeLink and
blankList of data.exch2 are not set by any M2 experiment and raise (their REAL*4 decimal conversion and list order
are not ported). The copies into the common block (:146-154) copy the local lists.
"""

import numpy as np

from mitjax.eesupp.print import SQUEEZE_RIGHT, STANDARD_MESSAGE_UNIT, internal_write, print_message
from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.exch2.w2_exch2_h import W2_maxNbFacets, FIntArray

RP = "pkg/exch2/w2_readparms.F"


def use_cubed_sphere_exchange(exp):
    """useCubedSphereExchange of EEPARMS ("eedata"; default eesupp/src/eeset_parms.F:106 `.FALSE.`)."""
    rp = RunParams(exp.run)
    if rp.has("eedata", "EEPARMS", "useCubedSphereExchange"):
        return bool(rp.get("eedata", "EEPARMS", "useCubedSphereExchange"))
    return fortran_default("eesupp/src/eeset_parms.F:106", "useCubedSphereExchange", exp).value


def w2_readparms(w2, *, exp, io, myThid=1):
    """W2_READPARMS( myThid ): sets W2_printMsg, W2_mapIO, W2_useE2ioLayOut, preDefTopol, blankList, nFacets,
    nBlankTiles, facet_dims, facet_link, e2FillValue_* in `w2` (W2Common) and prints to `io` (MessageUnits)."""
    stdUnit = STANDARD_MESSAGE_UNIT                                           # :60
    namList_NbFacets = W2_maxNbFacets * 2                                     # :38-39 PARAMETER
    dimsFacets = FIntArray("dimsFacets", 2 * namList_NbFacets)                # :40
    facetEdgeLink = FIntArray("facetEdgeLink", 4, namList_NbFacets, dtype=np.float32, fill=0.0)   # :41

    # --   Default values for W2_EXCH2 (:62-73)
    w2.W2_printMsg = fortran_default(f"{RP}:63", "W2_printMsg", exp).value              # :63  W2_printMsg = -1
    w2.W2_mapIO = fortran_default(f"{RP}:64", "W2_mapIO", exp).value                    # :64  W2_mapIO = -1
    w2.W2_useE2ioLayOut = fortran_default(f"{RP}:65", "W2_useE2ioLayOut", exp).value    # :65  = .TRUE.
    if use_cubed_sphere_exchange(exp):                                        # :66
        # (fortran_default cannot cite :67: it reads the ELSE arm's :69 as a later assignment; ported as code)
        w2.preDefTopol = 3                                                    # :67  preDefTopol = 3
    else:
        w2.preDefTopol = fortran_default(f"{RP}:69", "preDefTopol", exp,
                                         condition="IF ( useCubedSphereExchange ) THEN ... ELSE").value  # :69  = 1
    for i in range(1, w2.W2_maxNbTiles + 1):                                  # :71-73
        w2.blankList[i] = 0

    # --   Initialise other params in namelist (:76-82)
    for j in range(1, W2_maxNbFacets * 2 + 1):
        dimsFacets[2 * j - 1] = 0
        dimsFacets[2 * j] = 0
        for i in range(1, 5):
            facetEdgeLink[i, j] = np.float32(0.0)                             # 0. (REAL*4)

    # -    Initialise other parameters (:85-93)
    w2.nFacets = 0
    w2.nBlankTiles = 0
    for j in range(1, W2_maxNbFacets + 1):
        w2.facet_dims[2 * j - 1] = 0
        w2.facet_dims[2 * j] = 0
        for i in range(1, 5):
            w2.facet_link[i, j] = np.float32(0.0)

    # Set filling value for face-corner halo regions (:96-99)
    w2.e2FillValue_RL = 0.0                                                   # 0. _d 0
    w2.e2FillValue_RS = 0.0                                                   # 0. _d 0
    w2.e2FillValue_R4 = np.float32(0.0)                                       # 0.e0
    w2.e2FillValue_R8 = 0.0                                                   # 0.d0

    # -    Check for file "data.ech2" (:107-143)
    fileExist = "data.exch2" in exp.run.files                                 # :108 INQUIRE( FILE='data.exch2' )
    if fileExist:
        msgBuf = internal_write("(A)", "W2_READPARMS: opening data.exch2")                     # :111
        print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)                          # :112
        open_copy_data_file_echo("data.exch2", _run_file_text(exp, "data.exch2"), io=io,
                                 nml_terminator=bool(exp.cfg.cpp.NML_TERMINATOR), myThid=myThid)  # :113-116
        # Read parameters from open data file (:119): the run's resolved namelist (mitjax/config/namelists.py)
        rp = RunParams(exp.run)
        for key in ("dimsFacets", "facetEdgeLink", "blankList"):
            if rp.has("data.exch2", "W2_EXCH2_PARM01", key):
                raise NotImplementedError(f"W2_READPARMS: {key} in data.exch2 is not ported")
        if rp.has("data.exch2", "W2_EXCH2_PARM01", "preDefTopol"):
            w2.preDefTopol = int(rp.get("data.exch2", "W2_EXCH2_PARM01", "preDefTopol"))
        if rp.has("data.exch2", "W2_EXCH2_PARM01", "W2_mapIO"):
            w2.W2_mapIO = int(rp.get("data.exch2", "W2_EXCH2_PARM01", "W2_mapIO"))
        if rp.has("data.exch2", "W2_EXCH2_PARM01", "W2_printMsg"):
            w2.W2_printMsg = int(rp.get("data.exch2", "W2_EXCH2_PARM01", "W2_printMsg"))
        if rp.has("data.exch2", "W2_EXCH2_PARM01", "W2_useE2ioLayOut"):
            w2.W2_useE2ioLayOut = bool(rp.get("data.exch2", "W2_EXCH2_PARM01", "W2_useE2ioLayOut"))
        msgBuf = internal_write("(A)", "W2_READPARMS: finished reading data.exch2")           # :120-121
        print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)                          # :122
        # :124-128 CLOSE (no STDOUT record)
    else:
        _not_found(w2, stdUnit, io, myThid)                                   # :129-142
    _after_read(w2, dimsFacets, facetEdgeLink, namList_NbFacets, stdUnit, io, myThid)
    return w2


def _not_found(w2, stdUnit, io, myThid):
    msgBuf = internal_write("(A)", "W2_READPARMS: file data.exch2 not found")                  # :130
    print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)                               # :131
    if w2.preDefTopol == 1:                                                                    # :132
        msgBuf = internal_write("(2A,I3)", "=> use W2_EXCH2 default:", " Single sub-domain (nFacets=1)")   # :133-134
    elif w2.preDefTopol == 3:                                                                  # :135
        msgBuf = internal_write("(2A,I3)", "=> use W2_EXCH2 default:", " regular 6-facets Cube")           # :136-137
    else:
        msgBuf = internal_write("(2A,I3)", "=> use W2_EXCH2 default:", " preDefTopol=", w2.preDefTopol)   # :139-140
    print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)                               # :142


def _after_read(w2, dimsFacets, facetEdgeLink, namList_NbFacets, stdUnit, io, myThid):
    # --   copy local arrays dimsFacets & facetEdgeLink to var in common block (:146-154)
    for j in range(1, 2 * W2_maxNbFacets + 1):
        w2.facet_dims[j] = dimsFacets[j]
    for j in range(1, W2_maxNbFacets + 1):
        for i in range(1, 5):
            w2.facet_link[i, j] = facetEdgeLink[i, j]

    # --   Check if too many values are specified in data.exch2 (:157-191)
    errCnt = 0
    for j in range(W2_maxNbFacets + 1, namList_NbFacets + 1):
        errFlag = False
        for i in range(1, 5):
            if facetEdgeLink[i, j] != np.float32(0.0):
                errFlag = True
        if errFlag:
            errCnt = errCnt + 1
    if errCnt > 0:
        raise RuntimeError("W2_READPARMS: Number of \"facetEdgeLink\" list in \"data.exch2\" exceeds maxNbFacets; "
                           "ABNORMAL END: S/R W2_READPARMS")
    errCnt = 0
    for j in range(2 * W2_maxNbFacets + 1, 2 * namList_NbFacets + 1):
        if dimsFacets[j] != 0:
            errCnt = errCnt + 1
    if errCnt > 0:
        raise RuntimeError("W2_READPARMS: Number of \"dimsFacets\" in \"data.exch2\" exceeds 2*maxNbFacets; "
                           "ABNORMAL END: S/R W2_READPARMS")

    # --   Print some Exch2 parameters (:194-202)
    msgBuf = internal_write("(A)", "W2_useE2ioLayOut=" + _fmt_L(w2.W2_useE2ioLayOut, 5)
                            + " ;/* T: use Exch2 glob IO map; F: use model default */")       # :194-195 '(A,L5,A)'
    print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)
    msgBuf = internal_write("(A,I4,A)", "W2_mapIO        =", w2.W2_mapIO,
                            " ; /* select option for Exch2 global-IO map */")                 # :197-198
    print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)
    msgBuf = internal_write("(A,I4,A)", "W2_printMsg     =", w2.W2_printMsg,
                            " ; /* select option for printing information */")                # :200-201
    print_message(msgBuf, stdUnit, SQUEEZE_RIGHT, myThid, io=io)


def _run_file_text(exp, name):
    """The text of the run directory's file `name` (the input file linkdata links)."""
    from pathlib import Path
    return Path(exp.run.files[name][1]).read_text()


def open_copy_data_file_echo(data_file, text, *, io, nml_terminator, myThid=1):
    """The STDOUT records of OPEN_COPY_DATA_FILE( data_file, ... ) (eesupp/src/open_copy_data_file.F @63cdc0b, the
    oracle's path: the file exists, no SINGLE_DISK_IO): the opening message (:66-69), the banner (:121-131), every
    record of the file as '>' followed by the record cut after its last non-blank character (at least one character;
    :134-145, RECORD is CHARACTER*(MAX_LEN_PREC) = 200, EEPARAMS.h:27, read with FMT='(A)'), and a blank record
    (:148-150). The scratch copy the namelist READ reads is mitjax/io/namelist.preprocess."""
    from mitjax.eesupp.print import ilnblnk
    MAX_LEN_PREC = 200                                                        # EEPARAMS.h:27
    COMMENT_CHARACTER = "#"                                                   # EEPARAMS.h:125 commentCharacter
    nmlEnd = " /" if nml_terminator else " &"                                 # nml_change_syntax.F:47-51
    msgBuf = internal_write("(A,A)", " OPEN_COPY_DATA_FILE: opening file ", data_file)          # :66-67
    print_message(msgBuf, STANDARD_MESSAGE_UNIT, SQUEEZE_RIGHT, myThid, io=io)                  # :68-69
    msgBuf = internal_write("(A)", "// =======================================================")   # :121-124
    print_message(msgBuf, STANDARD_MESSAGE_UNIT, SQUEEZE_RIGHT, myThid, io=io)
    msgBuf = internal_write("(A,A,A)", '// Parameter file "', data_file, '"')                     # :125-127
    print_message(msgBuf, STANDARD_MESSAGE_UNIT, SQUEEZE_RIGHT, myThid, io=io)
    msgBuf = internal_write("(A)", "// =======================================================")   # :128-131
    print_message(msgBuf, STANDARD_MESSAGE_UNIT, SQUEEZE_RIGHT, myThid, io=io)
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    for line in lines:                                                        # :134-145
        if len(line) > MAX_LEN_PREC or "\t" in line:
            raise NotImplementedError("OPEN_COPY_DATA_FILE: records longer than MAX_LEN_PREC or with tabs")
        RECORD = line.ljust(MAX_LEN_PREC)
        IL = max(ilnblnk(RECORD), 1)                                          # :136 MINMAX-INT: integer (no tie or NaN case)
        if RECORD[0] != COMMENT_CHARACTER:                                    # :137
            # NML_CHANGE_SYNTAX( RECORD ) (:139; nml_change_syntax.F:72-77): a record ' &' becomes nmlEnd
            # (nml_change_syntax.F:47-51: ' /' under NML_TERMINATOR, else ' &'); the echo prints the changed record
            il = max(ilnblnk(RECORD), 1)                                      # MINMAX-INT: integer (no tie or NaN case)
            if il == 2 and RECORD[:2] == " &":
                RECORD = nmlEnd + RECORD[2:]
        msgBuf = internal_write("(A,A)", ">", RECORD[:IL])                    # :142-144
        print_message(msgBuf, STANDARD_MESSAGE_UNIT, SQUEEZE_RIGHT, myThid, io=io)
    msgBuf = internal_write("(A)", " ")                                       # :148-150
    print_message(msgBuf, STANDARD_MESSAGE_UNIT, SQUEEZE_RIGHT, myThid, io=io)


def _fmt_L(v, w):
    """`Lw` output editing (F2008 10.7.3): w-1 blanks followed by T or F (mitjax.io.fortran_format has no L)."""
    return " " * (w - 1) + ("T" if v else "F")
