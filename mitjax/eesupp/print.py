"""PRINT_MESSAGE (eesupp/src/print.F @63cdc0b) on the path the oracle takes, and the Fortran units it writes to.

    io = MessageUnits()
    print_message('W2_READPARMS: file data.exch2 not found', STANDARD_MESSAGE_UNIT, SQUEEZE_RIGHT, io=io)
    io.records(STANDARD_MESSAGE_UNIT)   -> ['(PID.TID 0000.0001) W2_READPARMS: file data.exch2 not found']

Host side, on Python strings. A Fortran unit is a list of records (`MessageUnits.units`); a formatted
`WRITE(unit, fmt)` appends one record (`MessageUnits.write`). The oracle path (one process, one thread, no MPI):
numberOfProcs = 1 (eesupp/src/eeboot_minimal.F:84), pidIO = myProcId = 0 (:85-86), so PRINT_MESSAGE takes its
multi-process branch (print.F:101-140) with the prefix `(PID.TID 0000.0001) ` (fmtStr '(I4.4,A,I4.4)', print.F:112,
nPx*nPy < 10000). The single-process branch (numberOfProcs = 0, :94-100) and the copy of error messages to unit 0
(:148-157, errorMessageUnit only) are not on this path and raise.

The monitor lane's stand-in (mitjax/pkg/monitor/monitor_h.py: print_message) follows the same lines; this module is
the eesupp home the plan asks for (Task 11 follow-up: "move the PRINT_MESSAGE stand-ins to mitjax/eesupp").
"""

from dataclasses import dataclass, field

from mitjax.io.fortran_format import fortran_write

# eesupp/inc/EEPARAMS.h:22-25 @63cdc0b
MAX_LEN_MBUF = 512          # EEPARAMS.h:23  PARAMETER ( MAX_LEN_MBUF = 512 )
MAX_LEN_FNAM = 512          # EEPARAMS.h:25  PARAMETER ( MAX_LEN_FNAM = 512 )
# eesupp/inc/EEPARAMS.h:112-117
SQUEEZE_RIGHT = "R"         # EEPARAMS.h:113
SQUEEZE_LEFT = "L"          # EEPARAMS.h:115
SQUEEZE_BOTH = "B"          # EEPARAMS.h:117
# eesupp/src/eeboot.F:91, :104 (no HACK_FOR_GMAO_CPL)
STANDARD_MESSAGE_UNIT = 6   # eeboot.F:91   standardMessageUnit = 6
ERROR_MESSAGE_UNIT = 15     # eeboot.F:104  errorMessageUnit    = 15
PROCESS_HEADER = "PID.TID"  # eesupp/inc/EESUPPORT.h:23


@dataclass
class MessageUnits:
    """The records written to each Fortran unit (unit number, or the file name of a unit opened on a file), and the
    process/thread identity the prefix shows (eeboot_minimal.F:85 myProcId = 0; one thread, myThid = 1)."""
    myProcId: int = 0
    numberOfProcs: int = 1          # eeboot_minimal.F:84
    units: dict = field(default_factory=dict)

    def write(self, unit, record):
        """A formatted sequential WRITE of one record to `unit`."""
        self.units.setdefault(unit, []).append(record)

    def records(self, unit):
        return list(self.units.get(unit, []))


def ifnblnk(s):
    """IFNBLNK (eesupp/src/utils.F:88-116): 1-based index of the first non-blank character, 0 if none."""
    for n, c in enumerate(s, 1):
        if c != " ":
            return n
    return 0


def ilnblnk(s):
    """ILNBLNK (eesupp/src/utils.F:123-152): 1-based index of the last non-blank character, 0 if none."""
    for n in range(len(s), 0, -1):
        if s[n - 1] != " ":
            return n
    return 0


def internal_write(fmt, *values, length=MAX_LEN_MBUF):
    """WRITE(msgBuf, fmt) values... into a CHARACTER*(length) internal file: the record, blank-padded to `length`
    (a longer record is a run-time error in Fortran)."""
    rec = fortran_write(fmt, *values)
    if len(rec) > length:
        raise ValueError(f"internal WRITE: record of {len(rec)} characters exceeds CHARACTER*({length})")
    return rec.ljust(length)


def print_message(message, unit, sq, myThid=1, *, io):
    """PRINT_MESSAGE( message, unit, sq, myThid ) (print.F:23-161) on the oracle's path (module docstring)."""
    if sq in (SQUEEZE_BOTH, SQUEEZE_LEFT):                                     # print.F:80-85
        iStart = ifnblnk(message)
    else:
        iStart = 1
    if sq in (SQUEEZE_BOTH, SQUEEZE_RIGHT):                                    # :86-91
        iEnd = ilnblnk(message)
    else:
        iEnd = len(message)
    if io.numberOfProcs == 0:                                                  # :94-100
        raise NotImplementedError("PRINT_MESSAGE: single-process format (numberOfProcs = 0) is not ported")
    if io.myProcId != 0:                                                       # :101 pidIO .EQ. myProcId (pidIO = 0)
        raise NotImplementedError("PRINT_MESSAGE: myProcId /= pidIO is not ported")
    idString = fortran_write("(I4.4,A,I4.4)", io.myProcId, ".", myThid)       # :112, :120
    iTmp = ilnblnk(idString)                                                   # :121
    if message.strip(" ") == "":                                               # :122 message .EQ. ' '
        io.write(unit, "(" + PROCESS_HEADER + " " + idString[:iTmp] + ")" + " ")                     # :123-124
    else:
        io.write(unit, "(" + PROCESS_HEADER + " " + idString[:iTmp] + ")" + " " + message[iStart - 1:iEnd])  # :126-128
    if unit == ERROR_MESSAGE_UNIT and message.strip(" ") != "":                # :148-157 copy to unit 0
        raise NotImplementedError("PRINT_MESSAGE: error-unit copy to unit 0 is not ported")
