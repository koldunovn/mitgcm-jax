#!/usr/bin/env python3
"""Generate the gfortran replay of GAD_ADVECTION's cubed-sphere pass structure (plan Task 22 item (c)).

    make_program.py UPSTREAM_ROOT OUT.F

The program's statements that decide the passes are copied verbatim, by line number, from the pinned
pkg/generic_advdiff/gad_advection.F (each copied line's text is checked; a different file is refused): the pass
flags (:347-370) and the conditions around the FILL_CS_CORNER_TR_RL calls and the flux/update blocks (:384, :389,
:392, :459, :605, :610, :613, :680). The CALLs are replaced by WRITEs of the event. The driver loops over facets
1..6, the 16 combinations of the N/S/E/W edge flags and the three passes, with useCubedSphereExchange = .TRUE.
Output lines: `P nCFace iE ipass overlapOnly interiorOnly calc_fluxes_X calc_fluxes_Y` then `E <event>` lines;
iE bits: 1 N_edge, 2 S_edge, 4 E_edge, 8 W_edge.
"""

import sys
from pathlib import Path

SRC = "pkg/generic_advdiff/gad_advection.F"
EXPECT = {
    347: "        interiorOnly = .FALSE.",
    348: "        overlapOnly  = .FALSE.",
    370: "        ENDIF",
    384: "        IF (calc_fluxes_X) THEN",
    389: "         IF ( .NOT.overlapOnly .OR. N_edge .OR. S_edge ) THEN",
    392: "          IF ( overlapOnly ) THEN",
    459: "          IF ( overlapOnly .AND. ipass.EQ.1 ) THEN",
    605: "        IF (calc_fluxes_Y) THEN",
    610: "         IF ( .NOT.overlapOnly .OR. E_edge .OR. W_edge ) THEN",
    613: "          IF ( overlapOnly ) THEN",
    680: "          IF ( overlapOnly .AND. ipass.EQ.1 ) THEN",
}


def main(root, out):
    lines = (Path(root) / SRC).read_text().split("\n")
    L = lambda n: lines[n - 1]                                              # noqa: E731
    for n, text in EXPECT.items():
        if L(n).rstrip() != text:
            raise SystemExit(f"{SRC}:{n} is {L(n)!r}, expected {text!r}")
    W = "           WRITE(*,'(A)') '{}'"
    prog = ["      PROGRAM GADPASS",
            "      IMPLICIT NONE",
            "      LOGICAL useCubedSphereExchange, calc_fluxes_X, calc_fluxes_Y",
            "      LOGICAL interiorOnly, overlapOnly",
            "      LOGICAL N_edge, S_edge, E_edge, W_edge",
            "      INTEGER nCFace, ipass, npass, iE",
            "      useCubedSphereExchange = .TRUE.",
            "      npass = 3",
            "      DO nCFace = 1, 6",
            "      DO iE = 0, 15",
            "       N_edge = MOD(iE,2).EQ.1",
            "       S_edge = MOD(iE/2,2).EQ.1",
            "       E_edge = MOD(iE/4,2).EQ.1",
            "       W_edge = MOD(iE/8,2).EQ.1",
            "       DO ipass=1,npass"]
    prog += [L(n) for n in range(347, 371)]
    prog += ["        WRITE(*,'(A,3I3,4L2)') 'P', nCFace, iE, ipass,",
             "     &    overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y",
             L(384), L(389), L(392), W.format("E fill 1"), "          ENDIF", W.format("E flux X"),
             L(459), W.format("E fill 2"), "          ENDIF", "         ENDIF", W.format("E update X"),
             "        ENDIF",
             L(605), L(610), L(613), W.format("E fill 2"), "          ENDIF", W.format("E flux Y"),
             L(680), W.format("E fill 1"), "          ENDIF", "         ENDIF", W.format("E update Y"),
             "        ENDIF",
             "       ENDDO", "      ENDDO", "      ENDDO", "      END", ""]
    Path(out).write_text("\n".join(prog))


if __name__ == "__main__":
    main(*sys.argv[1:3])
