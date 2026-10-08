#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax MOM replay
C     | harness (reference/replay_mom; M1 sub-lane MOM; a copy of
C     | the Task 8 harness reference/replay). Not an MITgcm
C     | routine: it replaces model/src/the_main_loop.F in a
C     | genmake2 build of a verification experiment's code dir.
C     | THE_MODEL_MAIN has already called INITIALISE_FIXED, so
C     | GRID.h holds the experiment's real grid (land, partial
C     | cells, halos filled by the exchanges). This routine
C     |  1. reads synthetic inputs written by Python
C     |     (replay_mom/replay_io.py, big-endian stream),
C     |  2. overrides PARAMS.h/GRID.h values that are trivial in
C     |     the experiment (cosFac*, deep/rho/rVel2w factors,
C     |     fCoriCos, angleCos/SinC, tanPhiAt*, viscosity and
C     |     drag parameters) with the input values, so that a
C     |     missing or swapped factor shows in the gate,
C     |  3. calls the leaf kernels of pkg/mom_common and
C     |     pkg/mom_fluxform for every tile and level, each with
C     |     the branch selectors (PARAMS.h) set per case, every
C     |     output first set to a Python-given prior field
C     |     (points a routine does not write keep it),
C     |  4. writes every output, all points incl. halos
C     |     (replay_out.bin), and the grid and parameter values
C     |     the routines read (replay_grid.bin).
C     | No time stepping. useDiagnostics, staggerTimeStep and
C     | useGrdchk are set .FALSE. The case list and the file
C     | layout are mirrored by replay_mom/replay_io.py (OUT3).
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
      INTEGER nIn3, nOut3, nSPar
      PARAMETER ( nIn3 = 20, nOut3 = 79, nSPar = 12 )
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261002, VERSN = 1 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / MOMRP_BUF / in3, out3
      _RL uF  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vF  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uTr (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vTr (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL rTr (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS hFZ (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS h0FZ(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL d2u (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL d2v (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vAhD(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vAhZ(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vA4D(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vA4Z(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL KEi (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL kapU(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1)
      _RL kapV(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1)
      _RL wA  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL wB  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS hZ  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS rhZ (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dFacA(Nr)
      _RL spar(nSPar), rAwMax, vA4GMx
      INTEGER hdr(10), ohdr(4), iflg(5)
      INTEGER uIn, uOut, uGrd, ios
      INTEGER bi, bj, i, j, k, kk, iO, iKE, iC, iB
      INTEGER bSel(0:6)
      LOGICAL bNsb(0:6), bPce(0:6), bInp(0:6)
C     botdrag cases b0..b6: selectBotDragQuadr, no_slip_bottom,
C     bottomVisc_pCell, inp_KE
      DATA bSel / -1, -1, -1, 0, 0, 1, 2 /
      DATA bNsb / .TRUE., .TRUE., .FALSE., .FALSE., .FALSE.,
     &            .TRUE., .TRUE. /
      DATA bPce / .FALSE., .TRUE., .FALSE., .FALSE., .FALSE.,
     &            .FALSE., .FALSE. /
      DATA bInp / .TRUE., .TRUE., .TRUE., .TRUE., .FALSE.,
     &            .TRUE., .TRUE. /

      uIn  = 91
      uOut = 92
      uGrd = 93

C--   1. inputs (layout: replay_mom/replay_io.py)
      OPEN( uIn, FILE='replay_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN', IOSTAT=ios )
      IF ( ios .NE. 0 ) STOP 'REPLAY: cannot open replay_in.bin'
      READ( uIn ) hdr
      IF ( hdr(1).NE.MAGIC .OR. hdr(2).NE.VERSN )
     &   STOP 'REPLAY: bad magic/version in replay_in.bin'
      IF ( hdr(3).NE.sNx .OR. hdr(4).NE.sNy .OR. hdr(5).NE.OLx
     &.OR. hdr(6).NE.OLy .OR. hdr(7).NE.nSx .OR. hdr(8).NE.nSy
     &.OR. hdr(9).NE.Nr  .OR. hdr(10).NE.nIn3 )
     &   STOP 'REPLAY: replay_in.bin does not match SIZE.h'
      READ( uIn ) spar
      READ( uIn ) recip_deepFacC, recip_deepFac2C, recip_gravFacC,
     &            dFacA
      READ( uIn ) deepFac2F, rhoFacF, rVel2wUnit
      READ( uIn ) cosFacU, cosFacV, sqCosFacU, sqCosFacV
      READ( uIn ) fCoriCos, angleCosC, angleSinC, tanPhiAtU, tanPhiAtV
      READ( uIn ) in3
      CLOSE( uIn )

C--   2. overrides (see header)
      useDiagnostics  = .FALSE.
      staggerTimeStep = .FALSE.
      useGrdchk       = .FALSE.
      rAwMax = 0. _d 0
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          rAwMax = MAX( rAwMax, rAw(i,j,bi,bj) )
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      sideDragFactor = spar(1)
      viscAh         = spar(2)
      viscAhGrid     = spar(3)
      viscAhMax      = viscAh + spar(4)*viscAhGrid*rAwMax/spar(10)
      viscA4         = spar(5)
      viscA4Grid     = spar(6)
      viscA4Max      = viscA4
     &               + spar(7)*viscA4Grid*rAwMax*rAwMax/spar(10)
      viscA4GridMax  = spar(8)
      viscA4GridMin  = spar(9)
      deltaTMom      = spar(10)
      bottomDragLinear    = spar(11)
      bottomDragQuadratic = spar(12)
      vA4GMx = viscA4GridMax
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           wVel(i,j,k,bi,bj) = in3(i,j,k,bi,bj,3)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   3. replay
      DO bj = 1, nSy
       DO bi = 1, nSx
C     vertical viscosity (1..Nr+1): levels 1..Nr from in3 15/16, level
C     Nr+1 from level 1 of in3 17/18
        DO kk = 1, Nr+1
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           IF ( kk.LE.Nr ) THEN
            kapU(i,j,kk) = in3(i,j,kk,bi,bj,15)
            kapV(i,j,kk) = in3(i,j,kk,bi,bj,16)
           ELSE
            kapU(i,j,kk) = in3(i,j,1,bi,bj,17)
            kapV(i,j,kk) = in3(i,j,1,bi,bj,18)
           ENDIF
          ENDDO
         ENDDO
        ENDDO
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uF  (i,j) = in3(i,j,k,bi,bj, 1)
           vF  (i,j) = in3(i,j,k,bi,bj, 2)
           uTr (i,j) = in3(i,j,k,bi,bj, 4)
           vTr (i,j) = in3(i,j,k,bi,bj, 5)
           rTr (i,j) = in3(i,j,k,bi,bj, 6)
           hFZ (i,j) = in3(i,j,k,bi,bj, 7)
           h0FZ(i,j) = in3(i,j,k,bi,bj, 8)
           d2u (i,j) = in3(i,j,k,bi,bj, 9)
           d2v (i,j) = in3(i,j,k,bi,bj,10)
           vAhD(i,j) = in3(i,j,k,bi,bj,11)
           vAhZ(i,j) = in3(i,j,k,bi,bj,12)
           vA4D(i,j) = in3(i,j,k,bi,bj,13)
           vA4Z(i,j) = in3(i,j,k,bi,bj,14)
           KEi (i,j) = in3(i,j,k,bi,bj,19)
          ENDDO
         ENDDO
         iO = 0

C     out 1-2: MOM_CALC_HFACZ -> hFacZ, r_hFacZ
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           hZ (i,j) = in3(i,j,k,bi,bj,20)
           rhZ(i,j) = in3(i,j,k,bi,bj,20)
          ENDDO
         ENDDO
         CALL MOM_CALC_HFACZ( bi, bj, k, hZ, rhZ, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj,iO+1) = hZ(i,j)
           out3(i,j,k,bi,bj,iO+2) = rhZ(i,j)
          ENDDO
         ENDDO
         iO = iO + 2

C     out 3-7: MOM_CALC_KE, KEscheme = -1..3
         DO iKE = -1, 3
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_CALC_KE( bi, bj, k, iKE, uF, vF, wA, myThid )
          iO = iO + 1
          CALL MOMRP_PUT( wA, bi, bj, k, iO )
         ENDDO

C     out 8-11: MOM_CALC_RTRANS at k and k+1 (select_rStar = 0)
         select_rStar = 0
         DO kk = k, k+1
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOMRP_PRIOR( wB, bi, bj, k )
          CALL MOM_CALC_RTRANS( kk, bi, bj, wA, wB,
     &                          myTime, myIter, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+1 )
          CALL MOMRP_PUT( wB, bi, bj, k, iO+2 )
          iO = iO + 2
         ENDDO

C     out 12: MOM_U_ADV_UU
         CALL MOMRP_PRIOR( wA, bi, bj, k )
         CALL MOM_U_ADV_UU( bi, bj, k, uTr, uF, wA, myThid )
         iO = iO + 1
         CALL MOMRP_PUT( wA, bi, bj, k, iO )
C     out 13-14: MOM_U_ADV_VU, selectMetricTerms = 0, 3
         DO iC = 0, 1
          selectMetricTerms = 3*iC
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_U_ADV_VU( bi, bj, k, vTr, uF, wA, myThid )
          iO = iO + 1
          CALL MOMRP_PUT( wA, bi, bj, k, iO )
         ENDDO
C     out 15-18: MOM_U_ADV_WU cases c0..c3 (see MOMRP_WCASE)
         DO iC = 0, 3
          CALL MOMRP_WCASE( iC, k, kk )
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_U_ADV_WU( bi, bj, kk, dFacA,
     &         in3(1-OLx,1-OLy,1,1,1,1), in3(1-OLx,1-OLy,1,1,1,3),
     &         rTr, wA, myThid )
          iO = iO + 1
          CALL MOMRP_PUT( wA, bi, bj, k, iO )
         ENDDO
         rigidLid = .FALSE.
         select_rStar = 0
C     out 19: MOM_V_ADV_UV ; out 20: MOM_V_ADV_VV
         CALL MOMRP_PRIOR( wA, bi, bj, k )
         CALL MOM_V_ADV_UV( bi, bj, k, uTr, vF, wA, myThid )
         iO = iO + 1
         CALL MOMRP_PUT( wA, bi, bj, k, iO )
         CALL MOMRP_PRIOR( wA, bi, bj, k )
         CALL MOM_V_ADV_VV( bi, bj, k, vTr, vF, wA, myThid )
         iO = iO + 1
         CALL MOMRP_PUT( wA, bi, bj, k, iO )
C     out 21-24: MOM_V_ADV_WV cases c0..c3
         DO iC = 0, 3
          CALL MOMRP_WCASE( iC, k, kk )
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_V_ADV_WV( bi, bj, kk, dFacA,
     &         in3(1-OLx,1-OLy,1,1,1,2), in3(1-OLx,1-OLy,1,1,1,3),
     &         rTr, wA, myThid )
          iO = iO + 1
          CALL MOMRP_PUT( wA, bi, bj, k, iO )
         ENDDO
         rigidLid = .FALSE.
         select_rStar = 0

C     out 25-29: MOM_U_CORIOLIS, out 30-34: MOM_V_CORIOLIS,
C     selectCoriScheme = 0..4
         DO iC = 0, 4
          selectCoriScheme = iC
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_U_CORIOLIS( bi, bj, k, vF, wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+1+iC )
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_V_CORIOLIS( bi, bj, k, uF, wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+6+iC )
         ENDDO
         iO = iO + 10

C     out 35-36: MOM_U_METRIC_SPHERE, out 37-38: MOM_V_METRIC_SPHERE,
C     selectMetricTerms = 1, 2
         DO iC = 1, 2
          selectMetricTerms = iC
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_U_METRIC_SPHERE( bi, bj, k, uF, vF, wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+iC )
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_V_METRIC_SPHERE( bi, bj, k, uF, wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+2+iC )
         ENDDO
         iO = iO + 4

C     out 39-42: MOM_U_XVISCFLUX, MOM_U_YVISCFLUX, MOM_V_XVISCFLUX,
C     MOM_V_YVISCFLUX
         CALL MOMRP_PRIOR( wA, bi, bj, k )
         CALL MOM_U_XVISCFLUX( bi, bj, k, uF, d2u, wA,
     &                         vAhD, vA4D, myThid )
         CALL MOMRP_PUT( wA, bi, bj, k, iO+1 )
         CALL MOMRP_PRIOR( wA, bi, bj, k )
         CALL MOM_U_YVISCFLUX( bi, bj, k, uF, d2u, hFZ, wA,
     &                         vAhZ, vA4Z, myThid )
         CALL MOMRP_PUT( wA, bi, bj, k, iO+2 )
         CALL MOMRP_PRIOR( wA, bi, bj, k )
         CALL MOM_V_XVISCFLUX( bi, bj, k, vF, d2v, hFZ, wA,
     &                         vAhZ, vA4Z, myThid )
         CALL MOMRP_PUT( wA, bi, bj, k, iO+3 )
         CALL MOMRP_PRIOR( wA, bi, bj, k )
         CALL MOM_V_YVISCFLUX( bi, bj, k, vF, d2v, wA,
     &                         vAhD, vA4D, myThid )
         CALL MOMRP_PUT( wA, bi, bj, k, iO+4 )
         iO = iO + 4

C     out 43-44: MOM_U_RVISCFLUX at k, k+1; out 45-46: MOM_V_RVISCFLUX
         DO iC = 0, 1
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_U_RVISCFLUX( bi, bj, k+iC,
     &         in3(1-OLx,1-OLy,1,1,1,1), kapU, wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+1+iC )
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_V_RVISCFLUX( bi, bj, k+iC,
     &         in3(1-OLx,1-OLy,1,1,1,2), kapV, wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+3+iC )
         ENDDO
         iO = iO + 4

C     out 47-48: MOM_U_DEL2U, out 49-50: MOM_V_DEL2V,
C     no_slip_sides = F, T
         DO iC = 0, 1
          no_slip_sides = iC.EQ.1
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_U_DEL2U( bi, bj, k, uF, hFZ, h0FZ, wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+1+iC )
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_V_DEL2V( bi, bj, k, vF, hFZ, h0FZ, wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+3+iC )
         ENDDO
         iO = iO + 4

C     out 51-53: MOM_U_SIDEDRAG, out 54-56: MOM_V_SIDEDRAG:
C     sideDragFactor = spar(1) (> 0); = -1 (old branch); = -1 with
C     viscA4GridMax = 0 (the LOGICAL arguments harmonic, biharmonic,
C     useVariableViscosity are not read by the routines)
         DO iC = 0, 2
          IF ( iC.EQ.0 ) THEN
           sideDragFactor = spar(1)
           viscA4GridMax  = vA4GMx
          ELSEIF ( iC.EQ.1 ) THEN
           sideDragFactor = -1. _d 0
           viscA4GridMax  = vA4GMx
          ELSE
           sideDragFactor = -1. _d 0
           viscA4GridMax  = 0. _d 0
          ENDIF
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_U_SIDEDRAG( bi, bj, k, uF, d2u, hFZ, vAhZ, vA4Z,
     &         .TRUE., .TRUE., .FALSE.,
     &         wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+1+iC )
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_V_SIDEDRAG( bi, bj, k, vF, d2v, hFZ, vAhZ, vA4Z,
     &         .TRUE., .TRUE., .FALSE.,
     &         wA, myThid )
          CALL MOMRP_PUT( wA, bi, bj, k, iO+4+iC )
         ENDDO
         sideDragFactor = spar(1)
         viscA4GridMax  = vA4GMx
         iO = iO + 6

C     out 57-64: MOM_U_BOTDRAG_COEFF cases b0..b6 (cDrag) + KE of b4;
C     out 65-72: MOM_V_BOTDRAG_COEFF the same
         DO iC = 0, 1
          DO iB = 0, 6
           selectBotDragQuadr = bSel(iB)
           no_slip_bottom     = bNsb(iB)
           bottomVisc_pCell   = bPce(iB)
           CALL MOMRP_PRIOR( wA, bi, bj, k )
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             wB(i,j) = KEi(i,j)
            ENDDO
           ENDDO
           IF ( iC.EQ.0 ) THEN
            CALL MOM_U_BOTDRAG_COEFF( bi, bj, k, bInp(iB),
     &           uF, vF, kapU, wB, wA, myIter, myThid )
           ELSE
            CALL MOM_V_BOTDRAG_COEFF( bi, bj, k, bInp(iB),
     &           uF, vF, kapV, wB, wA, myIter, myThid )
           ENDIF
           CALL MOMRP_PUT( wA, bi, bj, k, iO+1+iB )
           IF ( iB.EQ.4 ) CALL MOMRP_PUT( wB, bi, bj, k, iO+8 )
          ENDDO
          iO = iO + 8
         ENDDO

C     out 73-74: MOM_U_CORIOLIS_NH, select3dCoriScheme = 1, 2
         DO iC = 1, 2
          select3dCoriScheme = iC
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_U_CORIOLIS_NH( bi, bj, k,
     &         in3(1-OLx,1-OLy,1,1,1,3), wA, myThid )
          iO = iO + 1
          CALL MOMRP_PUT( wA, bi, bj, k, iO )
         ENDDO
C     out 75: MOM_U_METRIC_NH ; out 76: MOM_V_METRIC_NH
         CALL MOMRP_PRIOR( wA, bi, bj, k )
         CALL MOM_U_METRIC_NH( bi, bj, k, uF,
     &        in3(1-OLx,1-OLy,1,1,1,3), wA, myThid )
         iO = iO + 1
         CALL MOMRP_PUT( wA, bi, bj, k, iO )
         CALL MOMRP_PRIOR( wA, bi, bj, k )
         CALL MOM_V_METRIC_NH( bi, bj, k, vF,
     &        in3(1-OLx,1-OLy,1,1,1,3), wA, myThid )
         iO = iO + 1
         CALL MOMRP_PUT( wA, bi, bj, k, iO )

C     out 77-79: MOM_QUASIHYDROSTATIC, (select3dCoriScheme,
C     useNHMTerms) = (1,F), (0,T), (2,T)
         DO iC = 1, 3
          IF ( iC.EQ.1 ) THEN
           select3dCoriScheme = 1
           useNHMTerms = .FALSE.
          ELSEIF ( iC.EQ.2 ) THEN
           select3dCoriScheme = 0
           useNHMTerms = .TRUE.
          ELSE
           select3dCoriScheme = 2
           useNHMTerms = .TRUE.
          ENDIF
          CALL MOMRP_PRIOR( wA, bi, bj, k )
          CALL MOM_QUASIHYDROSTATIC( bi, bj, k,
     &         in3(1-OLx,1-OLy,1,1,1,1), in3(1-OLx,1-OLy,1,1,1,2),
     &         wA, myTime, myIter, myThid )
          iO = iO + 1
          CALL MOMRP_PUT( wA, bi, bj, k, iO )
         ENDDO

         IF ( iO.NE.nOut3 ) STOP 'REPLAY: output count mismatch'
        ENDDO
       ENDDO
      ENDDO

C--   4a. outputs: header int32 (MAGIC, VERSN, nOut3, Nr); out3
      OPEN( uOut, FILE='replay_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(1) = MAGIC
      ohdr(2) = VERSN
      ohdr(3) = nOut3
      ohdr(4) = Nr
      WRITE( uOut ) ohdr
      WRITE( uOut ) out3
      CLOSE( uOut )

C--   4b. grid and parameters as the routines read them (order:
C     replay_mom/replay_io.py read_grid)
      OPEN( uGrd, FILE='replay_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(3) = 11
      ohdr(4) = 26
      WRITE( uGrd ) ohdr
      WRITE( uGrd ) hFacC, hFacW, hFacS, recip_hFacC, recip_hFacW,
     &              recip_hFacS, maskC, maskW, maskS
#ifdef NONLIN_FRSURF
      WRITE( uGrd ) h0FacW, h0FacS
#else
      WRITE( uGrd ) hFacW, hFacS
#endif
      WRITE( uGrd ) rA, rAw, rAs, recip_rAw, recip_rAs, recip_rAz,
     &              dxC, dxG, dyG, dxF, dyF, dxV, dyU,
     &              recip_dxC, recip_dyC, recip_dxF, recip_dyF,
     &              recip_dxV, recip_dyU,
     &              fCori, fCoriCos, angleCosC, angleSinC,
     &              tanPhiAtU, tanPhiAtV, recip_rA
      WRITE( uGrd ) cosFacU, cosFacV, sqCosFacU, sqCosFacV
      WRITE( uGrd ) drF, recip_drF, recip_deepFacC, recip_deepFac2C,
     &              recip_gravFacC, dFacA
      WRITE( uGrd ) recip_drC, deepFac2F, rhoFacF, rVel2wUnit
      WRITE( uGrd ) rkSign, gravitySign, recip_rSphere, rhoConst,
     &              mass2rUnit, recip_gravity, sideDragFactor,
     &              viscAh, viscAhGrid, viscAhMax, viscA4, viscA4Grid,
     &              viscA4Max, viscA4GridMax, viscA4GridMin, deltaTMom,
     &              bottomDragLinear, bottomDragQuadratic
      iflg(1) = 0
      iflg(2) = 0
      iflg(3) = 0
      iflg(4) = 0
      iflg(5) = 0
      IF ( useRealFreshWaterFlux ) iflg(1) = 1
      IF ( usingPCoords )          iflg(2) = 1
      IF ( usingZCoords )          iflg(3) = 1
      IF ( fluidIsWater )          iflg(4) = 1
#ifdef NONLIN_FRSURF
      iflg(5) = 1
#endif
      WRITE( uGrd ) iflg
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A,I4,A)')
     &  'REPLAY MOM harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed,', nOut3, ' outputs'
      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: MOMRP_PRIOR
C     !INTERFACE:
      SUBROUTINE MOMRP_PRIOR( a, bi, bj, k )
C     !DESCRIPTION: a(:,:) = prior field (in3 field 20, level k)
      IMPLICIT NONE
#include "SIZE.h"
      INTEGER nIn3, nOut3
      PARAMETER ( nIn3 = 20, nOut3 = 79 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / MOMRP_BUF / in3, out3
      _RL a(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER bi, bj, k
CEOP
      INTEGER i, j
      DO j = 1-OLy, sNy+OLy
       DO i = 1-OLx, sNx+OLx
        a(i,j) = in3(i,j,k,bi,bj,20)
       ENDDO
      ENDDO
      RETURN
      END

CBOP
C     !ROUTINE: MOMRP_PUT
C     !INTERFACE:
      SUBROUTINE MOMRP_PUT( a, bi, bj, k, iO )
C     !DESCRIPTION: out3(:,:,k,bi,bj,iO) = a(:,:)
      IMPLICIT NONE
#include "SIZE.h"
      INTEGER nIn3, nOut3
      PARAMETER ( nIn3 = 20, nOut3 = 79 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / MOMRP_BUF / in3, out3
      _RL a(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER bi, bj, k, iO
CEOP
      INTEGER i, j
      IF ( iO.LT.1 .OR. iO.GT.nOut3 ) STOP 'REPLAY: bad output index'
      DO j = 1-OLy, sNy+OLy
       DO i = 1-OLx, sNx+OLx
        out3(i,j,k,bi,bj,iO) = a(i,j)
       ENDDO
      ENDDO
      RETURN
      END

CBOP
C     !ROUTINE: MOMRP_WCASE
C     !INTERFACE:
      SUBROUTINE MOMRP_WCASE( iC, k, kk )
C     !DESCRIPTION: selectors of the MOM_U_ADV_WU / MOM_V_ADV_WV
C     cases: c0 (k, rigidLid F, select_rStar 0), c1 (k, rigidLid T),
C     c2 (k, select_rStar 1), c3 (k+1, rigidLid F, select_rStar 0)
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
      INTEGER iC, k, kk
CEOP
      kk = k
      rigidLid = .FALSE.
      select_rStar = 0
      IF ( iC.EQ.1 ) rigidLid = .TRUE.
      IF ( iC.EQ.2 ) select_rStar = 1
      IF ( iC.EQ.3 ) kk = k + 1
      RETURN
      END
