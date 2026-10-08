#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax GAD-C replay
C     | harness (reference/replay_gad_c; a copy of the Task 8
C     | harness reference/replay). Not an MITgcm routine: it
C     | replaces model/src/the_main_loop.F in a genmake2 build of
C     | advect_xz/code. THE_MODEL_MAIN has already called
C     | INITIALISE_FIXED, so GRID.h holds the experiment's real
C     | grid (maskC with the slope's land, exchanged halos). This
C     | routine
C     |  1. reads synthetic inputs written by Python
C     |     (replay_in.bin, big-endian stream),
C     |  2. overrides the metric factors that are uniform in the
C     |     experiment (dxF, dyF, recip_dxF, recip_dyF,
C     |     recip_dxC, recip_dyC, drF, recip_drF, recip_drC,
C     |     recip_deepFacC) with the input values, so that a
C     |     missing or swapped factor is visible in the gate,
C     |  3. calls GAD_PPM_ADV_X/Y (meth 40,41,42) and
C     |     GAD_PQM_ADV_X/Y (meth 50,51,52), each with calc_CFL
C     |     = T (velocity) and F (CFL number given), for every
C     |     tile and level, and GAD_PPM_ADV_R / GAD_PQM_ADV_R
C     |     for every tile and method, each output array first
C     |     set to a Python-given prior field,
C     |  4. writes every output array, all points incl. halos
C     |     (replay_out.bin), and the grid fields the routines
C     |     read (replay_grid.bin).
C     | No time stepping.
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
C     nIn3 :: 3-D input fields (layout: reference/replay_gad_c/replay_io.py)
C     nOut3:: 3-D output fields
      INTEGER nIn3, nOut3
      PARAMETER ( nIn3 = 10, nOut3 = 30 )
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261002, VERSN = 1 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / REPLAY_GADC_BUF / in3, out3
      _RL uV (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uC (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uT (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vV (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vC (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vT (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL trc(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL flx(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL wV3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL wT3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL trc3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL flx3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL dtLoc, dtR(Nr)
      INTEGER hdr(10), ohdr(4)
      INTEGER uIn, uOut, uGrd, ios
      INTEGER bi, bj, i, j, k, iM, iC, iD, n, meth
      LOGICAL calc

      uIn  = 91
      uOut = 92
      uGrd = 93

C--   1. inputs: header int32 (MAGIC, VERSN, sNx, sNy, OLx, OLy,
C     nSx, nSy, Nr, nIn3); real*8 deltaT (x,y), dtR(Nr) (r);
C     recip_deepFacC(Nr), drF(Nr), recip_drF(Nr), recip_drC(Nr+1);
C     dxF, dyF, recip_dxF, recip_dyF, recip_dxC, recip_dyC
C     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy); in3
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
      READ( uIn ) dtLoc, dtR
      READ( uIn ) recip_deepFacC, drF, recip_drF, recip_drC
      READ( uIn ) dxF, dyF, recip_dxF, recip_dyF, recip_dxC, recip_dyC
      READ( uIn ) in3
      CLOSE( uIn )

C--   3. replay: in3 fields 1 uVel, 2 uCFL, 3 uTrans, 4 vVel, 5 vCFL,
C     6 vTrans, 7 wVel, 8 wTrans, 9 tracer, 10 flux prior
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uV (i,j) = in3(i,j,k,bi,bj, 1)
           uC (i,j) = in3(i,j,k,bi,bj, 2)
           uT (i,j) = in3(i,j,k,bi,bj, 3)
           vV (i,j) = in3(i,j,k,bi,bj, 4)
           vC (i,j) = in3(i,j,k,bi,bj, 5)
           vT (i,j) = in3(i,j,k,bi,bj, 6)
           trc(i,j) = in3(i,j,k,bi,bj, 9)
          ENDDO
         ENDDO
C     out3 index n = 1 + 2*(iD*3 + iM-1) + (iC-1): iD = 0 PPM_X,
C     1 PQM_X, 2 PPM_Y, 3 PQM_Y; iM = 1..3 (meth 40+iM-1 or 50+iM-1);
C     iC = 1 calc_CFL=T (velocity), 2 calc_CFL=F (CFL number)
         DO iD = 0, 3
          DO iM = 1, 3
           DO iC = 1, 2
            calc = iC .EQ. 1
            n = 1 + 2*(iD*3 + iM-1) + (iC-1)
            DO j = 1-OLy, sNy+OLy
             DO i = 1-OLx, sNx+OLx
              flx(i,j) = in3(i,j,k,bi,bj,10)
             ENDDO
            ENDDO
            IF ( iD.EQ.0 ) THEN
             meth = 39 + iM
             IF ( calc ) THEN
              CALL GAD_PPM_ADV_X( meth, bi, bj, k, .TRUE., dtLoc,
     &                            uV, uT, trc, flx, myThid )
             ELSE
              CALL GAD_PPM_ADV_X( meth, bi, bj, k, .FALSE., dtLoc,
     &                            uC, uT, trc, flx, myThid )
             ENDIF
            ELSEIF ( iD.EQ.1 ) THEN
             meth = 49 + iM
             IF ( calc ) THEN
              CALL GAD_PQM_ADV_X( meth, bi, bj, k, .TRUE., dtLoc,
     &                            uV, uT, trc, flx, myThid )
             ELSE
              CALL GAD_PQM_ADV_X( meth, bi, bj, k, .FALSE., dtLoc,
     &                            uC, uT, trc, flx, myThid )
             ENDIF
            ELSEIF ( iD.EQ.2 ) THEN
             meth = 39 + iM
             IF ( calc ) THEN
              CALL GAD_PPM_ADV_Y( meth, bi, bj, k, .TRUE., dtLoc,
     &                            vV, vT, trc, flx, myThid )
             ELSE
              CALL GAD_PPM_ADV_Y( meth, bi, bj, k, .FALSE., dtLoc,
     &                            vC, vT, trc, flx, myThid )
             ENDIF
            ELSE
             meth = 49 + iM
             IF ( calc ) THEN
              CALL GAD_PQM_ADV_Y( meth, bi, bj, k, .TRUE., dtLoc,
     &                            vV, vT, trc, flx, myThid )
             ELSE
              CALL GAD_PQM_ADV_Y( meth, bi, bj, k, .FALSE., dtLoc,
     &                            vC, vT, trc, flx, myThid )
             ENDIF
            ENDIF
            DO j = 1-OLy, sNy+OLy
             DO i = 1-OLx, sNx+OLx
              out3(i,j,k,bi,bj,n) = flx(i,j)
             ENDDO
            ENDDO
           ENDDO
          ENDDO
         ENDDO
        ENDDO
C     R: out3 index 25..27 GAD_PPM_ADV_R meth 40..42, 28..30
C     GAD_PQM_ADV_R meth 50..52
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           wV3 (i,j,k) = in3(i,j,k,bi,bj, 7)
           wT3 (i,j,k) = in3(i,j,k,bi,bj, 8)
           trc3(i,j,k) = in3(i,j,k,bi,bj, 9)
          ENDDO
         ENDDO
        ENDDO
        DO iD = 0, 1
         DO iM = 1, 3
          n = 25 + iD*3 + iM-1
          DO k = 1, Nr
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             flx3(i,j,k) = in3(i,j,k,bi,bj,10)
            ENDDO
           ENDDO
          ENDDO
          IF ( iD.EQ.0 ) THEN
           CALL GAD_PPM_ADV_R( 39+iM, bi, bj,
     &                         dtR, wV3, wT3, trc3, flx3, myThid )
          ELSE
           CALL GAD_PQM_ADV_R( 49+iM, bi, bj,
     &                         dtR, wV3, wT3, trc3, flx3, myThid )
          ENDIF
          DO k = 1, Nr
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             out3(i,j,k,bi,bj,n) = flx3(i,j,k)
            ENDDO
           ENDDO
          ENDDO
         ENDDO
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

C--   4b. grid as the routines read it: header int32 (MAGIC, VERSN,
C     1, 6); maskC; dxF dyF recip_dxF recip_dyF recip_dxC recip_dyC;
C     recip_deepFacC, drF, recip_drF, recip_drC
      OPEN( uGrd, FILE='replay_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(3) = 1
      ohdr(4) = 6
      WRITE( uGrd ) ohdr
      WRITE( uGrd ) maskC
      WRITE( uGrd ) dxF, dyF, recip_dxF, recip_dyF, recip_dxC, recip_dyC
      WRITE( uGrd ) recip_deepFacC, drF, recip_drF, recip_drC
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A,I4,A)')
     &  'REPLAY harness GAD-C: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed,', nOut3, ' outputs'
      RETURN
      END
