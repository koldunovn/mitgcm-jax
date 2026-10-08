#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax GAD-A replay
C     | harness (reference/replay_gad_a; plan M1 sub-lane GAD-A,
C     | copied from the Task 8 harness reference/replay). Not an
C     | MITgcm routine: it replaces model/src/the_main_loop.F in
C     | a genmake2 build of a verification experiment's code/
C     | dir. THE_MODEL_MAIN has already called INITIALISE_FIXED,
C     | so GRID.h holds the experiment's real grid (land, partial
C     | cells, halos filled by the exchanges). This routine
C     |  1. reads synthetic inputs written by Python
C     |     (replay_in.bin, big-endian stream),
C     |  2. overrides the vertical/metric factors that are 1 in
C     |     the experiment (recip_deepFacC, recip_deepFac2C,
C     |     deepFac2F, recip_deepFac2F, rhoFacF, recip_rhoFacF,
C     |     recip_rhoFacC, cosFacU, cosFacV) with input values,
C     |     so that a missing or swapped factor is visible,
C     |  3. calls the simple flux routines of pkg/generic_advdiff
C     |     (C2, C4, U3, DST3, DST3FL, FLUXLIMIT advective fluxes
C     |     in x, y (and r for C2); DIFF_X/Y/R diffusive fluxes)
C     |     for every tile and level k = 1..Nr, each output first
C     |     set to a Python-given prior (points a routine does not
C     |     write keep it); and the implicit vertical matrix
C     |     routines GAD_U3C4_IMPL_R (advectionScheme = U3) and
C     |     GAD_FLUXLIMIT_IMPL_R for k = 1..Nr accumulating into
C     |     prior-initialised 3-D matrices,
C     |  4. writes every output array, all points incl. halos
C     |     (replay_out.bin) and the grid fields the routines read
C     |     (replay_grid.bin).
C     | No time stepping.
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "GAD.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
C     nIn3 :: 3-D input fields (layout: replay_gad_a/replay_io.py)
C     nOut3:: 3-D output fields
      INTEGER nIn3, nOut3
      PARAMETER ( nIn3 = 17, nOut3 = 28 )
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261002, VERSN = 1 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / REPLAY_GAD_A_BUF / in3, out3
      _RL uTr (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vTr (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL rTr (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uVl (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vVl (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uCf (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vCf (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL trc (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL kapR(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS xAl (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS yAl (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS mskW(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS mskS(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS mskU(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL fx  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL trc3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RS rhF (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL a5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL b5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL c5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL d5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL e5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL a3  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL b3  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL c3  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL dTarg(Nr)
      _RL dtLoc, dKh
      INTEGER hdr(10), ohdr(4)
      INTEGER uIn, uOut, uGrd, ios
      INTEGER bi, bj, i, j, k, km1

      uIn  = 91
      uOut = 92
      uGrd = 93

C--   1. inputs: header int32 (MAGIC, VERSN, sNx, sNy, OLx, OLy,
C     nSx, nSy, Nr, nIn3); real*8 deltaTloc, diffKh; deltaTarg(Nr);
C     recip_deepFacC, recip_deepFac2C, recip_rhoFacC (Nr);
C     deepFac2F, recip_deepFac2F, rhoFacF, recip_rhoFacF (Nr+1);
C     cosFacU, cosFacV (1-OLy:sNy+OLy,nSx,nSy); in3
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
      READ( uIn ) dtLoc, dKh
      READ( uIn ) dTarg
C--   2. overrides (see header)
      READ( uIn ) recip_deepFacC, recip_deepFac2C, recip_rhoFacC
      READ( uIn ) deepFac2F, recip_deepFac2F, rhoFacF, recip_rhoFacF
      READ( uIn ) cosFacU, cosFacV
      READ( uIn ) in3
      CLOSE( uIn )

C--   3. replay
      DO bj = 1, nSy
       DO bi = 1, nSx
C     3-D tracer, recip_hFacC and the prior matrices of the
C     implicit routines (a5..e5: U3C4_IMPL_R; a3..c3: FLUXLIMIT)
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           trc3(i,j,k) = in3(i,j,k,bi,bj, 8)
           rhF (i,j,k) = recip_hFacC(i,j,k,bi,bj)
           a5(i,j,k) = in3(i,j,k,bi,bj,13)
           b5(i,j,k) = in3(i,j,k,bi,bj,14)
           c5(i,j,k) = in3(i,j,k,bi,bj,15)
           d5(i,j,k) = in3(i,j,k,bi,bj,16)
           e5(i,j,k) = in3(i,j,k,bi,bj,17)
           a3(i,j,k) = in3(i,j,k,bi,bj,13)
           b3(i,j,k) = in3(i,j,k,bi,bj,14)
           c3(i,j,k) = in3(i,j,k,bi,bj,15)
          ENDDO
         ENDDO
        ENDDO
        DO k = 1, Nr
         km1 = MAX(1,k-1)
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uTr (i,j) = in3(i,j,k,bi,bj, 1)
           vTr (i,j) = in3(i,j,k,bi,bj, 2)
           rTr (i,j) = in3(i,j,k,bi,bj, 3)
           uVl (i,j) = in3(i,j,k,bi,bj, 4)
           vVl (i,j) = in3(i,j,k,bi,bj, 5)
           uCf (i,j) = in3(i,j,k,bi,bj, 6)
           vCf (i,j) = in3(i,j,k,bi,bj, 7)
           trc (i,j) = in3(i,j,k,bi,bj, 8)
           kapR(i,j) = in3(i,j,k,bi,bj, 9)
           xAl (i,j) = in3(i,j,k,bi,bj,10)
           yAl (i,j) = in3(i,j,k,bi,bj,11)
           mskW(i,j) = maskW(i,j,k,bi,bj)
           mskS(i,j) = maskS(i,j,k,bi,bj)
           mskU(i,j) = maskC(i,j,km1,bi,bj)*maskC(i,j,k,bi,bj)
          ENDDO
         ENDDO
C     out3(.., n): the output of call n below, prior = in3(.., 12)
         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_C2_ADV_X( bi,bj,k, uTr, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj, 1), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_C2_ADV_Y( bi,bj,k, vTr, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj, 2), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_C2_ADV_R( bi,bj,k, rTr, trc3, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj, 3), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_C4_ADV_X( bi,bj,k, uTr, mskW, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj, 4), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_C4_ADV_Y( bi,bj,k, vTr, mskS, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj, 5), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_U3_ADV_X( bi,bj,k, uTr, mskW, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj, 6), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_U3_ADV_Y( bi,bj,k, vTr, mskS, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj, 7), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_DST3FL_ADV_X( bi,bj,k, .TRUE., dtLoc,
     &                          uTr, uVl, mskW, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj, 8), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_DST3FL_ADV_X( bi,bj,k, .FALSE., dtLoc,
     &                          uTr, uCf, mskW, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj, 9), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_DST3FL_ADV_Y( bi,bj,k, .TRUE., dtLoc,
     &                          vTr, vVl, mskS, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,10), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_DST3FL_ADV_Y( bi,bj,k, .FALSE., dtLoc,
     &                          vTr, vCf, mskS, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,11), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_DST3_ADV_X( bi,bj,k, .TRUE., dtLoc,
     &                        uTr, uVl, mskW, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,12), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_DST3_ADV_X( bi,bj,k, .FALSE., dtLoc,
     &                        uTr, uCf, mskW, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,13), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_FLUXLIMIT_ADV_X( bi,bj,k, .TRUE., dtLoc,
     &                             uTr, uVl, mskW, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,14), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_FLUXLIMIT_ADV_X( bi,bj,k, .FALSE., dtLoc,
     &                             uTr, uCf, mskW, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,15), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_FLUXLIMIT_ADV_Y( bi,bj,k, .TRUE., dtLoc,
     &                             vTr, vVl, mskS, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,16), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_FLUXLIMIT_ADV_Y( bi,bj,k, .FALSE., dtLoc,
     &                             vTr, vCf, mskS, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,17), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_DIFF_X( bi,bj,k, xAl, dKh, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,18), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_DIFF_Y( bi,bj,k, yAl, dKh, trc, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,19), fx )

         CALL RGA_COPY( fx, in3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL GAD_DIFF_R( bi,bj,k, mskU, kapR, trc3, fx, myThid )
         CALL RGA_COPY( out3(1-OLx,1-OLy,k,bi,bj,20), fx )

C     implicit vertical matrices, iMin..jMax as gad_implicit_r.F:78-79
         CALL GAD_U3C4_IMPL_R( bi,bj,k, 1,sNx,1,sNy,
     &        ENUM_UPWIND_3RD, dTarg, rTr, rhF,
     &        a5, b5, c5, d5, e5, myThid )
         CALL GAD_FLUXLIMIT_IMPL_R( bi,bj,k, 1,sNx,1,sNy,
     &        dTarg, rTr, rhF, trc3,
     &        a3, b3, c3, myThid )
        ENDDO
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj,21) = a5(i,j,k)
           out3(i,j,k,bi,bj,22) = b5(i,j,k)
           out3(i,j,k,bi,bj,23) = c5(i,j,k)
           out3(i,j,k,bi,bj,24) = d5(i,j,k)
           out3(i,j,k,bi,bj,25) = e5(i,j,k)
           out3(i,j,k,bi,bj,26) = a3(i,j,k)
           out3(i,j,k,bi,bj,27) = b3(i,j,k)
           out3(i,j,k,bi,bj,28) = c3(i,j,k)
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
C     5, 4); 3-D hFacC recip_hFacC maskC maskW maskS; 2-D
C     recip_dxC recip_dyC rA recip_rA; cosFacU cosFacV;
C     (Nr) recip_deepFacC recip_deepFac2C recip_rhoFacC recip_drF;
C     (Nr+1) deepFac2F recip_deepFac2F rhoFacF recip_rhoFacF
C     recip_drC; rkSign
      OPEN( uGrd, FILE='replay_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(3) = 5
      ohdr(4) = 4
      WRITE( uGrd ) ohdr
      WRITE( uGrd ) hFacC, recip_hFacC, maskC, maskW, maskS
      WRITE( uGrd ) recip_dxC, recip_dyC, rA, recip_rA
      WRITE( uGrd ) cosFacU, cosFacV
      WRITE( uGrd ) recip_deepFacC, recip_deepFac2C, recip_rhoFacC,
     &              recip_drF
      WRITE( uGrd ) deepFac2F, recip_deepFac2F, rhoFacF,
     &              recip_rhoFacF, recip_drC
      WRITE( uGrd ) rkSign
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A)')
     &  'REPLAY GAD-A harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed'
      RETURN
      END

CBOP
C     !ROUTINE: RGA_COPY
C     !INTERFACE:
      SUBROUTINE RGA_COPY( a, b )
C     !DESCRIPTION: \bv
C     a(i,j) = b(i,j) on every point of a tile (halos included);
C     replay helper of reference/replay_gad_a.
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
      _RL a(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL b(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
CEOP
      INTEGER i, j
      DO j = 1-OLy, sNy+OLy
       DO i = 1-OLx, sNx+OLx
        a(i,j) = b(i,j)
       ENDDO
      ENDDO
      RETURN
      END
