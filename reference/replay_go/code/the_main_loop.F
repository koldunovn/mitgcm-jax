#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax GO replay
C     | harness (reference/replay_go; M1 lane GO, copied from the
C     | GAD-A harness reference/replay_gad_a). Not an MITgcm
C     | routine: it replaces model/src/the_main_loop.F in a
C     | genmake2 build of a verification experiment's code/ dir.
C     | THE_MODEL_MAIN has already called INITIALISE_FIXED, so
C     | GRID.h holds the experiment's real grid. This routine
C     |  1. reads synthetic inputs written by Python
C     |     (replay_in.bin, big-endian stream),
C     |  2. overrides recip_deepFacC, recip_deepFac2C,
C     |     recip_rhoFacC, deepFac2F, recip_deepFac2F, rhoFacF,
C     |     recip_rhoFacF (1 in the experiment) with input values,
C     |  3. calls GAD_DST2U1_IMPL_R for k = 1..Nr with
C     |     advectionScheme = ENUM_UPWIND_1RST and = ENUM_DST2,
C     |     each accumulating into prior-initialised 3-D matrices,
C     |     and SOLVE_PENTADIAGONAL on synthetic 5-diagonal
C     |     systems (errCode = -1 on entry, as GAD_IMPLICIT_R),
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
C     nIn3 :: 3-D input fields (layout: replay_go/replay_io.py)
C     nOut3:: 3-D output fields
      INTEGER nIn3, nOut3
      PARAMETER ( nIn3 = 10, nOut3 = 8 )
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261003, VERSN = 1 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / REPLAY_GO_BUF / in3, out3
      _RL rTr (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS rhF (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL au  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL bu  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL cu  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL ad  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL bd  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL cd  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL a5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL b5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL c5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL d5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL e5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL y5  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL dTarg(Nr)
      INTEGER hdr(10), ohdr(4)
      INTEGER uIn, uOut, uGrd, ios
      INTEGER bi, bj, i, j, k, errCode

      uIn  = 91
      uOut = 92
      uGrd = 93

C--   1. inputs: header int32 (MAGIC, VERSN, sNx, sNy, OLx, OLy,
C     nSx, nSy, Nr, nIn3); real*8 deltaTarg(Nr);
C     recip_deepFacC, recip_deepFac2C, recip_rhoFacC (Nr);
C     deepFac2F, recip_deepFac2F, rhoFacF, recip_rhoFacF (Nr+1); in3
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
      READ( uIn ) dTarg
C--   2. overrides (see header)
      READ( uIn ) recip_deepFacC, recip_deepFac2C, recip_rhoFacC
      READ( uIn ) deepFac2F, recip_deepFac2F, rhoFacF, recip_rhoFacF
      READ( uIn ) in3
      CLOSE( uIn )

C--   3. replay
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           rhF(i,j,k) = recip_hFacC(i,j,k,bi,bj)
           au(i,j,k) = in3(i,j,k,bi,bj, 2)
           bu(i,j,k) = in3(i,j,k,bi,bj, 3)
           cu(i,j,k) = in3(i,j,k,bi,bj, 4)
           ad(i,j,k) = in3(i,j,k,bi,bj, 2)
           bd(i,j,k) = in3(i,j,k,bi,bj, 3)
           cd(i,j,k) = in3(i,j,k,bi,bj, 4)
           a5(i,j,k) = in3(i,j,k,bi,bj, 5)
           b5(i,j,k) = in3(i,j,k,bi,bj, 6)
           c5(i,j,k) = in3(i,j,k,bi,bj, 7)
           d5(i,j,k) = in3(i,j,k,bi,bj, 8)
           e5(i,j,k) = in3(i,j,k,bi,bj, 9)
           y5(i,j,k) = in3(i,j,k,bi,bj,10)
          ENDDO
         ENDDO
        ENDDO
C     implicit vertical matrices, iMin..jMax as gad_implicit_r.F:78-79
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           rTr(i,j) = in3(i,j,k,bi,bj, 1)
          ENDDO
         ENDDO
         CALL GAD_DST2U1_IMPL_R( bi,bj,k, 1,sNx,1,sNy,
     &        ENUM_UPWIND_1RST, dTarg, rTr, rhF,
     &        au, bu, cu, myThid )
         CALL GAD_DST2U1_IMPL_R( bi,bj,k, 1,sNx,1,sNy,
     &        ENUM_DST2, dTarg, rTr, rhF,
     &        ad, bd, cd, myThid )
        ENDDO
C     penta-diagonal solve, errCode = -1 on entry (gad_implicit_r.F:274)
        errCode = -1
        CALL SOLVE_PENTADIAGONAL( 1,sNx, 1,sNy,
     &        a5, b5, c5, d5, e5,
     &        y5, errCode, bi, bj, myThid )
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj,1) = au(i,j,k)
           out3(i,j,k,bi,bj,2) = bu(i,j,k)
           out3(i,j,k,bi,bj,3) = cu(i,j,k)
           out3(i,j,k,bi,bj,4) = ad(i,j,k)
           out3(i,j,k,bi,bj,5) = bd(i,j,k)
           out3(i,j,k,bi,bj,6) = cd(i,j,k)
           out3(i,j,k,bi,bj,7) = y5(i,j,k)
           out3(i,j,k,bi,bj,8) = DBLE(errCode)
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
C     1, 1); 3-D recip_hFacC; 2-D recip_rA; (Nr) recip_deepFac2C
C     recip_rhoFacC recip_drF; rkSign
      OPEN( uGrd, FILE='replay_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(3) = 1
      ohdr(4) = 1
      WRITE( uGrd ) ohdr
      WRITE( uGrd ) recip_hFacC
      WRITE( uGrd ) recip_rA
      WRITE( uGrd ) recip_deepFac2C, recip_rhoFacC, recip_drF
      WRITE( uGrd ) rkSign
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A)')
     &  'REPLAY GO harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed'
      RETURN
      END
