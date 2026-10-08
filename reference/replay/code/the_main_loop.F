#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax one-routine replay
C     | harness (reference/replay; plan Task 8). Not an MITgcm
C     | routine: it replaces model/src/the_main_loop.F in a
C     | genmake2 build of a verification experiment's code/ dir.
C     | THE_MODEL_MAIN has already called INITIALISE_FIXED, so
C     | GRID.h holds the experiment's real grid (land, partial
C     | cells, halos filled by the exchanges). This routine
C     |  1. reads synthetic inputs written by Python
C     |     (replay_in.bin, big-endian stream),
C     |  2. overrides the grid/parameter values that are trivial
C     |     in the experiment (cosFacU, cosFacV, recip_deepFacC =
C     |     1; viscAhD = viscAhZ) with the input values, so that
C     |     a missing or swapped factor is visible in the gate,
C     |  3. calls MOM_CALC_KE (KEscheme -1..3), GAD_DST3_ADV_X
C     |     (calcCFL T and F) and MOM_VI_HDISSIP (all 8 values of
C     |     harmonic, biharmonic, useVariableViscosity) for every
C     |     tile and level, each output array first set to a
C     |     Python-given prior field (points the routine does not
C     |     write keep it),
C     |  4. writes every output array, all points incl. halos
C     |     (replay_out.bin), the grid fields the routines read
C     |     (replay_grid.bin), and the float64 divide probe
C     |     (divide_in.bin -> divide_out.bin).
C     | No time stepping. useDiagnostics is set .FALSE. (pkg
C     | diagnostics is not ported; its FILL calls do not change
C     | the outputs).
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
C     nIn3 :: 3-D input fields (layout: reference/replay/replay_io.py)
C     nOut3:: 3-D output fields
      INTEGER nIn3, nOut3, nDvQ
      PARAMETER ( nIn3 = 19, nOut3 = 23, nDvQ = 5 )
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261001, VERSN = 1 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / REPLAY_BUF / in3, out3
      _RL uF (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vF (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL KE (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uTr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uVl(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL cfl(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL trc(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uT (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS mskW(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL hDv(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vrt(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dSt(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL zSt(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS hFZ(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vAhZ(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vAhD(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vA4Z(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vA4D(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uD (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vD (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dtLoc, par(4)
      Real*8 dvA, dvB, dvQ(nDvQ)
      INTEGER hdr(10), ohdr(4), dhdr(3)
      INTEGER uIn, uOut, uGrd, uDvI, uDvO, ios
      INTEGER bi, bj, i, j, k, iKE, iCmb, n, nDv
      LOGICAL harm, bihr, useVV

      uIn  = 91
      uOut = 92
      uGrd = 93
      uDvI = 94
      uDvO = 95

C--   1. inputs: header int32 (MAGIC, VERSN, sNx, sNy, OLx, OLy,
C     nSx, nSy, Nr, nIn3); real*8 deltaTloc, viscAhD, viscAhZ,
C     viscA4D, viscA4Z; recip_deepFacC(Nr); cosFacU, cosFacV
C     (1-OLy:sNy+OLy,nSx,nSy); in3
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
      READ( uIn ) dtLoc, par
      READ( uIn ) recip_deepFacC
      READ( uIn ) cosFacU, cosFacV
      READ( uIn ) in3
      CLOSE( uIn )

C--   2. overrides (see header)
      viscAhD = par(1)
      viscAhZ = par(2)
      viscA4D = par(3)
      viscA4Z = par(4)
      useDiagnostics = .FALSE.

C--   3. replay
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uF (i,j) = in3(i,j,k,bi,bj, 1)
           vF (i,j) = in3(i,j,k,bi,bj, 2)
           uTr(i,j) = in3(i,j,k,bi,bj, 4)
           uVl(i,j) = in3(i,j,k,bi,bj, 5)
           cfl(i,j) = in3(i,j,k,bi,bj, 6)
           trc(i,j) = in3(i,j,k,bi,bj, 7)
           hDv(i,j) = in3(i,j,k,bi,bj, 9)
           vrt(i,j) = in3(i,j,k,bi,bj,10)
           dSt(i,j) = in3(i,j,k,bi,bj,11)
           zSt(i,j) = in3(i,j,k,bi,bj,12)
           hFZ(i,j) = in3(i,j,k,bi,bj,13)
           vAhZ(i,j)= in3(i,j,k,bi,bj,14)
           vAhD(i,j)= in3(i,j,k,bi,bj,15)
           vA4Z(i,j)= in3(i,j,k,bi,bj,16)
           vA4D(i,j)= in3(i,j,k,bi,bj,17)
           mskW(i,j)= maskW(i,j,k,bi,bj)
          ENDDO
         ENDDO
C     MOM_CALC_KE, KEscheme = -1, 0, 1, 2, 3 -> out3(.., 1..5)
         DO iKE = -1, 3
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            KE(i,j) = in3(i,j,k,bi,bj, 3)
           ENDDO
          ENDDO
          CALL MOM_CALC_KE( bi, bj, k, iKE, uF, vF, KE, myThid )
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            out3(i,j,k,bi,bj, iKE+2) = KE(i,j)
           ENDDO
          ENDDO
         ENDDO
C     GAD_DST3_ADV_X, calcCFL = T (uFld = velocity) -> out3(.., 6),
C     calcCFL = F (uFld = CFL number) -> out3(.., 7)
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uT(i,j) = in3(i,j,k,bi,bj, 8)
          ENDDO
         ENDDO
         CALL GAD_DST3_ADV_X( bi, bj, k, .TRUE., dtLoc,
     &                        uTr, uVl, mskW, trc, uT, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj, 6) = uT(i,j)
           uT(i,j) = in3(i,j,k,bi,bj, 8)
          ENDDO
         ENDDO
         CALL GAD_DST3_ADV_X( bi, bj, k, .FALSE., dtLoc,
     &                        uTr, cfl, mskW, trc, uT, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj, 7) = uT(i,j)
          ENDDO
         ENDDO
C     MOM_VI_HDISSIP, iCmb = 0..7: harmonic = bit 0, biharmonic =
C     bit 1, useVariableViscosity = bit 2 -> out3(.., 8+2*iCmb)
C     (uDissip) and out3(.., 9+2*iCmb) (vDissip)
         DO iCmb = 0, 7
          harm  = MOD(iCmb,2) .EQ. 1
          bihr  = MOD(iCmb/2,2) .EQ. 1
          useVV = MOD(iCmb/4,2) .EQ. 1
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            uD(i,j) = in3(i,j,k,bi,bj,18)
            vD(i,j) = in3(i,j,k,bi,bj,19)
           ENDDO
          ENDDO
          CALL MOM_VI_HDISSIP( bi, bj, k,
     &         hDv, vrt, dSt, zSt, hFZ,
     &         vAhZ, vAhD, vA4Z, vA4D,
     &         harm, bihr, useVV,
     &         uD, vD,
     &         myThid )
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            out3(i,j,k,bi,bj, 8+2*iCmb) = uD(i,j)
            out3(i,j,k,bi,bj, 9+2*iCmb) = vD(i,j)
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
C     8, 7); 3-D hFacC hFacW hFacS recip_hFacC recip_hFacW
C     recip_hFacS maskW maskS; 2-D rAw rAs recip_rA recip_dxC
C     recip_dyC recip_dxG recip_dyG; cosFacU cosFacV;
C     recip_deepFacC; viscAhD viscAhZ viscA4D viscA4Z
      OPEN( uGrd, FILE='replay_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(3) = 8
      ohdr(4) = 7
      WRITE( uGrd ) ohdr
      WRITE( uGrd ) hFacC, hFacW, hFacS, recip_hFacC, recip_hFacW,
     &              recip_hFacS, maskW, maskS
      WRITE( uGrd ) rAw, rAs, recip_rA, recip_dxC, recip_dyC,
     &              recip_dxG, recip_dyG
      WRITE( uGrd ) cosFacU, cosFacV
      WRITE( uGrd ) recip_deepFacC
      WRITE( uGrd ) viscAhD, viscAhZ, viscA4D, viscA4Z
      CLOSE( uGrd )

C--   4c. divide probe: in = int32 (MAGIC, VERSN, n), real*8 a(n),
C     b(n) interleaved as pairs; out = int32 (MAGIC, VERSN, nDvQ),
C     then per pair: a/b, a/3.D0, a/1000.D0, a/7.D0, a/0.1D0
      OPEN( uDvI, FILE='divide_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN', IOSTAT=ios )
      IF ( ios .NE. 0 ) STOP 'REPLAY: cannot open divide_in.bin'
      READ( uDvI ) dhdr
      IF ( dhdr(1).NE.MAGIC .OR. dhdr(2).NE.VERSN )
     &   STOP 'REPLAY: bad magic/version in divide_in.bin'
      nDv = dhdr(3)
      OPEN( uDvO, FILE='divide_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      dhdr(3) = nDvQ
      WRITE( uDvO ) dhdr
      DO n = 1, nDv
       READ( uDvI ) dvA, dvB
       dvQ(1) = dvA/dvB
       dvQ(2) = dvA/3.D0
       dvQ(3) = dvA/1000.D0
       dvQ(4) = dvA/7.D0
       dvQ(5) = dvA/0.1D0
       WRITE( uDvO ) dvQ
      ENDDO
      CLOSE( uDvI )
      CLOSE( uDvO )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A,I9,A)')
     &  'REPLAY harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed; divide probe ', nDv, ' pairs'
      RETURN
      END
