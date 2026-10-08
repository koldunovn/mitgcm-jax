#include "GAD_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax GAD-B replay
C     | harness (reference/replay_gad_b, a copy of lane C's
C     | reference/replay for the SOM advection path). Not an
C     | MITgcm routine: it replaces model/src/the_main_loop.F in
C     | a genmake2 build of advect_xy/code or advect_xz/code.
C     | THE_MODEL_MAIN has already called INITIALISE_FIXED, so
C     | GRID.h holds the experiment's real grid and GAD.h the
C     | run's SOM switches. This routine
C     |  1. reads synthetic inputs written by Python
C     |     (replay_in.bin, big-endian stream),
C     |  2. overrides the vertical factors that are 1 in these
C     |     experiments (deepFacC, deepFac2C, recip_deepFac2C,
C     |     deepFac2F, rhoFacC, recip_rhoFacC, rhoFacF) with the
C     |     input values, so that a missing factor is visible,
C     |  3. builds velocities from CFL numbers on the real grid
C     |     and SOM moments scaled by the real cell volumes,
C     |     and calls GAD_SOM_ADVECT (schemes 80 and 81) for
C     |     every tile; GAD_SOM_ADV_X and GAD_SOM_ADV_Y (limiter
C     |     0 and 1) for every tile and level on synthetic
C     |     volumes, moments and transports; GAD_SOM_LIM_R
C     |     (limiter 1) for every tile; GAD_SOM_EXCHANGES on
C     |     synthetic som_T, som_S and GAD_EXCH_SOM on a third
C     |     synthetic moment array,
C     |  4. writes every output array, all points incl. halos
C     |     (replay_out.bin, with the run's switches in the
C     |     header) and the grid as the routines read it
C     |     (replay_grid.bin).
C     | No time stepping. useDiagnostics is set .FALSE. (pkg
C     | diagnostics is not ported; the FILL calls do not change
C     | the outputs).
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "GAD.h"
#include "GAD_SOM_VARS.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
C     nIn3 :: 3-D input fields (layout: reference/replay_gad_b/replay_io.py)
C     nOut3:: 3-D output fields
      INTEGER nIn3, nOut3, nHdr
      PARAMETER ( nIn3 = 54, nOut3 = 118, nHdr = 15 )
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261002, VERSN = 1 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      _RL smTr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nSOM)
      _RL trc (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      _RL l3  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,11)
      COMMON / REPLAY_BUF / in3, out3, smTr, trc, l3
      _RL uF (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL vF (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL wF (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL gTr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL lv (1-OLx:sNx+OLx,1-OLy:sNy+OLy,11)
      _RL tr2(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL fT2(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dTlev(Nr)
      INTEGER hdr(10), ohdr(nHdr)
      INTEGER uIn, uOut, uGrd, ios
      INTEGER bi, bj, i, j, k, km1, n, iS, lim, o
      INTEGER schemes(2)

      uIn  = 91
      uOut = 92
      uGrd = 93
      schemes(1) = ENUM_SOM_PRATHER
      schemes(2) = ENUM_SOM_LIMITER

C--   1. inputs: header int32 (MAGIC, VERSN, sNx, sNy, OLx, OLy,
C     nSx, nSy, Nr, nIn3); real*8 dTlev(Nr); deepFacC(Nr),
C     deepFac2C(Nr), recip_deepFac2C(Nr), rhoFacC(Nr),
C     recip_rhoFacC(Nr), deepFac2F(Nr+1), rhoFacF(Nr+1); in3
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
      READ( uIn ) dTlev
C--   2. overrides (see header)
      READ( uIn ) deepFacC, deepFac2C, recip_deepFac2C,
     &            rhoFacC, recip_rhoFacC, deepFac2F, rhoFacF
      READ( uIn ) in3
      CLOSE( uIn )
      useDiagnostics = .FALSE.

C--   3a. GAD_SOM_ADVECT: velocities from CFL numbers (in3 2..4),
C     moments = ratio (in3 5..13) x cell volume; out3 1..3 = uFld,
C     vFld, wFld, out3 4..12 = moments before, then per scheme
C     (80, 81) gTracer and the 9 moments after: out3 13..22, 23..32
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         km1 = MAX(k-1,1)
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           trc(i,j,k,bi,bj) = in3(i,j,k,bi,bj,1)
           out3(i,j,k,bi,bj,1) = in3(i,j,k,bi,bj,2)
     &                         *dxC(i,j,bi,bj)/dTlev(k)
           out3(i,j,k,bi,bj,2) = in3(i,j,k,bi,bj,3)
     &                         *dyC(i,j,bi,bj)/dTlev(k)
           out3(i,j,k,bi,bj,3) = in3(i,j,k,bi,bj,4)
     &        *MIN( drF(k)*hFacC(i,j,k,bi,bj),
     &              drF(km1)*hFacC(i,j,km1,bi,bj) )/dTlev(k)
           DO n = 1, nSOM
            out3(i,j,k,bi,bj,3+n) = in3(i,j,k,bi,bj,4+n)
     &        *rA(i,j,bi,bj)*drF(k)*hFacC(i,j,k,bi,bj)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      DO iS = 1, 2
       DO n = 1, nSOM
        DO bj = 1, nSy
         DO bi = 1, nSx
          DO k = 1, Nr
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             smTr(i,j,k,bi,bj,n) = out3(i,j,k,bi,bj,3+n)
            ENDDO
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            uF(i,j,k) = out3(i,j,k,bi,bj,1)
            vF(i,j,k) = out3(i,j,k,bi,bj,2)
            wF(i,j,k) = out3(i,j,k,bi,bj,3)
            gTr(i,j,k) = -999.
           ENDDO
          ENDDO
         ENDDO
         CALL GAD_SOM_ADVECT(
     I        .FALSE., schemes(iS), schemes(iS),
     I        GAD_TEMPERATURE, dTlev,
     I        uF, vF, wF, trc,
     U        smTr,
     O        gTr,
     I        bi, bj, myTime, myIter, myThid )
         o = 13 + 10*(iS-1)
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            out3(i,j,k,bi,bj,o) = gTr(i,j,k)
            DO n = 1, nSOM
             out3(i,j,k,bi,bj,o+n) = smTr(i,j,k,bi,bj,n)
            ENDDO
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   3b. GAD_SOM_ADV_X / _Y, limiter 0 and 1, every tile and level:
C     volume, content, 9 moments = in3 16..26, transport in3 14
C     (X) or 15 (Y), prior flux in3 27; out3 33..44 (X, lim 0),
C     45..56 (X, lim 1), 57..68 (Y, lim 0), 69..80 (Y, lim 1):
C     11 updated fields (v,o,x,y,z,xx,yy,zz,xy,xz,yz) + flux
      DO lim = 0, 1
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO k = 1, Nr
C-    X direction
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            DO n = 1, 11
             lv(i,j,n) = in3(i,j,k,bi,bj,15+n)
            ENDDO
            tr2(i,j) = in3(i,j,k,bi,bj,14)
            fT2(i,j) = in3(i,j,k,bi,bj,27)
           ENDDO
          ENDDO
          CALL GAD_SOM_ADV_X(
     I           bi,bj,k, lim,
     I           .FALSE., .FALSE.,
     I           .FALSE., .FALSE., .FALSE., .FALSE.,
     I           dTlev(k), tr2, maskInC(1-OLx,1-OLy,bi,bj),
     U           lv(1-OLx,1-OLy, 1), lv(1-OLx,1-OLy, 2),
     U           lv(1-OLx,1-OLy, 3), lv(1-OLx,1-OLy, 4),
     U           lv(1-OLx,1-OLy, 5), lv(1-OLx,1-OLy, 6),
     U           lv(1-OLx,1-OLy, 7), lv(1-OLx,1-OLy, 8),
     U           lv(1-OLx,1-OLy, 9), lv(1-OLx,1-OLy,10),
     U           lv(1-OLx,1-OLy,11),
     O           fT2,
     I           myThid )
          o = 33 + 12*lim
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            DO n = 1, 11
             out3(i,j,k,bi,bj,o+n-1) = lv(i,j,n)
            ENDDO
            out3(i,j,k,bi,bj,o+11) = fT2(i,j)
           ENDDO
          ENDDO
C-    Y direction
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            DO n = 1, 11
             lv(i,j,n) = in3(i,j,k,bi,bj,15+n)
            ENDDO
            tr2(i,j) = in3(i,j,k,bi,bj,15)
            fT2(i,j) = in3(i,j,k,bi,bj,27)
           ENDDO
          ENDDO
          CALL GAD_SOM_ADV_Y(
     I           bi,bj,k, lim,
     I           .FALSE., .FALSE.,
     I           .FALSE., .FALSE., .FALSE., .FALSE.,
     I           dTlev(k), tr2, maskInC(1-OLx,1-OLy,bi,bj),
     U           lv(1-OLx,1-OLy, 1), lv(1-OLx,1-OLy, 2),
     U           lv(1-OLx,1-OLy, 3), lv(1-OLx,1-OLy, 4),
     U           lv(1-OLx,1-OLy, 5), lv(1-OLx,1-OLy, 6),
     U           lv(1-OLx,1-OLy, 7), lv(1-OLx,1-OLy, 8),
     U           lv(1-OLx,1-OLy, 9), lv(1-OLx,1-OLy,10),
     U           lv(1-OLx,1-OLy,11),
     O           fT2,
     I           myThid )
          o = 57 + 12*lim
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            DO n = 1, 11
             out3(i,j,k,bi,bj,o+n-1) = lv(i,j,n)
            ENDDO
            out3(i,j,k,bi,bj,o+11) = fT2(i,j)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   3c. GAD_SOM_LIM_R, limiter 1, every tile: the 11 fields of
C     in3 16..26 (all levels); out3 81..91
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO n = 1, 11
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            l3(i,j,k,n) = in3(i,j,k,bi,bj,15+n)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        CALL GAD_SOM_LIM_R( bi,bj, 1,
     U           l3(1-OLx,1-OLy,1, 1), l3(1-OLx,1-OLy,1, 2),
     U           l3(1-OLx,1-OLy,1, 3), l3(1-OLx,1-OLy,1, 4),
     U           l3(1-OLx,1-OLy,1, 5), l3(1-OLx,1-OLy,1, 6),
     U           l3(1-OLx,1-OLy,1, 7), l3(1-OLx,1-OLy,1, 8),
     U           l3(1-OLx,1-OLy,1, 9), l3(1-OLx,1-OLy,1,10),
     U           l3(1-OLx,1-OLy,1,11),
     I           myThid )
        DO n = 1, 11
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            out3(i,j,k,bi,bj,80+n) = l3(i,j,k,n)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   3d. GAD_SOM_EXCHANGES on som_T = in3 28..36, som_S = in3
C     37..45 (out3 92..100, 101..109); GAD_EXCH_SOM on in3 46..54
C     (out3 110..118)
      DO n = 1, nSOM
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            som_T(i,j,k,bi,bj,n) = in3(i,j,k,bi,bj,27+n)
            som_S(i,j,k,bi,bj,n) = in3(i,j,k,bi,bj,36+n)
            smTr (i,j,k,bi,bj,n) = in3(i,j,k,bi,bj,45+n)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL GAD_SOM_EXCHANGES( myThid )
      CALL GAD_EXCH_SOM( smTr, Nr, myThid )
      DO n = 1, nSOM
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            out3(i,j,k,bi,bj, 91+n) = som_T(i,j,k,bi,bj,n)
            out3(i,j,k,bi,bj,100+n) = som_S(i,j,k,bi,bj,n)
            out3(i,j,k,bi,bj,109+n) = smTr (i,j,k,bi,bj,n)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   4a. outputs: header int32 (MAGIC, VERSN, nOut3, Nr, then the
C     run's switches: tempSOM_Advection, saltSOM_Advection,
C     rigidLid, nonlinFreeSurf, select_rStar,
C     useCubedSphereExchange, uniformFreeSurfLev, tempAdvScheme,
C     saltAdvScheme, tempVertAdvScheme, saltVertAdvScheme); out3
      DO n = 1, nHdr
       ohdr(n) = 0
      ENDDO
      ohdr(1) = MAGIC
      ohdr(2) = VERSN
      ohdr(3) = nOut3
      ohdr(4) = Nr
      IF ( tempSOM_Advection ) ohdr(5) = 1
      IF ( saltSOM_Advection ) ohdr(6) = 1
      IF ( rigidLid ) ohdr(7) = 1
      ohdr(8) = nonlinFreeSurf
      ohdr(9) = select_rStar
      IF ( useCubedSphereExchange ) ohdr(10) = 1
      IF ( uniformFreeSurfLev ) ohdr(11) = 1
      ohdr(12) = tempAdvScheme
      ohdr(13) = saltAdvScheme
      ohdr(14) = tempVertAdvScheme
      ohdr(15) = saltVertAdvScheme
      OPEN( uOut, FILE='replay_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      WRITE( uOut ) ohdr
      WRITE( uOut ) out3
      CLOSE( uOut )

C--   4b. grid as the routines read it: header int32 (MAGIC, VERSN,
C     7, 5); 2-D dxC dyC dxG dyG rA recip_rA maskInC; 3-D hFacC
C     hFacW hFacS recip_hFacC maskC; 1-D drF recip_drF deepFacC
C     deepFac2C recip_deepFac2C rhoFacC recip_rhoFacC (Nr) and
C     deepFac2F rhoFacF (Nr+1)
      OPEN( uGrd, FILE='replay_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(3) = 7
      ohdr(4) = 5
      WRITE( uGrd ) (ohdr(n), n=1,4)
      WRITE( uGrd ) dxC, dyC, dxG, dyG, rA, recip_rA, maskInC
      WRITE( uGrd ) hFacC, hFacW, hFacS, recip_hFacC, maskC
      WRITE( uGrd ) drF, recip_drF, deepFacC, deepFac2C,
     &              recip_deepFac2C, rhoFacC, recip_rhoFacC,
     &              deepFac2F, rhoFacF
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A)')
     &  'REPLAY GAD-B harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed'
      RETURN
      END
