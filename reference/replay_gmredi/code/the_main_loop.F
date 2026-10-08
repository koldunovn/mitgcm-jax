#include "GMREDI_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax GM/Redi replay
C     | harness (reference/replay_gmredi; a copy of the Task 8
C     | harness reference/replay for the M1 sub-lane GMREDI).
C     | Not an MITgcm routine: it replaces the_main_loop.F in a
C     | genmake2 build of global_ocean.90x40x15/code or of
C     | tutorial_global_oce_optim/code_ad (forward only).
C     | THE_MODEL_MAIN has called INITIALISE_FIXED (real grid,
C     | GMREDI_READPARMS, GMREDI_INIT_FIXED). This routine
C     |  1. writes GMREDI_INIT_FIXED's factors and the GMREDI.h
C     |     parameters as read (replay_fixed.bin),
C     |  2. reads the synthetic inputs (replay_in.bin),
C     |  3. calls GMREDI_INIT_VARIA on prior-filled tensors,
C     |  4. overrides values that are trivial in the runs
C     |     (unit factors, 1-D/2-D GM factors, deepFac, the
C     |     isopycnal K and GM_skewflx) so that a missing or
C     |     swapped factor shows, and sets hMixLayer,
C     |  5. calls GMREDI_CALC_TENSOR (sigmaX/Y/R = inputs times
C     |     maskW/S/C, land zero as GRAD_SIGMA leaves them),
C     |     GMREDI_SLOPE_LIMIT (every
C     |     kPos, level), GMREDI_X/Y/RTRANSPORT, GMREDI_CALC_DIFF
C     |     (kArg = 0 for trIdentity 1 and 3, kArg = k) and
C     |     GMREDI_RESIDUAL_FLOW for every tile, each output
C     |     first set to a Python-given prior,
C     |  6. writes every output, all points incl. halos
C     |     (replay_out.bin) and the grid/parameters the routines
C     |     read (replay_grid.bin).
C     | useDiagnostics, useGrdchk and useCTRL are set .FALSE.
C     | (no time stepping; output-only and adjoint-check code).
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "GMREDI.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
C     layout: reference/replay_gmredi/replay_io.py
      INTEGER nIn3, nIn2, nOut3, nPar, nGp
      PARAMETER ( nIn3 = 29, nIn2 = 3, nOut3 = 42, nPar = 3 )
      PARAMETER ( nGp = 11 )
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261002, VERSN = 1 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL in2 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,nIn2)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / REPLAY_GM_BUF / in3, in2, out3
      _RL sigX(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL sigY(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL sigR(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL trc (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kapR(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL uF  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL vF  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL wF  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL slX (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL slY (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL slSq(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL tapF(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL hTrL(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL bSlp(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL rLmb(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dSr (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dSx (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dSy (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL lrho(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL hML (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS xA  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS yA  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS mFk (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS mUp (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL df  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL par(nPar), gp(nGp), rDep
      INTEGER hdr(11), ohdr(4)
      INTEGER uIn, uOut, uGrd, uFix, ios
      INTEGER bi, bj, i, j, k, kPos, iC, iMin, iMax, jMin, jMax
      INTEGER oBase

      uIn  = 91
      uOut = 92
      uGrd = 93
      uFix = 94

C--   1. GMREDI_INIT_FIXED factors and the GMREDI.h parameters as
C     GMREDI_READPARMS left them: int32 (MAGIC, VERSN, nGp, Nr);
C     GM_isoFac2d, GM_bolFac2d (xy tiles); GM_isoFac1d,
C     GM_bolFac1d (Nr); gp(nGp)
      gp( 1) = GM_isopycK
      gp( 2) = GM_background_K
      gp( 3) = GM_maxSlope
      gp( 4) = GM_Kmin_horiz
      gp( 5) = GM_Small_Number
      gp( 6) = GM_slopeSqCutoff
      gp( 7) = GM_Scrit
      gp( 8) = GM_Sd
      gp( 9) = GM_rMaxSlope
      gp(10) = GM_skewflx
      gp(11) = 0. _d 0
      IF ( GM_ExtraDiag ) gp(11) = 1. _d 0
      OPEN( uFix, FILE='replay_fixed.bin', STATUS='NEW',
     &      ACCESS='STREAM', FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(1) = MAGIC
      ohdr(2) = VERSN
      ohdr(3) = nGp
      ohdr(4) = Nr
      WRITE( uFix ) ohdr
      WRITE( uFix ) GM_isoFac2d, GM_bolFac2d
      WRITE( uFix ) GM_isoFac1d, GM_bolFac1d
      WRITE( uFix ) gp
      CLOSE( uFix )

C--   2. inputs: int32 (MAGIC, VERSN, sNx, sNy, OLx, OLy, nSx, nSy,
C     Nr, nIn3, nIn2); real*8 par(nPar); wUnit2rVel, rVel2wUnit
C     (Nr+1); rUnit2z, z2rUnit, GM_isoFac1d, GM_bolFac1d,
C     recip_deepFacC (Nr); deepFac2F (Nr+1); in2; in3
      OPEN( uIn, FILE='replay_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN', IOSTAT=ios )
      IF ( ios .NE. 0 ) STOP 'REPLAY: cannot open replay_in.bin'
      READ( uIn ) hdr
      IF ( hdr(1).NE.MAGIC .OR. hdr(2).NE.VERSN )
     &   STOP 'REPLAY: bad magic/version in replay_in.bin'
      IF ( hdr(3).NE.sNx .OR. hdr(4).NE.sNy .OR. hdr(5).NE.OLx
     &.OR. hdr(6).NE.OLy .OR. hdr(7).NE.nSx .OR. hdr(8).NE.nSy
     &.OR. hdr(9).NE.Nr  .OR. hdr(10).NE.nIn3 .OR. hdr(11).NE.nIn2 )
     &   STOP 'REPLAY: replay_in.bin does not match SIZE.h'
      READ( uIn ) par
      READ( uIn ) wUnit2rVel, rVel2wUnit
      READ( uIn ) rUnit2z, z2rUnit
      READ( uIn ) GM_isoFac1d, GM_bolFac1d
      READ( uIn ) recip_deepFacC
      READ( uIn ) deepFac2F
      READ( uIn ) in2
      READ( uIn ) in3
      CLOSE( uIn )

      useDiagnostics = .FALSE.
      useGrdchk = .FALSE.
      useCTRL = .FALSE.

C--   3. GMREDI_INIT_VARIA on prior-filled tensors -> out3(1..9)
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           Kwx(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 4)
           Kwy(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 5)
           Kwz(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 6)
           Kux(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 7)
           Kvy(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 8)
           Kuz(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 9)
           Kvz(i,j,k,bi,bj) = in3(i,j,k,bi,bj,10)
#ifdef GM_BOLUS_ADVEC
           GM_PsiX(i,j,k,bi,bj) = in3(i,j,k,bi,bj,11)
           GM_PsiY(i,j,k,bi,bj) = in3(i,j,k,bi,bj,12)
#endif
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL GMREDI_INIT_VARIA( myThid )
      oBase = 0
      CALL REPLAY_GM_TENSOR_OUT( oBase, myThid )

C--   4. overrides
      GM_isopycK = par(1)
      GM_skewflx = par(2)
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          GM_isoFac2d(i,j,bi,bj) = in2(i,j,bi,bj,1)
          GM_bolFac2d(i,j,bi,bj) = in2(i,j,bi,bj,2)
          hMixLayer  (i,j,bi,bj) = in2(i,j,bi,bj,3)
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   5a. GMREDI_CALC_TENSOR -> out3(10..18); do_oceanic_phys.F
C     bounds iMin = 1-OLx, iMax = sNx+OLx, jMin = 1-OLy, jMax = sNy+OLy
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           Kwx(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 4)
           Kwy(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 5)
           Kwz(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 6)
           Kux(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 7)
           Kvy(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 8)
           Kuz(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 9)
           Kvz(i,j,k,bi,bj) = in3(i,j,k,bi,bj,10)
#ifdef GM_BOLUS_ADVEC
           GM_PsiX(i,j,k,bi,bj) = in3(i,j,k,bi,bj,11)
           GM_PsiY(i,j,k,bi,bj) = in3(i,j,k,bi,bj,12)
#endif
           sigX(i,j,k) = in3(i,j,k,bi,bj, 1)*maskW(i,j,k,bi,bj)
           sigY(i,j,k) = in3(i,j,k,bi,bj, 2)*maskS(i,j,k,bi,bj)
           sigR(i,j,k) = in3(i,j,k,bi,bj, 3)*maskC(i,j,k,bi,bj)
          ENDDO
         ENDDO
        ENDDO
        CALL GMREDI_CALC_TENSOR(
     I       1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     I       sigX, sigY, sigR,
     I       bi, bj, myTime, myIter, myThid )
       ENDDO
      ENDDO
      oBase = 9
      CALL REPLAY_GM_TENSOR_OUT( oBase, myThid )

C--   5b. GMREDI_SLOPE_LIMIT, kPos = 1, 2, 3, every level ->
C     out3(19..33): (SlopeX, SlopeY, SlopeSqr, taperFct,
C     dSigmaDr) for kPos 1, then 2, then 3
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO kPos = 1, 3
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            dSx (i,j) = in3(i,j,k,bi,bj,13)
            dSy (i,j) = in3(i,j,k,bi,bj,14)
            dSr (i,j) = in3(i,j,k,bi,bj,15)
            slX (i,j) = in3(i,j,k,bi,bj,16)
            slY (i,j) = in3(i,j,k,bi,bj,17)
            slSq(i,j) = in3(i,j,k,bi,bj,18)
            tapF(i,j) = in3(i,j,k,bi,bj,19)
            hTrL(i,j) = R_low(i,j,bi,bj)
            bSlp(i,j) = 0. _d 0
            rLmb(i,j) = 0. _d 0
            lrho(i,j) = 0. _d 0
            hML (i,j) = hMixLayer(i,j,bi,bj)
           ENDDO
          ENDDO
          IF ( kPos.EQ.3 ) THEN
           rDep = rF(1) - rF(k)
           CALL GMREDI_SLOPE_LIMIT(
     O          slX, slY, slSq, tapF,
     U          hTrL, bSlp, rLmb, dSr,
     I          dSx, dSy, lrho, hML, rDep, rF,
     I          kLowC(1-OLx,1-OLy,bi,bj),
     I          kPos, k, bi, bj, myTime, myIter, myThid )
          ELSE
           rDep = rF(1) - rC(k)
           CALL GMREDI_SLOPE_LIMIT(
     O          slX, slY, slSq, tapF,
     U          hTrL, bSlp, rLmb, dSr,
     I          dSx, dSy, lrho, hML, rDep, rC,
     I          kLowC(1-OLx,1-OLy,bi,bj),
     I          kPos, k, bi, bj, myTime, myIter, myThid )
          ENDIF
          iC = 18 + 5*(kPos-1)
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            out3(i,j,k,bi,bj,iC+1) = slX (i,j)
            out3(i,j,k,bi,bj,iC+2) = slY (i,j)
            out3(i,j,k,bi,bj,iC+3) = slSq(i,j)
            out3(i,j,k,bi,bj,iC+4) = tapF(i,j)
            out3(i,j,k,bi,bj,iC+5) = dSr (i,j)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   5c. GMREDI_X/Y/RTRANSPORT (trIdentity 1), bounds as
C     GAD_CALC_RHS calls them with TEMP_INTEGRATE's iMin = 0,
C     iMax = sNx+1, jMin = 0, jMax = sNy+1 -> out3(34..36);
C     tensor = the priors in3(4..8)
      iMin = 0
      iMax = sNx+1
      jMin = 0
      jMax = sNy+1
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           Kwx(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 4)
           Kwy(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 5)
           Kwz(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 6)
           Kux(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 7)
           Kvy(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 8)
           trc(i,j,k) = in3(i,j,k,bi,bj,20)
          ENDDO
         ENDDO
        ENDDO
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           xA (i,j) = in3(i,j,k,bi,bj,21)
           yA (i,j) = in3(i,j,k,bi,bj,22)
           mFk(i,j) = in3(i,j,k,bi,bj,23)
           df (i,j) = in3(i,j,k,bi,bj,25)
          ENDDO
         ENDDO
         CALL GMREDI_XTRANSPORT( 1, bi, bj, k,
     I        iMin, iMax+1, jMin, jMax,
     I        xA, mFk, trc,
     U        df, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj,34) = df(i,j)
           df (i,j) = in3(i,j,k,bi,bj,25)
          ENDDO
         ENDDO
         CALL GMREDI_YTRANSPORT( 1, bi, bj, k,
     I        iMin, iMax, jMin, jMax+1,
     I        yA, mFk, trc,
     U        df, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj,35) = df(i,j)
           df (i,j) = in3(i,j,k,bi,bj,25)
           mUp(i,j) = in3(i,j,k,bi,bj,24)
          ENDDO
         ENDDO
         CALL GMREDI_RTRANSPORT( 1, bi, bj, k,
     I        iMin, iMax, jMin, jMax,
     I        mUp, trc,
     U        df, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj,36) = df(i,j)
          ENDDO
         ENDDO
        ENDDO

C--   5d. GMREDI_CALC_DIFF -> out3(37..39): kArg = 0 with
C     tracerIdentity 1 and 3; kArg = k for every k (trId 2)
        DO iC = 1, 3
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            kapR(i,j,k) = in3(i,j,k,bi,bj,26)
           ENDDO
          ENDDO
         ENDDO
         IF ( iC.EQ.1 ) THEN
          CALL GMREDI_CALC_DIFF( bi, bj, iMin, iMax, jMin, jMax,
     &                           0, Nr, kapR, 1, myThid )
         ELSEIF ( iC.EQ.2 ) THEN
          CALL GMREDI_CALC_DIFF( bi, bj, iMin, iMax, jMin, jMax,
     &                           0, Nr, kapR, 3, myThid )
         ELSE
          DO k = 1, Nr
           CALL GMREDI_CALC_DIFF( bi, bj, iMin, iMax, jMin, jMax,
     &                            k, Nr, kapR, 2, myThid )
          ENDDO
         ENDIF
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            out3(i,j,k,bi,bj,36+iC) = kapR(i,j,k)
           ENDDO
          ENDDO
         ENDDO
        ENDDO

C--   5e. GMREDI_RESIDUAL_FLOW -> out3(40..42)
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uF(i,j,k) = in3(i,j,k,bi,bj,27)
           vF(i,j,k) = in3(i,j,k,bi,bj,28)
           wF(i,j,k) = in3(i,j,k,bi,bj,29)
          ENDDO
         ENDDO
        ENDDO
        CALL GMREDI_RESIDUAL_FLOW( uF, vF, wF, bi, bj, myIter, myThid )
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj,40) = uF(i,j,k)
           out3(i,j,k,bi,bj,41) = vF(i,j,k)
           out3(i,j,k,bi,bj,42) = wF(i,j,k)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   6a. outputs: int32 (MAGIC, VERSN, nOut3, Nr); out3
      OPEN( uOut, FILE='replay_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(1) = MAGIC
      ohdr(2) = VERSN
      ohdr(3) = nOut3
      ohdr(4) = Nr
      WRITE( uOut ) ohdr
      WRITE( uOut ) out3
      CLOSE( uOut )

C--   6b. grid and parameters as the routines read them: int32
C     (MAGIC, VERSN, nGp, Nr); xyz maskC, maskW, maskS; xy R_low,
C     recip_dxC, recip_dyC, rA, maskInC, kLowC (as real*8); rF,
C     recip_drC (Nr+1); rC, recip_deepFacC (Nr); deepFac2F,
C     wUnit2rVel, rVel2wUnit (Nr+1); rUnit2z, z2rUnit (Nr);
C     GM_isoFac1d, GM_bolFac1d (Nr); gravitySign; gp(nGp) after
C     the overrides; usingZCoords (1./0.)
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          xA(i,j) = kLowC(i,j,bi,bj)
          in2(i,j,bi,bj,1) = xA(i,j)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      gp( 1) = GM_isopycK
      gp(10) = GM_skewflx
      rDep = 0. _d 0
      IF ( usingZCoords ) rDep = 1. _d 0
      OPEN( uGrd, FILE='replay_grid.bin', STATUS='NEW',
     &      ACCESS='STREAM', FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(1) = MAGIC
      ohdr(2) = VERSN
      ohdr(3) = nGp
      ohdr(4) = Nr
      WRITE( uGrd ) ohdr
      WRITE( uGrd ) maskC, maskW, maskS
      WRITE( uGrd ) R_low, recip_dxC, recip_dyC, rA, maskInC
      WRITE( uGrd ) ((((in2(i,j,bi,bj,1),i=1-OLx,sNx+OLx),
     &               j=1-OLy,sNy+OLy),bi=1,nSx),bj=1,nSy)
      WRITE( uGrd ) rF, recip_drC
      WRITE( uGrd ) rC, recip_deepFacC
      WRITE( uGrd ) deepFac2F, wUnit2rVel, rVel2wUnit
      WRITE( uGrd ) rUnit2z, z2rUnit
      WRITE( uGrd ) GM_isoFac1d, GM_bolFac1d
      WRITE( uGrd ) gravitySign
      WRITE( uGrd ) gp
      WRITE( uGrd ) rDep
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A)')
     &  'REPLAY GMREDI harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed'
      RETURN
      END

CBOP
C     !ROUTINE: REPLAY_GM_TENSOR_OUT
C     !INTERFACE:
      SUBROUTINE REPLAY_GM_TENSOR_OUT( oBase, myThid )
C     !DESCRIPTION: \bv
C     Copy the GMREDI.h tensor (Kwx Kwy Kwz Kux Kvy Kuz Kvz, then
C     GM_PsiX GM_PsiY under GM_BOLUS_ADVEC, else the priors
C     in3(11), in3(12)) into out3(oBase+1 .. oBase+9).
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "GMREDI.h"
      INTEGER oBase, myThid
CEOP
      INTEGER nIn3, nIn2, nOut3
      PARAMETER ( nIn3 = 29, nIn2 = 3, nOut3 = 42 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL in2 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,nIn2)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / REPLAY_GM_BUF / in3, in2, out3
      INTEGER bi, bj, i, j, k
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj,oBase+1) = Kwx(i,j,k,bi,bj)
           out3(i,j,k,bi,bj,oBase+2) = Kwy(i,j,k,bi,bj)
           out3(i,j,k,bi,bj,oBase+3) = Kwz(i,j,k,bi,bj)
           out3(i,j,k,bi,bj,oBase+4) = Kux(i,j,k,bi,bj)
           out3(i,j,k,bi,bj,oBase+5) = Kvy(i,j,k,bi,bj)
           out3(i,j,k,bi,bj,oBase+6) = Kuz(i,j,k,bi,bj)
           out3(i,j,k,bi,bj,oBase+7) = Kvz(i,j,k,bi,bj)
#ifdef GM_BOLUS_ADVEC
           out3(i,j,k,bi,bj,oBase+8) = GM_PsiX(i,j,k,bi,bj)
           out3(i,j,k,bi,bj,oBase+9) = GM_PsiY(i,j,k,bi,bj)
#else
           out3(i,j,k,bi,bj,oBase+8) = in3(i,j,k,bi,bj,11)
           out3(i,j,k,bi,bj,oBase+9) = in3(i,j,k,bi,bj,12)
#endif
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      RETURN
      END
