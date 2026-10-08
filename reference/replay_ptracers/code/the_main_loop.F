#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax PTRACERS replay
C     | harness (reference/replay_ptracers; M2 sub-lane PTRACERS,
C     | a copy of the COL harness reference/replay_col). Not an
C     | MITgcm routine: it replaces model/src/the_main_loop.F in a
C     | genmake2 build of tutorial_tracer_adjsens/code_ad.
C     | THE_MODEL_MAIN has already called INITIALISE_FIXED, so
C     | GRID.h, PARAMS.h, CG2D.h (INI_CG2D) and PTRACERS_PARAMS.h
C     | hold the experiment's real grid and parameters. This
C     | routine
C     |  1. reads synthetic inputs written by Python
C     |     (pt_in.bin: named big-endian stream records),
C     |  2. calls the lane's leaf routines for every tile as the
C     |     model calls them, each output first set to a
C     |     Python-given prior: CONVECTIVE_ADJUSTMENT,
C     |     CONVECTIVE_ADJUSTMENT_INI (with PTRACERS_CONVECT),
C     |     CG2D_NSA (twice: zero and random first guess),
C     |     PTRACERS_APPLY_FORCING (every k), PTRACERS_FORCING_SURF
C     |     (the experiment's own version; two index ranges),
C     |     KPP_CALC_DUMMY, COST_TRACER, SWFRAC, GAD_DST3FL_ADV_R,
C     |  3. writes every output (all points incl. halos) and the
C     |     parameters/grid the routines read (pt_out.bin and
C     |     pt_grid.bin, named records).
C     | No time stepping. A second call (GRDCHK in the code_ad
C     | build) returns at once.
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
#include "FFIELDS.h"
#include "CG2D.h"
#include "EOS.h"
#include "PTRACERS_SIZE.h"
#include "PTRACERS_PARAMS.h"
#include "PTRACERS_START.h"
#include "PTRACERS_FIELDS.h"
#include "KPP.h"
#include "cost.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
C     nIn3 :: 3-D input fields (order: replay_ptracers/replay_io.py)
      INTEGER nIn3, nIn2, nSw
      PARAMETER ( nIn3 = 12, nIn2 = 8, nSw = 64 )
      INTEGER sz2, sz3
      PARAMETER ( sz2 = (sNx+2*OLx)*(sNy+2*OLy)*nSx*nSy )
      PARAMETER ( sz3 = sz2*Nr )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL in2 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,nIn2)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,8)
      _RL out2(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,8)
      COMMON / PT_BUF / in3, in2, out3, out2
      _RL g2d(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL cgB(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL cgX(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL swIn(nSw), swOut(nSw), swFact(1), cgRes(10), objf(nSx,nSy)
      _RL scal(1), tCall
      _RL firstRes, minResSq, lastRes
      INTEGER numIt, nItMin, iCall, n
      INTEGER uIn, uOut, uGrd
      INTEGER bi, bj, i, j, k
      LOGICAL firstCall
      SAVE firstCall
      DATA firstCall / .TRUE. /

      IF ( .NOT.firstCall ) RETURN
      firstCall = .FALSE.
      uIn  = 91
      uOut = 92
      uGrd = 93

C--   1. inputs
      OPEN( uIn, FILE='pt_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL COL_RD( uIn, 'tA', sz3, in3(1-OLx,1-OLy,1,1,1, 1) )
      CALL COL_RD( uIn, 'sA', sz3, in3(1-OLx,1-OLy,1,1,1, 2) )
      CALL COL_RD( uIn, 'pA', sz3, in3(1-OLx,1-OLy,1,1,1, 3) )
      CALL COL_RD( uIn, 'tB', sz3, in3(1-OLx,1-OLy,1,1,1, 4) )
      CALL COL_RD( uIn, 'sB', sz3, in3(1-OLx,1-OLy,1,1,1, 5) )
      CALL COL_RD( uIn, 'pB', sz3, in3(1-OLx,1-OLy,1,1,1, 6) )
      CALL COL_RD( uIn, 'gPrior', sz3, in3(1-OLx,1-OLy,1,1,1, 7) )
      CALL COL_RD( uIn, 'ivdc', sz3, in3(1-OLx,1-OLy,1,1,1, 8) )
      CALL COL_RD( uIn, 'dKr', sz3, in3(1-OLx,1-OLy,1,1,1, 9) )
      CALL COL_RD( uIn, 'pTr', sz3, in3(1-OLx,1-OLy,1,1,1,10) )
      CALL COL_RD( uIn, 'wF', sz3, in3(1-OLx,1-OLy,1,1,1,11) )
      CALL COL_RD( uIn, 'rTr', sz3, in3(1-OLx,1-OLy,1,1,1,12) )
      CALL COL_RD( uIn, 'cgB', sz2, in2(1-OLx,1-OLy,1,1,1) )
      CALL COL_RD( uIn, 'cgX0', sz2, in2(1-OLx,1-OLy,1,1,2) )
      CALL COL_RD( uIn, 'sfS', sz2, in2(1-OLx,1-OLy,1,1,3) )
      CALL COL_RD( uIn, 'sfP', sz2, in2(1-OLx,1-OLy,1,1,4) )
      CALL COL_RD( uIn, 'relaxS', sz2, in2(1-OLx,1-OLy,1,1,5) )
      CALL COL_RD( uIn, 'emp', sz2, in2(1-OLx,1-OLy,1,1,6) )
      CALL COL_RD( uIn, 'sfPrior', sz2, in2(1-OLx,1-OLy,1,1,7) )
      CALL COL_RD( uIn, 'kppPrior', sz2, in2(1-OLx,1-OLy,1,1,8) )
      CALL COL_RD( uIn, 'swdk', nSw, swIn )
      CALL COL_RD( uIn, 'swFact', 1, swFact )
      CALL COL_RD( uIn, 'objfPrior', nSx*nSy, objf )
      CLOSE( uIn )

C     PTRACERS_START.h as PTRACERS_INIT_VARIA sets it (ptracers_init_varia
C     .F:44-47; INITIALISE_VARIA does not run in the harness): every
C     tracer is stepped, so PTRACERS_CONVECT mixes it
      DO n = 1, PTRACERS_num
       PTRACERS_StepFwd(n) = .TRUE.
      ENDDO

C     time of the CONVECTIVE_ADJUSTMENT call: 3 clock steps after
C     startTime (DIFFERENT_MULTIPLE(cAdjFreq,..) as in a run)
      iCall = nIter0 + 3
      tCall = startTime + 3. _d 0*deltaTClock

C--   2a. CONVECTIVE_ADJUSTMENT (out3 1..3), then
C     CONVECTIVE_ADJUSTMENT_INI (out3 4..6) on the second set
      DO n = 1, 2
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            theta(i,j,k,bi,bj) = in3(i,j,k,bi,bj,3*n-2)
            salt (i,j,k,bi,bj) = in3(i,j,k,bi,bj,3*n-1)
            pTracer(i,j,k,bi,bj,1) = in3(i,j,k,bi,bj,3*n)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       DO bj = 1, nSy
        DO bi = 1, nSx
         IF ( n.EQ.1 ) THEN
          CALL CONVECTIVE_ADJUSTMENT( bi, bj, tCall, iCall, myThid )
         ELSE
          CALL CONVECTIVE_ADJUSTMENT_INI( bi, bj, startTime, nIter0,
     &                                    myThid )
         ENDIF
         DO k = 1, Nr
          CALL COL_PUT( theta(1-OLx,1-OLy,k,bi,bj),
     &                  out3(1-OLx,1-OLy,k,bi,bj,3*n-2) )
          CALL COL_PUT( salt(1-OLx,1-OLy,k,bi,bj),
     &                  out3(1-OLx,1-OLy,k,bi,bj,3*n-1) )
          CALL COL_PUT( pTracer(1-OLx,1-OLy,k,bi,bj,1),
     &                  out3(1-OLx,1-OLy,k,bi,bj,3*n) )
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   2b. PTRACERS_APPLY_FORCING (out3 7): ptracers_integrate.F:296-305
C     (iMin..jMax = 0..sNx+1), every k, gTrForc = prior
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         CALL COL_GET( in3(1-OLx,1-OLy,k,bi,bj,7), g2d )
         CALL PTRACERS_APPLY_FORCING( g2d, in2(1-OLx,1-OLy,bi,bj,4),
     &                  0, sNx+1, 0, sNy+1, k, bi, bj,
     &                  1, myTime, myIter, myThid )
         CALL COL_PUT( g2d, out3(1-OLx,1-OLy,k,bi,bj,7) )
        ENDDO
       ENDDO
      ENDDO

C--   2c. PTRACERS_FORCING_SURF (out2 1: range of DO_OCEANIC_PHYS,
C     1-OLx..sNx+OLx; out2 2: range 1..sNx, 1..sNy), prior sfPrior;
C     surfaceForcingS = sfS, EmPmR = emp, pTracer = pTr
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           pTracer(i,j,k,bi,bj,1) = in3(i,j,k,bi,bj,10)
          ENDDO
         ENDDO
        ENDDO
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          surfaceForcingS(i,j,bi,bj) = in2(i,j,bi,bj,3)
          EmPmR(i,j,bi,bj) = in2(i,j,bi,bj,6)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      DO n = 1, 2
       DO bj = 1, nSy
        DO bi = 1, nSx
         CALL COL_PUT( in2(1-OLx,1-OLy,bi,bj,7),
     &                 surfaceForcingPTr(1-OLx,1-OLy,bi,bj,1) )
         IF ( n.EQ.1 ) THEN
          CALL PTRACERS_FORCING_SURF( in2(1-OLx,1-OLy,bi,bj,5),
     &         bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     &         myTime, myIter, myThid )
         ELSE
          CALL PTRACERS_FORCING_SURF( in2(1-OLx,1-OLy,bi,bj,5),
     &         bi, bj, 1, sNx, 1, sNy, myTime, myIter, myThid )
         ENDIF
         CALL COL_PUT( surfaceForcingPTr(1-OLx,1-OLy,bi,bj,1),
     &                 out2(1-OLx,1-OLy,bi,bj,n) )
        ENDDO
       ENDDO
      ENDDO

C--   2d. KPP_CALC_DUMMY (KPP.h): IVDConvCount = ivdc, diffKr = dKr,
C     KPP fields first set to priors
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           IVDConvCount(i,j,k,bi,bj) = in3(i,j,k,bi,bj,8)
#ifdef ALLOW_3D_DIFFKR
           diffKr(i,j,k,bi,bj) = in3(i,j,k,bi,bj,9)
#endif
           KPPghat(i,j,k,bi,bj)    = in3(i,j,k,bi,bj,7)
           KPPviscAz(i,j,k,bi,bj)  = in3(i,j,k,bi,bj,7)
           KPPdiffKzS(i,j,k,bi,bj) = in3(i,j,k,bi,bj,7)
           KPPdiffKzT(i,j,k,bi,bj) = in3(i,j,k,bi,bj,7)
          ENDDO
         ENDDO
        ENDDO
        CALL COL_PUT( in2(1-OLx,1-OLy,bi,bj,8),
     &                KPPhbl(1-OLx,1-OLy,bi,bj) )
        CALL COL_PUT( in2(1-OLx,1-OLy,bi,bj,8),
     &                KPPfrac(1-OLx,1-OLy,bi,bj) )
        CALL KPP_CALC_DUMMY( bi, bj, myTime, myIter, myThid )
        CALL COL_PUT( KPPhbl(1-OLx,1-OLy,bi,bj),
     &                out2(1-OLx,1-OLy,bi,bj,3) )
        CALL COL_PUT( KPPfrac(1-OLx,1-OLy,bi,bj),
     &                out2(1-OLx,1-OLy,bi,bj,4) )
       ENDDO
      ENDDO

C--   2e. COST_TRACER (objf_tracer = prior, pTracer = pTr)
      DO bj = 1, nSy
       DO bi = 1, nSx
        objf_tracer(bi,bj) = objf(bi,bj)
        CALL COST_TRACER( bi, bj, myThid )
        objf(bi,bj) = objf_tracer(bi,bj)
       ENDDO
      ENDDO

C--   2f. CG2D_NSA (CG2D.h of INI_CG2D): n=1 first guess 0, n=2
C     first guess cgX0; numIters = cg2dMaxIters, nIterMin = -1
C     (solve_for_pressure.F:277-278 with cg2dUseMinResSol = 0)
      DO n = 1, 2
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           cgB(i,j,bi,bj) = in2(i,j,bi,bj,1)
           cgX(i,j,bi,bj) = 0. _d 0
           IF ( n.EQ.2 ) cgX(i,j,bi,bj) = in2(i,j,bi,bj,2)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       numIt  = cg2dMaxIters
       nItMin = -1
       CALL CG2D_NSA( cgB, cgX, firstRes, minResSq, lastRes,
     &                numIt, nItMin, myThid )
       CALL COL_PUT2( cgX, out2(1-OLx,1-OLy,1,1,3+2*n) )
       CALL COL_PUT2( cgB, out2(1-OLx,1-OLy,1,1,4+2*n) )
       cgRes(5*n-4) = firstRes
       cgRes(5*n-3) = minResSq
       cgRes(5*n-2) = lastRes
       cgRes(5*n-1) = numIt
       cgRes(5*n)   = nItMin
      ENDDO

C--   2h. GAD_DST3FL_ADV_R (out3 8): gad_advection.F:999-1002 form,
C     every k, tracer = tA, wFld = wF(k), rTrans = rTr(k),
C     dTarg = dTtracerLev(k); wT first set to the prior gPrior(k)
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         CALL COL_GET( in3(1-OLx,1-OLy,k,bi,bj,7), g2d )
         CALL GAD_DST3FL_ADV_R( bi, bj, k, dTtracerLev(k),
     &        in3(1-OLx,1-OLy,k,bi,bj,12), in3(1-OLx,1-OLy,k,bi,bj,11),
     &        in3(1-OLx,1-OLy,1,bi,bj,1), g2d, myThid )
         CALL COL_PUT( g2d, out3(1-OLx,1-OLy,k,bi,bj,8) )
        ENDDO
       ENDDO
      ENDDO

C--   2g. SWFRAC (ini_forcing.F:163-166 form: imax, fact, swdk)
      DO n = 1, nSw
       swOut(n) = swIn(n)
      ENDDO
      CALL SWFRAC( nSw, swFact(1), swOut, myTime, myIter, myThid )

C--   3a. outputs
      OPEN( uOut, FILE='pt_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL COL_WR( uOut, 'convA_t', 3, sz3, out3(1-OLx,1-OLy,1,1,1,1) )
      CALL COL_WR( uOut, 'convA_s', 3, sz3, out3(1-OLx,1-OLy,1,1,1,2) )
      CALL COL_WR( uOut, 'convA_p', 3, sz3, out3(1-OLx,1-OLy,1,1,1,3) )
      CALL COL_WR( uOut, 'convI_t', 3, sz3, out3(1-OLx,1-OLy,1,1,1,4) )
      CALL COL_WR( uOut, 'convI_s', 3, sz3, out3(1-OLx,1-OLy,1,1,1,5) )
      CALL COL_WR( uOut, 'convI_p', 3, sz3, out3(1-OLx,1-OLy,1,1,1,6) )
      CALL COL_WR( uOut, 'gForc', 3, sz3, out3(1-OLx,1-OLy,1,1,1,7) )
      CALL COL_WR( uOut, 'dst3flR', 3, sz3, out3(1-OLx,1-OLy,1,1,1,8) )
      CALL COL_WR( uOut, 'sfPTr_full', 2, sz2, out2(1-OLx,1-OLy,1,1,1) )
      CALL COL_WR( uOut, 'sfPTr_int', 2, sz2, out2(1-OLx,1-OLy,1,1,2) )
      CALL COL_WR( uOut, 'KPPhbl', 2, sz2, out2(1-OLx,1-OLy,1,1,3) )
      CALL COL_WR( uOut, 'KPPfrac', 2, sz2, out2(1-OLx,1-OLy,1,1,4) )
      CALL COL_WR( uOut, 'KPPghat', 3, sz3, KPPghat )
      CALL COL_WR( uOut, 'KPPviscAz', 3, sz3, KPPviscAz )
      CALL COL_WR( uOut, 'KPPdiffKzS', 3, sz3, KPPdiffKzS )
      CALL COL_WR( uOut, 'KPPdiffKzT', 3, sz3, KPPdiffKzT )
      CALL COL_WR( uOut, 'objf_tracer', 0, nSx*nSy, objf )
      CALL COL_WR( uOut, 'cg_x1', 2, sz2, out2(1-OLx,1-OLy,1,1,5) )
      CALL COL_WR( uOut, 'cg_b1', 2, sz2, out2(1-OLx,1-OLy,1,1,6) )
      CALL COL_WR( uOut, 'cg_x2', 2, sz2, out2(1-OLx,1-OLy,1,1,7) )
      CALL COL_WR( uOut, 'cg_b2', 2, sz2, out2(1-OLx,1-OLy,1,1,8) )
      CALL COL_WR( uOut, 'cg_res', 0, 10, cgRes )
      CALL COL_WR( uOut, 'swdk', 1, nSw, swOut )
      CLOSE( uOut )

C--   3b. parameters and grid as the routines read them
      OPEN( uGrd, FILE='pt_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL COL_WR( uGrd, 'hFacC', 3, sz3, hFacC )
      CALL COL_WR( uGrd, 'recip_hFacC', 3, sz3, recip_hFacC )
      CALL COL_WR( uGrd, 'maskC', 3, sz3, maskC )
      CALL COL_WR( uGrd, 'rA', 2, sz2, rA )
      CALL COL_WR( uGrd, 'aW2d', 2, sz2, aW2d )
      CALL COL_WR( uGrd, 'aS2d', 2, sz2, aS2d )
      CALL COL_WR( uGrd, 'aC2d', 2, sz2, aC2d )
      CALL COL_WR( uGrd, 'pW', 2, sz2, pW )
      CALL COL_WR( uGrd, 'pS', 2, sz2, pS )
      CALL COL_WR( uGrd, 'pC', 2, sz2, pC )
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          out2(i,j,bi,bj,1) = kSurfC(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL COL_WR( uGrd, 'kSurfC', 2, sz2, out2(1-OLx,1-OLy,1,1,1) )
      CALL COL_WR( uGrd, 'rF', 1, Nr+1, rF )
      CALL COL_WR( uGrd, 'drF', 1, Nr, drF )
      CALL COL_WR( uGrd, 'recip_drC', 1, Nr+1, recip_drC )
      CALL COL_WR( uGrd, 'recip_drF', 1, Nr, recip_drF )
      CALL COL_WR( uGrd, 'tRef', 1, Nr, tRef )
      CALL COL_WR( uGrd, 'sRef', 1, Nr, sRef )
      CALL COL_WR( uGrd, 'viscArNr', 1, Nr, viscArNr )
      CALL COL_WR( uGrd, 'dTtracerLev', 1, Nr, dTtracerLev )
      CALL COL_WR( uGrd, 'PTRACERS_dTLev', 1, Nr, PTRACERS_dTLev )
      CALL COL_WR( uGrd, 'PTRACERS_diffKrNr', 1, Nr,
     &             PTRACERS_diffKrNr )
      CALL COL_WR( uGrd, 'PTRACERS_ref', 1, Nr, PTRACERS_ref )
      scal(1) = PTRACERS_EvPrRn(1)
      CALL COL_WR( uGrd, 'PTRACERS_EvPrRn', 0, 1, scal )
      scal(1) = lambdaTr1ClimRelax
      CALL COL_WR( uGrd, 'lambdaTr1ClimRelax', 0, 1, scal )
      scal(1) = tAlpha
      CALL COL_WR( uGrd, 'tAlpha', 0, 1, scal )
      scal(1) = sBeta
      CALL COL_WR( uGrd, 'sBeta', 0, 1, scal )
      scal(1) = rhoNil
      CALL COL_WR( uGrd, 'rhoNil', 0, 1, scal )
      scal(1) = rhoConst
      CALL COL_WR( uGrd, 'rhoConst', 0, 1, scal )
      scal(1) = gravitySign
      CALL COL_WR( uGrd, 'gravitySign', 0, 1, scal )
      scal(1) = rkSign
      CALL COL_WR( uGrd, 'rkSign', 0, 1, scal )
      scal(1) = cAdjFreq
      CALL COL_WR( uGrd, 'cAdjFreq', 0, 1, scal )
      scal(1) = deltaTClock
      CALL COL_WR( uGrd, 'deltaTClock', 0, 1, scal )
      scal(1) = tCall
      CALL COL_WR( uGrd, 'tCall', 0, 1, scal )
      scal(1) = startTime
      CALL COL_WR( uGrd, 'startTime', 0, 1, scal )
      scal(1) = ivdc_kappa
      CALL COL_WR( uGrd, 'ivdc_kappa', 0, 1, scal )
      scal(1) = diffKrBL79surf
      CALL COL_WR( uGrd, 'diffKrBL79surf', 0, 1, scal )
      scal(1) = diffKrBL79deep
      CALL COL_WR( uGrd, 'diffKrBL79deep', 0, 1, scal )
      scal(1) = diffKrBL79Ho
      CALL COL_WR( uGrd, 'diffKrBL79Ho', 0, 1, scal )
      scal(1) = diffKrBL79scl
      CALL COL_WR( uGrd, 'diffKrBL79scl', 0, 1, scal )
      scal(1) = cg2dNorm
      CALL COL_WR( uGrd, 'cg2dNorm', 0, 1, scal )
      scal(1) = cg2dTolerance_sq
      CALL COL_WR( uGrd, 'cg2dTolerance_sq', 0, 1, scal )
      scal(1) = cg2dMaxIters
      CALL COL_WR( uGrd, 'cg2dMaxIters', 0, 1, scal )
      scal(1) = cg2dMinItersNSA
      CALL COL_WR( uGrd, 'cg2dMinItersNSA', 0, 1, scal )
      scal(1) = convertFW2Salt
      CALL COL_WR( uGrd, 'convertFW2Salt', 0, 1, scal )
      scal(1) = mass2rUnit
      CALL COL_WR( uGrd, 'mass2rUnit', 0, 1, scal )
      scal(1) = nonlinFreeSurf
      CALL COL_WR( uGrd, 'nonlinFreeSurf', 0, 1, scal )
      scal(1) = 0.
      IF ( useRealFreshWaterFlux ) scal(1) = 1.
      CALL COL_WR( uGrd, 'useRealFreshWaterFlux', 0, 1, scal )
      scal(1) = 0.
      IF ( PTRACERS_addSrelax2EmP ) scal(1) = 1.
      CALL COL_WR( uGrd, 'PTRACERS_addSrelax2EmP', 0, 1, scal )
      scal(1) = 0.
      IF ( PTRACERS_StepFwd(1) ) scal(1) = 1.
      CALL COL_WR( uGrd, 'PTRACERS_StepFwd', 0, 1, scal )
      scal(1) = 0.
      IF ( PTRACERS_linFSConserve(1) ) scal(1) = 1.
      CALL COL_WR( uGrd, 'PTRACERS_linFSConserve', 0, 1, scal )
      scal(1) = PTRACERS_numInUse
      CALL COL_WR( uGrd, 'PTRACERS_numInUse', 0, 1, scal )
      scal(1) = 0.
      IF ( usePTRACERS ) scal(1) = 1.
      CALL COL_WR( uGrd, 'usePTRACERS', 0, 1, scal )
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A,I6)')
     &  'REPLAY PTRACERS harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed; cg2d_nsa iters ', INT(cgRes(4))
      RETURN
      END

C     named records (big-endian stream): CHARACTER*16 name,
C     INTEGER rank, INTEGER n, REAL*8 a(n)
      SUBROUTINE COL_RD( u, nm, n, a )
      IMPLICIT NONE
      INTEGER u, n
      CHARACTER*(*) nm
      Real*8 a(n)
      CHARACTER*16 got
      INTEGER rk, m
      READ( u ) got, rk, m
      IF ( got.NE.nm .OR. m.NE.n ) THEN
       WRITE(0,*) 'COL_RD: want ', nm, n, ' got ', got, m
       STOP 'REPLAY PTRACERS: bad input record'
      ENDIF
      READ( u ) a
      RETURN
      END

      SUBROUTINE COL_WR( u, nm, rk, n, a )
      IMPLICIT NONE
      INTEGER u, rk, n
      CHARACTER*(*) nm
      Real*8 a(n)
      CHARACTER*16 nm16
      nm16 = nm
      WRITE( u ) nm16, rk, n
      WRITE( u ) a
      RETURN
      END

      SUBROUTINE COL_PUT( a, b )
      IMPLICIT NONE
#include "SIZE.h"
      Real*8 a(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      Real*8 b(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER i, j
      DO j = 1-OLy, sNy+OLy
       DO i = 1-OLx, sNx+OLx
        b(i,j) = a(i,j)
       ENDDO
      ENDDO
      RETURN
      END

      SUBROUTINE COL_GET( a, b )
      IMPLICIT NONE
#include "SIZE.h"
      Real*8 a(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      Real*8 b(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      CALL COL_PUT( a, b )
      RETURN
      END

      SUBROUTINE COL_PUT2( a, b )
      IMPLICIT NONE
#include "SIZE.h"
      Real*8 a(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      Real*8 b(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      INTEGER i, j, bi, bj
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          b(i,j,bi,bj) = a(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      RETURN
      END
