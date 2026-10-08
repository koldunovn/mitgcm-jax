#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax COL replay
C     | harness (reference/replay_col; M1 sub-lane COL, a copy of
C     | the Task 8 harness reference/replay). Not an MITgcm
C     | routine: it replaces model/src/the_main_loop.F in a
C     | genmake2 build of a verification experiment's code dir.
C     | THE_MODEL_MAIN has already called INITIALISE_FIXED, so
C     | GRID.h, PARAMS.h and EOS.h hold the experiment's real
C     | grid and parameters (INI_EOS has run). This routine
C     |  1. reads synthetic inputs written by Python
C     |     (col_in.bin: named big-endian stream records),
C     |  2. overrides PARAMS.h/GRID.h values that are trivial in
C     |     the experiment (deepFac*, rhoFac* = 1, viscArNr and
C     |     dTtracerLev uniform) with input values, so that a
C     |     missing or misplaced factor shows in the gates,
C     |  3. calls the column-physics routines of the lane for
C     |     every tile and level as the model calls them (index
C     |     ranges of the callers), each output first set to a
C     |     Python-given prior (points a routine does not write
C     |     keep it): FIND_RHO_2D, FIND_RHOP0, PRESSURE_FOR_EOS,
C     |     FIND_BULKMOD, FIND_RHO_SCALAR, FIND_ALPHA,
C     |     SOLVE_TRIDIAGONAL, IMPLDIFF, CALC_IVDC,
C     |     INTEGRATE_FOR_W, CALC_VISCOSITY, CALC_OCE_MXLAYER,
C     |     CALC_PHI_HYD,
C     |  4. writes every output array (all points incl. halos)
C     |     and the parameters/grid the routines read (col_out.bin
C     |     and col_grid.bin, named records).
C     | No time stepping. A second call (GRDCHK in a code_ad
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
#include "EOS.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
C     nIn3 :: 3-D input fields (order: reference/replay_col/replay_io.py)
      INTEGER nIn3
      PARAMETER ( nIn3 = 17 )
      INTEGER nOut3
      PARAMETER ( nOut3 = 19 )
      INTEGER sz2, sz3, sz3p
      PARAMETER ( sz2 = (sNx+2*OLx)*(sNy+2*OLy)*nSx*nSy )
      PARAMETER ( sz3 = sz2*Nr, sz3p = sz2*(Nr+1) )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      _RL kapU(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1,nSx,nSy)
      _RL kapV(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1,nSx,nSy)
      _RL in2 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,2)
      _RL out2(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,3)
      _RL u5(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      _RL v5(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      _RL w5(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      COMMON / COL_BUF / in3, out3, kapU, kapV, in2, out2, u5, v5, w5
      _RL tmp3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      COMMON / COL_TMP / tmp3
      _RL t2d(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL s2d(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL r2d(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL p2d(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL a3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL b3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL c3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL y3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kap3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL g3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL sigR(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kRU(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1)
      _RL kRV(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1)
      _RL phF(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL phC(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dPX(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dPY(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL rDh(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL errT(nSx,nSy)
      _RL scal(1), dp0, rS, sS
      INTEGER uIn, uOut, uGrd, errCode
      INTEGER bi, bj, i, j, k, kRef, n
      LOGICAL firstCall
      SAVE firstCall
      DATA firstCall / .TRUE. /

      IF ( .NOT.firstCall ) RETURN
      firstCall = .FALSE.
      uIn  = 91
      uOut = 92
      uGrd = 93

C--   1. inputs
      OPEN( uIn, FILE='col_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL COL_RD( uIn, 'tFld', sz3, in3(1-OLx,1-OLy,1,1,1, 1) )
      CALL COL_RD( uIn, 'sFld', sz3, in3(1-OLx,1-OLy,1,1,1, 2) )
      CALL COL_RD( uIn, 'totPhi', sz3, in3(1-OLx,1-OLy,1,1,1, 3) )
      CALL COL_RD( uIn, 'pLoc', sz3, in3(1-OLx,1-OLy,1,1,1, 4) )
      CALL COL_RD( uIn, 'aTri', sz3, in3(1-OLx,1-OLy,1,1,1, 5) )
      CALL COL_RD( uIn, 'bTri', sz3, in3(1-OLx,1-OLy,1,1,1, 6) )
      CALL COL_RD( uIn, 'cTri', sz3, in3(1-OLx,1-OLy,1,1,1, 7) )
      CALL COL_RD( uIn, 'yTri', sz3, in3(1-OLx,1-OLy,1,1,1, 8) )
      CALL COL_RD( uIn, 'kappaRX', sz3, in3(1-OLx,1-OLy,1,1,1, 9) )
      CALL COL_RD( uIn, 'gTr', sz3, in3(1-OLx,1-OLy,1,1,1,10) )
      CALL COL_RD( uIn, 'sigmaR', sz3, in3(1-OLx,1-OLy,1,1,1,11) )
      CALL COL_RD( uIn, 'uFld', sz3, in3(1-OLx,1-OLy,1,1,1,12) )
      CALL COL_RD( uIn, 'vFld', sz3, in3(1-OLx,1-OLy,1,1,1,13) )
      CALL COL_RD( uIn, 'wPrior', sz3, in3(1-OLx,1-OLy,1,1,1,14) )
      CALL COL_RD( uIn, 'rhoPrior', sz3, in3(1-OLx,1-OLy,1,1,1,15) )
      CALL COL_RD( uIn, 'ivdcPrior', sz3, in3(1-OLx,1-OLy,1,1,1,16) )
      CALL COL_RD( uIn, 'phiPrior', sz3, in3(1-OLx,1-OLy,1,1,1,17) )
      CALL COL_RD( uIn, 'rStarDhDt', sz2, in2(1-OLx,1-OLy,1,1,1) )
      CALL COL_RD( uIn, 'hMixPrior', sz2, in2(1-OLx,1-OLy,1,1,2) )
C--   2. overrides of PARAMS.h / GRID.h values (see header)
      CALL COL_RD( uIn, 'deepFacC', Nr, deepFacC )
      CALL COL_RD( uIn, 'deepFac2F', Nr+1, deepFac2F )
      CALL COL_RD( uIn, 'recip_deepFac2C', Nr, recip_deepFac2C )
      CALL COL_RD( uIn, 'recip_deepFac2F', Nr+1, recip_deepFac2F )
      CALL COL_RD( uIn, 'rhoFacC', Nr, rhoFacC )
      CALL COL_RD( uIn, 'rhoFacF', Nr+1, rhoFacF )
      CALL COL_RD( uIn, 'recip_rhoFacC', Nr, recip_rhoFacC )
      CALL COL_RD( uIn, 'recip_rhoFacF', Nr+1, recip_rhoFacF )
      CALL COL_RD( uIn, 'viscArNr', Nr, viscArNr )
      CALL COL_RD( uIn, 'dTtracerLev', Nr, dTtracerLev )
      CLOSE( uIn )

C     model state the routines read from DYNVARS.h
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           theta(i,j,k,bi,bj)     = in3(i,j,k,bi,bj, 1)
           salt (i,j,k,bi,bj)     = in3(i,j,k,bi,bj, 2)
           totPhiHyd(i,j,k,bi,bj) = in3(i,j,k,bi,bj, 3)
           uVel(i,j,k,bi,bj)      = in3(i,j,k,bi,bj,12)
           vVel(i,j,k,bi,bj)      = in3(i,j,k,bi,bj,13)
           wVel(i,j,k,bi,bj)      = in3(i,j,k,bi,bj,14)
           IVDConvCount(i,j,k,bi,bj) = in3(i,j,k,bi,bj,16)
           u5(i,j,k,bi,bj) = in3(i,j,k,bi,bj,12)
           v5(i,j,k,bi,bj) = in3(i,j,k,bi,bj,13)
           w5(i,j,k,bi,bj) = in3(i,j,k,bi,bj,14)
          ENDDO
         ENDDO
        ENDDO
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          hMixLayer(i,j,bi,bj) = in2(i,j,bi,bj,2)
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C     pkg/diagnostics as at the first time step (forward_step.F calls
C     DIAGNOSTICS_SWITCH_ONOFF before the physics): fills are allowed and
C     DIAGNOSTICS_IS_ON (CALC_OCE_MXLAYER) answers as in the model
#ifdef ALLOW_DIAGNOSTICS
      IF ( useDiagnostics )
     &  CALL DIAGNOSTICS_SWITCH_ONOFF( 1, myTime, myIter, myThid )
#endif

C--   3. replay, tile by tile
      DO bj = 1, nSy
       DO bi = 1, nSx
C-    EOS (out3 1..6): do_oceanic_phys.F:759-766 (full range, kRef=k);
C     dynamics/temp_integrate range 0..sNx+1 with kRef=k-1 (part);
C     the EOS pieces FIND_RHOP0, PRESSURE_FOR_EOS, FIND_BULKMOD;
C     FIND_RHO_SCALAR at every point with ABS(salinity)
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           t2d(i,j) = in3(i,j,k,bi,bj,1)
           s2d(i,j) = in3(i,j,k,bi,bj,2)
           r2d(i,j) = in3(i,j,k,bi,bj,15)
          ENDDO
         ENDDO
         CALL FIND_RHO_2D( 1-OLx, sNx+OLx, 1-OLy, sNy+OLy, k,
     &                     t2d, s2d, r2d, k, bi, bj, myThid )
         CALL COL_PUT( r2d, out3(1-OLx,1-OLy,k,bi,bj,1) )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           rhoInSitu(i,j,k,bi,bj) = r2d(i,j)
           r2d(i,j) = in3(i,j,k,bi,bj,15)
          ENDDO
         ENDDO
         kRef = MAX(k-1,1)
         CALL FIND_RHO_2D( 0, sNx+1, 0, sNy+1, kRef,
     &                     t2d, s2d, r2d, k, bi, bj, myThid )
         CALL COL_PUT( r2d, out3(1-OLx,1-OLy,k,bi,bj,2) )
         CALL COL_GET( in3(1-OLx,1-OLy,k,bi,bj,15), r2d )
         CALL FIND_RHOP0( 1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     &                    t2d, s2d, r2d, myThid )
         CALL COL_PUT( r2d, out3(1-OLx,1-OLy,k,bi,bj,3) )
         CALL COL_GET( in3(1-OLx,1-OLy,k,bi,bj,15), p2d )
         dp0 = surf_pRef - eosRefP0
         CALL PRESSURE_FOR_EOS( bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     &                          k, dp0, p2d, myThid )
         CALL COL_PUT( p2d, out3(1-OLx,1-OLy,k,bi,bj,4) )
         CALL COL_GET( in3(1-OLx,1-OLy,k,bi,bj,15), r2d )
         CALL FIND_BULKMOD( 1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     &                      p2d, t2d, s2d, r2d, myThid )
         CALL COL_PUT( r2d, out3(1-OLx,1-OLy,k,bi,bj,5) )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
C     |salinity|: no negative value, so FIND_RHO_SCALAR prints no
C     warning to STDERR (the declared code_ad stop compares it)
           sS = ABS( in3(i,j,k,bi,bj,2) )
           CALL FIND_RHO_SCALAR( in3(i,j,k,bi,bj,1), sS,
     &                           in3(i,j,k,bi,bj,4), rS, myThid )
           out3(i,j,k,bi,bj,6) = rS
          ENDDO
         ENDDO
        ENDDO
C-    SOLVE_TRIDIAGONAL (out3 7, errT): gad_implicit_r.F:264 range
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           a3(i,j,k) = in3(i,j,k,bi,bj,5)
           b3(i,j,k) = in3(i,j,k,bi,bj,6)
           c3(i,j,k) = in3(i,j,k,bi,bj,7)
           y3(i,j,k) = in3(i,j,k,bi,bj,8)
          ENDDO
         ENDDO
        ENDDO
        errCode = -1
        CALL SOLVE_TRIDIAGONAL( 0, sNx+1, 0, sNy+1,
     &                          a3, b3, c3, y3, errCode,
     &                          bi, bj, myThid )
        errT(bi,bj) = errCode
        DO k = 1, Nr
         CALL COL_PUT( y3(1-OLx,1-OLy,k), out3(1-OLx,1-OLy,k,bi,bj,7) )
        ENDDO
C-    IMPLDIFF (out3 8: tracerId = GAD_TEMPERATURE = 1; out3 9:
C     tracerId = -1, momentum): temp_integrate.F:160-163 and
C     dynamics.F:191-192 range 0..sNx+1, recip_hFacC
        DO n = 1, 2
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            kap3(i,j,k) = in3(i,j,k,bi,bj,9)
            g3(i,j,k)   = in3(i,j,k,bi,bj,10)
           ENDDO
          ENDDO
         ENDDO
         IF ( n.EQ.1 ) THEN
          CALL IMPLDIFF( bi, bj, 0, sNx+1, 0, sNy+1, 1, kap3,
     &                   recip_hFacC(1-OLx,1-OLy,1,bi,bj), g3, myThid )
         ELSE
          CALL IMPLDIFF( bi, bj, 0, sNx+1, 0, sNy+1, -1, kap3,
     &                   recip_hFacC(1-OLx,1-OLy,1,bi,bj), g3, myThid )
         ENDIF
         DO k = 1, Nr
          CALL COL_PUT( g3(1-OLx,1-OLy,k),
     &                  out3(1-OLx,1-OLy,k,bi,bj,7+n) )
         ENDDO
        ENDDO
C-    CALC_IVDC (out3 10): do_oceanic_phys.F:867-875, k = Nr..2,
C     full range
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           sigR(i,j,k) = in3(i,j,k,bi,bj,11)
          ENDDO
         ENDDO
        ENDDO
        DO k = Nr, 2, -1
         CALL CALC_IVDC( bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy, k,
     &                   sigR, myTime, myIter, myThid )
        ENDDO
        DO k = 1, Nr
         CALL COL_PUT( IVDConvCount(1-OLx,1-OLy,k,bi,bj),
     &                 out3(1-OLx,1-OLy,k,bi,bj,10) )
        ENDDO
C-    INTEGRATE_FOR_W (out3 11): integr_continuity.F:268-273,
C     k = Nr..1, wFld = prior
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          rDh(i,j) = in2(i,j,bi,bj,1)
         ENDDO
        ENDDO
        DO k = Nr, 1, -1
         CALL INTEGRATE_FOR_W( bi, bj, k, u5, v5, u5, rDh,
     &                         w5, myIter, myThid )
        ENDDO
        DO k = 1, Nr
         CALL COL_PUT( w5(1-OLx,1-OLy,k,bi,bj),
     &                 out3(1-OLx,1-OLy,k,bi,bj,11) )
        ENDDO
C-    CALC_VISCOSITY (kapU, kapV): dynamics.F:386, range 0..sNx+1
        CALL CALC_VISCOSITY( bi, bj, 0, sNx+1, 0, sNy+1,
     &                       kRU, kRV, myThid )
        DO k = 1, Nr+1
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           kapU(i,j,k,bi,bj) = kRU(i,j,k)
           kapV(i,j,k,bi,bj) = kRV(i,j,k)
          ENDDO
         ENDDO
        ENDDO
C-    FIND_ALPHA (out2 1): calc_oce_mxlayer.F:113-115 arguments
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          r2d(i,j) = in3(i,j,1,bi,bj,15)
         ENDDO
        ENDDO
        CALL FIND_ALPHA( bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy, 1, 1,
     &                   r2d, myThid )
        CALL COL_PUT( r2d, out2(1-OLx,1-OLy,bi,bj,1) )
C-    CALC_OCE_MXLAYER (out2 2): do_oceanic_phys.F:895-900 with
C     rhoSurf = FIND_RHO_2D at kSrf = 1 (out3 1)
        CALL CALC_OCE_MXLAYER( out3(1-OLx,1-OLy,1,bi,bj,1), sigR,
     &                         bi, bj, myTime, myIter, myThid )
        CALL COL_PUT( hMixLayer(1-OLx,1-OLy,bi,bj),
     &                out2(1-OLx,1-OLy,bi,bj,2) )
C-    CALC_PHI_HYD (out3 12..15; out2 3 = phiHydF after k=Nr):
C     dynamics.F:478-488, k = 1..Nr, range 0..sNx+1,
C     rhoInSitu = FIND_RHO_2D full (out3 1); then totPhiHyd
C     (out3 16) as DIAGS_PHI_HYD leaves it
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          phF(i,j) = in3(i,j,1,bi,bj,17)
         ENDDO
        ENDDO
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           phC(i,j) = in3(i,j,k,bi,bj,17)
           dPX(i,j) = in3(i,j,k,bi,bj,17)
           dPY(i,j) = in3(i,j,k,bi,bj,17)
          ENDDO
         ENDDO
         CALL CALC_PHI_HYD( bi, bj, 0, sNx+1, 0, sNy+1, k,
     &                      phF, phC, dPX, dPY,
     &                      myTime, myIter, myThid )
         CALL COL_PUT( phC, out3(1-OLx,1-OLy,k,bi,bj,12) )
         CALL COL_PUT( phF, out3(1-OLx,1-OLy,k,bi,bj,13) )
         CALL COL_PUT( dPX, out3(1-OLx,1-OLy,k,bi,bj,14) )
         CALL COL_PUT( dPY, out3(1-OLx,1-OLy,k,bi,bj,15) )
        ENDDO
        DO k = 1, Nr
         CALL COL_PUT( totPhiHyd(1-OLx,1-OLy,k,bi,bj),
     &                 out3(1-OLx,1-OLy,k,bi,bj,16) )
        ENDDO
       ENDDO
      ENDDO

C--   4a. outputs
      OPEN( uOut, FILE='col_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL COL_WR( uOut, 'rhoFull', 3, sz3, out3(1-OLx,1-OLy,1,1,1,1) )
      CALL COL_WR( uOut, 'rhoPart', 3, sz3, out3(1-OLx,1-OLy,1,1,1,2) )
      CALL COL_WR( uOut, 'rhoP0', 3, sz3, out3(1-OLx,1-OLy,1,1,1,3) )
      CALL COL_WR( uOut, 'locPres', 3, sz3, out3(1-OLx,1-OLy,1,1,1,4) )
      CALL COL_WR( uOut, 'bulkMod', 3, sz3, out3(1-OLx,1-OLy,1,1,1,5) )
      CALL COL_WR( uOut, 'rhoScal', 3, sz3, out3(1-OLx,1-OLy,1,1,1,6) )
      CALL COL_WR( uOut, 'yTri', 3, sz3, out3(1-OLx,1-OLy,1,1,1,7) )
      CALL COL_WR( uOut, 'errTri', 0, nSx*nSy, errT )
      CALL COL_WR( uOut, 'gImpT', 3, sz3, out3(1-OLx,1-OLy,1,1,1,8) )
      CALL COL_WR( uOut, 'gImpM', 3, sz3, out3(1-OLx,1-OLy,1,1,1,9) )
      CALL COL_WR( uOut, 'ivdc', 3, sz3, out3(1-OLx,1-OLy,1,1,1,10) )
      CALL COL_WR( uOut, 'wFld', 3, sz3, out3(1-OLx,1-OLy,1,1,1,11) )
      CALL COL_WR( uOut, 'kappaRU', 4, sz3p, kapU )
      CALL COL_WR( uOut, 'kappaRV', 4, sz3p, kapV )
      CALL COL_WR( uOut, 'alpha', 2, sz2, out2(1-OLx,1-OLy,1,1,1) )
      CALL COL_WR( uOut, 'hMix', 2, sz2, out2(1-OLx,1-OLy,1,1,2) )
      CALL COL_WR( uOut, 'phiHydC', 3, sz3, out3(1-OLx,1-OLy,1,1,1,12) )
      CALL COL_WR( uOut, 'phiHydF', 3, sz3, out3(1-OLx,1-OLy,1,1,1,13) )
      CALL COL_WR( uOut, 'dPhiHydX', 3, sz3,
     &             out3(1-OLx,1-OLy,1,1,1,14) )
      CALL COL_WR( uOut, 'dPhiHydY', 3, sz3,
     &             out3(1-OLx,1-OLy,1,1,1,15) )
      CALL COL_WR( uOut, 'totPhiHyd', 3, sz3,
     &             out3(1-OLx,1-OLy,1,1,1,16) )
      CLOSE( uOut )

C--   4b. parameters and grid as the routines read them
      OPEN( uGrd, FILE='col_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL COL_WR( uGrd, 'hFacC', 3, sz3, hFacC )
      CALL COL_WR( uGrd, 'hFacW', 3, sz3, hFacW )
      CALL COL_WR( uGrd, 'hFacS', 3, sz3, hFacS )
      CALL COL_WR( uGrd, 'recip_hFacC', 3, sz3, recip_hFacC )
      CALL COL_WR( uGrd, 'recip_hFacW', 3, sz3, recip_hFacW )
      CALL COL_WR( uGrd, 'recip_hFacS', 3, sz3, recip_hFacS )
      CALL COL_WR( uGrd, 'maskC', 3, sz3, maskC )
      CALL COL_WR( uGrd, 'maskW', 3, sz3, maskW )
      CALL COL_WR( uGrd, 'maskS', 3, sz3, maskS )
#ifdef NONLIN_FRSURF
      CALL COL_WR( uGrd, 'h0FacC', 3, sz3, h0FacC )
#endif
      CALL COL_WR( uGrd, 'dxG', 2, sz2, dxG )
      CALL COL_WR( uGrd, 'dyG', 2, sz2, dyG )
      CALL COL_WR( uGrd, 'dxC', 2, sz2, dxC )
      CALL COL_WR( uGrd, 'dyC', 2, sz2, dyC )
      CALL COL_WR( uGrd, 'recip_dxC', 2, sz2, recip_dxC )
      CALL COL_WR( uGrd, 'recip_dyC', 2, sz2, recip_dyC )
      CALL COL_WR( uGrd, 'rA', 2, sz2, rA )
      CALL COL_WR( uGrd, 'recip_rA', 2, sz2, recip_rA )
      CALL COL_WR( uGrd, 'Ro_surf', 2, sz2, Ro_surf )
      CALL COL_WR( uGrd, 'R_low', 2, sz2, R_low )
      CALL COL_WR( uGrd, 'etaH', 2, sz2, etaH )
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          out2(i,j,bi,bj,1) = kLowC(i,j,bi,bj)
          out2(i,j,bi,bj,2) = kSurfC(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL COL_WR( uGrd, 'kLowC', 2, sz2, out2(1-OLx,1-OLy,1,1,1) )
      CALL COL_WR( uGrd, 'kSurfC', 2, sz2, out2(1-OLx,1-OLy,1,1,2) )
      CALL COL_WR( uGrd, 'rC', 1, Nr, rC )
      CALL COL_WR( uGrd, 'rF', 1, Nr+1, rF )
      CALL COL_WR( uGrd, 'drC', 1, Nr+1, drC )
      CALL COL_WR( uGrd, 'drF', 1, Nr, drF )
      CALL COL_WR( uGrd, 'recip_drC', 1, Nr+1, recip_drC )
      CALL COL_WR( uGrd, 'recip_drF', 1, Nr, recip_drF )
      CALL COL_WR( uGrd, 'deepFacC', 1, Nr, deepFacC )
      CALL COL_WR( uGrd, 'deepFac2F', 1, Nr+1, deepFac2F )
      CALL COL_WR( uGrd, 'recip_deepFac2C', 1, Nr, recip_deepFac2C )
      CALL COL_WR( uGrd, 'recip_deepFac2F', 1, Nr+1, recip_deepFac2F )
      CALL COL_WR( uGrd, 'rhoFacC', 1, Nr, rhoFacC )
      CALL COL_WR( uGrd, 'rhoFacF', 1, Nr+1, rhoFacF )
      CALL COL_WR( uGrd, 'recip_rhoFacC', 1, Nr, recip_rhoFacC )
      CALL COL_WR( uGrd, 'recip_rhoFacF', 1, Nr+1, recip_rhoFacF )
      CALL COL_WR( uGrd, 'gravFacC', 1, Nr, gravFacC )
      CALL COL_WR( uGrd, 'gravFacF', 1, Nr+1, gravFacF )
      CALL COL_WR( uGrd, 'viscArNr', 1, Nr, viscArNr )
      CALL COL_WR( uGrd, 'dTtracerLev', 1, Nr, dTtracerLev )
      CALL COL_WR( uGrd, 'tRef', 1, Nr, tRef )
      CALL COL_WR( uGrd, 'sRef', 1, Nr, sRef )
      CALL COL_WR( uGrd, 'pRef4EOS', 1, Nr, pRef4EOS )
      CALL COL_WR( uGrd, 'phiRef', 1, 2*Nr+1, phiRef )
      CALL COL_WR( uGrd, 'eosJMDCFw', 1, 6, eosJMDCFw )
      CALL COL_WR( uGrd, 'eosJMDCSw', 1, 9, eosJMDCSw )
      CALL COL_WR( uGrd, 'eosJMDCKFw', 1, 5, eosJMDCKFw )
      CALL COL_WR( uGrd, 'eosJMDCKSw', 1, 7, eosJMDCKSw )
      CALL COL_WR( uGrd, 'eosJMDCKP', 1, 14, eosJMDCKP )
      scal(1) = eosRefP0
      CALL COL_WR( uGrd, 'eosRefP0', 0, 1, scal )
      scal(1) = tAlpha
      CALL COL_WR( uGrd, 'tAlpha', 0, 1, scal )
      scal(1) = sBeta
      CALL COL_WR( uGrd, 'sBeta', 0, 1, scal )
      scal(1) = rhoNil
      CALL COL_WR( uGrd, 'rhoNil', 0, 1, scal )
      scal(1) = rhoConst
      CALL COL_WR( uGrd, 'rhoConst', 0, 1, scal )
      scal(1) = recip_rhoConst
      CALL COL_WR( uGrd, 'recip_rhoConst', 0, 1, scal )
      scal(1) = gravity
      CALL COL_WR( uGrd, 'gravity', 0, 1, scal )
      scal(1) = gravitySign
      CALL COL_WR( uGrd, 'gravitySign', 0, 1, scal )
      scal(1) = rkSign
      CALL COL_WR( uGrd, 'rkSign', 0, 1, scal )
      scal(1) = surf_pRef
      CALL COL_WR( uGrd, 'surf_pRef', 0, 1, scal )
      scal(1) = deltaTMom
      CALL COL_WR( uGrd, 'deltaTMom', 0, 1, scal )
      scal(1) = hMixCriteria
      CALL COL_WR( uGrd, 'hMixCriteria', 0, 1, scal )
      scal(1) = dRhoSmall
      CALL COL_WR( uGrd, 'dRhoSmall', 0, 1, scal )
      scal(1) = hMixSmooth
      CALL COL_WR( uGrd, 'hMixSmooth', 0, 1, scal )
      scal(1) = ivdc_kappa
      CALL COL_WR( uGrd, 'ivdc_kappa', 0, 1, scal )
      scal(1) = selectP_inEOS_Zc
      CALL COL_WR( uGrd, 'selectP_inEOS_Zc', 0, 1, scal )
      scal(1) = select_rStar
      CALL COL_WR( uGrd, 'select_rStar', 0, 1, scal )
      scal(1) = selectSigmaCoord
      CALL COL_WR( uGrd, 'selectSigmaCoord', 0, 1, scal )
      scal(1) = integr_GeoPot
      CALL COL_WR( uGrd, 'integr_GeoPot', 0, 1, scal )
      scal(1) = nonlinFreeSurf
      CALL COL_WR( uGrd, 'nonlinFreeSurf', 0, 1, scal )
      scal(1) = 0.
      IF ( rigidLid ) scal(1) = 1.
      CALL COL_WR( uGrd, 'rigidLid', 0, 1, scal )
      scal(1) = 0.
      IF ( usingPCoords ) scal(1) = 1.
      CALL COL_WR( uGrd, 'usingPCoords', 0, 1, scal )
      scal(1) = 0.
      IF ( useDiagnostics ) scal(1) = 1.
      CALL COL_WR( uGrd, 'useDiagnostics', 0, 1, scal )
      scal(1) = 0.
      IF ( uniformFreeSurfLev ) scal(1) = 1.
      CALL COL_WR( uGrd, 'uniformFreeSurfLev', 0, 1, scal )
      scal(1) = 0.
      IF ( quasiHydrostatic ) scal(1) = 1.
      CALL COL_WR( uGrd, 'quasiHydrostatic', 0, 1, scal )
      scal(1) = myIter
      CALL COL_WR( uGrd, 'myIter', 0, 1, scal )
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A,A)')
     &  'REPLAY COL harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed; eos ', equationOfState
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
       STOP 'REPLAY COL: bad input record'
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
