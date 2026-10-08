#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax RSTAR replay
C     | harness (M1 lane RSTAR; a copy of the Task 8 harness
C     | reference/replay). Not an MITgcm routine: it replaces
C     | model/src/the_main_loop.F in a genmake2 build of a
C     | verification experiment's code directory (NONLIN_FRSURF).
C     | THE_MODEL_MAIN has already called INITIALISE_FIXED, so
C     | GRID.h holds the real grid (hFac = h0Fac of
C     | INI_MASKS_ETC). This routine
C     |  1. writes the grid and parameters the r* routines read
C     |     (rstar_grid.bin), before any call;
C     |  2. reads synthetic inputs (rstar_in.bin) and sets the
C     |     SURFACE.h r* fields from them (INITIALISE_VARIA does
C     |     not run here);
C     |  3. A: CALC_R_STAR( etaA ) with etaA = etaBase +
C     |     spikeA*(Ro_surf-R_low) (columns pushed above hFacSup:
C     |     the counters, the warnings on errorMessageUnit =
C     |     STDERR.0000 and maxhFacC); UPDATE_R_STAR(.TRUE.);
C     |     UPDATE_R_STAR(.FALSE.);
C     |     B: vectorInvariantMomentum = .TRUE., selectKEscheme
C     |     = 1 (the simple average of CALC_R_STAR), CALC_R_STAR(
C     |     etaB ); RESET_NLFS_VARS;
C     |  4. writes every output (all points incl. halos) after
C     |     each call (rstar_out.bin).
C     | No time stepping; no STOP case (an abnormal end is not a
C     | run the oracle scripts accept).
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "SURFACE.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
      INTEGER sz2, sz3
      PARAMETER ( sz2 = (sNx+2*OLx)*(sNy+2*OLy)*nSx*nSy )
      PARAMETER ( sz3 = sz2*Nr )
      _RL etaA  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL etaB  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL spikeA(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL spikeB(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL prior (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL wrk   (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      Real*8 par(5)
      INTEGER uIn, uOut, uGrd, bi, bj, i, j
      LOGICAL firstCall
      SAVE firstCall
      DATA firstCall / .TRUE. /

      IF ( .NOT.firstCall ) RETURN
      firstCall = .FALSE.
      uIn  = 91
      uOut = 92
      uGrd = 93

C--   1. grid and parameters as the r* routines read them
      OPEN( uGrd, FILE='rstar_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          wrk(i,j,bi,bj) = kSurfC(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL RSR_WR( uGrd, 'kSurfC', 2, sz2, wrk )
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          wrk(i,j,bi,bj) = kSurfW(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL RSR_WR( uGrd, 'kSurfW', 2, sz2, wrk )
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          wrk(i,j,bi,bj) = kSurfS(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL RSR_WR( uGrd, 'kSurfS', 2, sz2, wrk )
      CALL RSR_WR( uGrd, 'Ro_surf', 2, sz2, Ro_surf )
      CALL RSR_WR( uGrd, 'R_low', 2, sz2, R_low )
      CALL RSR_WR( uGrd, 'recip_Rcol', 2, sz2, recip_Rcol )
      CALL RSR_WR( uGrd, 'rSurfW', 2, sz2, rSurfW )
      CALL RSR_WR( uGrd, 'rSurfS', 2, sz2, rSurfS )
      CALL RSR_WR( uGrd, 'rLowW', 2, sz2, rLowW )
      CALL RSR_WR( uGrd, 'rLowS', 2, sz2, rLowS )
      CALL RSR_WR( uGrd, 'rA', 2, sz2, rA )
      CALL RSR_WR( uGrd, 'recip_rAw', 2, sz2, recip_rAw )
      CALL RSR_WR( uGrd, 'recip_rAs', 2, sz2, recip_rAs )
      CALL RSR_WR( uGrd, 'maskC', 3, sz3, maskC )
      CALL RSR_WR( uGrd, 'maskW', 3, sz3, maskW )
      CALL RSR_WR( uGrd, 'maskS', 3, sz3, maskS )
      CALL RSR_WR( uGrd, 'h0FacC', 3, sz3, h0FacC )
      CALL RSR_WR( uGrd, 'h0FacW', 3, sz3, h0FacW )
      CALL RSR_WR( uGrd, 'h0FacS', 3, sz3, h0FacS )
      CALL RSR_WR( uGrd, 'hFacC', 3, sz3, hFacC )
      CALL RSR_WR( uGrd, 'hFacW', 3, sz3, hFacW )
      CALL RSR_WR( uGrd, 'hFacS', 3, sz3, hFacS )
      CALL RSR_WR( uGrd, 'recip_hFacC', 3, sz3, recip_hFacC )
      CALL RSR_WR( uGrd, 'recip_hFacW', 3, sz3, recip_hFacW )
      CALL RSR_WR( uGrd, 'recip_hFacS', 3, sz3, recip_hFacS )
      par(1) = hFacInf
      par(2) = hFacSup
      par(3) = deltaTFreeSurf
      par(4) = selectKEscheme
      par(5) = 0.
      IF ( vectorInvariantMomentum ) par(5) = 1.
      CALL RSR_WR( uGrd, 'params', 1, 5, par )
      CLOSE( uGrd )

C--   2. inputs
      OPEN( uIn, FILE='rstar_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL RSR_RD( uIn, 'etaBase', sz2, wrk )
      CALL RSR_RD( uIn, 'spikeA', sz2, spikeA )
      CALL RSR_RD( uIn, 'etaB', sz2, etaB )
      CALL RSR_RD( uIn, 'spikeB', sz2, spikeB )
      CALL RSR_RD( uIn, 'rStarFacC', sz2, rStarFacC )
      CALL RSR_RD( uIn, 'rStarFacW', sz2, rStarFacW )
      CALL RSR_RD( uIn, 'rStarFacS', sz2, rStarFacS )
      CALL RSR_RD( uIn, 'prior', sz2, prior )
      CLOSE( uIn )
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          etaA(i,j,bi,bj) = wrk(i,j,bi,bj)
     &      + spikeA(i,j,bi,bj)*(Ro_surf(i,j,bi,bj)-R_low(i,j,bi,bj))
          etaB(i,j,bi,bj) = etaB(i,j,bi,bj)
     &      + spikeB(i,j,bi,bj)*(Ro_surf(i,j,bi,bj)-R_low(i,j,bi,bj))
          rStarFacNm1C(i,j,bi,bj) = prior(i,j,bi,bj)
          rStarFacNm1W(i,j,bi,bj) = prior(i,j,bi,bj)
          rStarFacNm1S(i,j,bi,bj) = prior(i,j,bi,bj)
          rStarExpC(i,j,bi,bj) = prior(i,j,bi,bj)
          rStarExpW(i,j,bi,bj) = prior(i,j,bi,bj)
          rStarExpS(i,j,bi,bj) = prior(i,j,bi,bj)
          rStarDhCDt(i,j,bi,bj) = prior(i,j,bi,bj)
          rStarDhWDt(i,j,bi,bj) = prior(i,j,bi,bj)
          rStarDhSDt(i,j,bi,bj) = prior(i,j,bi,bj)
          pStarFacK(i,j,bi,bj) = prior(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   3./4. calls and outputs
      OPEN( uOut, FILE='rstar_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL RSR_WR( uOut, 'etaA', 2, sz2, etaA )
      CALL RSR_WR( uOut, 'etaB', 2, sz2, etaB )
      CALL CALC_R_STAR( etaA, myTime, 1, myThid )
      CALL RSR_WRS( uOut, 'A_' )
      CALL UPDATE_R_STAR( .TRUE., myTime, 1, myThid )
      CALL RSR_WRH( uOut, 'T_' )
      CALL UPDATE_R_STAR( .FALSE., myTime, 1, myThid )
      CALL RSR_WRH( uOut, 'F_' )
      vectorInvariantMomentum = .TRUE.
      selectKEscheme = 1
      CALL CALC_R_STAR( etaB, myTime, 2, myThid )
      CALL RSR_WRS( uOut, 'B_' )
      CALL RESET_NLFS_VARS( myTime, 2, myThid )
      CALL RSR_WR( uOut, 'R_pStarFacK', 2, sz2, pStarFacK )
      CLOSE( uOut )

      WRITE(standardMessageUnit,'(A,I4,A)')
     &  'REPLAY RSTAR harness: ', nSx*nSy,
     &  ' tiles; CALC_R_STAR x2, UPDATE_R_STAR x2, RESET_NLFS_VARS'
      RETURN
      END

C     the SURFACE.h r* fields, prefixed
      SUBROUTINE RSR_WRS( u, pre )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "SURFACE.h"
      INTEGER u
      CHARACTER*(2) pre
      INTEGER sz2
      PARAMETER ( sz2 = (sNx+2*OLx)*(sNy+2*OLy)*nSx*nSy )
      CALL RSR_WR( u, pre//'rStarFacC', 2, sz2, rStarFacC )
      CALL RSR_WR( u, pre//'rStarFacW', 2, sz2, rStarFacW )
      CALL RSR_WR( u, pre//'rStarFacS', 2, sz2, rStarFacS )
      CALL RSR_WR( u, pre//'rStarFacNm1C', 2, sz2, rStarFacNm1C )
      CALL RSR_WR( u, pre//'rStarFacNm1W', 2, sz2, rStarFacNm1W )
      CALL RSR_WR( u, pre//'rStarFacNm1S', 2, sz2, rStarFacNm1S )
      CALL RSR_WR( u, pre//'rStarExpC', 2, sz2, rStarExpC )
      CALL RSR_WR( u, pre//'rStarExpW', 2, sz2, rStarExpW )
      CALL RSR_WR( u, pre//'rStarExpS', 2, sz2, rStarExpS )
      CALL RSR_WR( u, pre//'rStarDhCDt', 2, sz2, rStarDhCDt )
      CALL RSR_WR( u, pre//'rStarDhWDt', 2, sz2, rStarDhWDt )
      CALL RSR_WR( u, pre//'rStarDhSDt', 2, sz2, rStarDhSDt )
      CALL RSR_WR( u, pre//'pStarFacK', 2, sz2, pStarFacK )
      RETURN
      END

C     the GRID.h hFac arrays, prefixed
      SUBROUTINE RSR_WRH( u, pre )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "GRID.h"
      INTEGER u
      CHARACTER*(2) pre
      INTEGER sz3
      PARAMETER ( sz3 = (sNx+2*OLx)*(sNy+2*OLy)*nSx*nSy*Nr )
      CALL RSR_WR( u, pre//'hFacC', 3, sz3, hFacC )
      CALL RSR_WR( u, pre//'hFacW', 3, sz3, hFacW )
      CALL RSR_WR( u, pre//'hFacS', 3, sz3, hFacS )
      CALL RSR_WR( u, pre//'recip_hFacC', 3, sz3, recip_hFacC )
      CALL RSR_WR( u, pre//'recip_hFacW', 3, sz3, recip_hFacW )
      CALL RSR_WR( u, pre//'recip_hFacS', 3, sz3, recip_hFacS )
      RETURN
      END

C     named records (big-endian stream): CHARACTER*16 name,
C     INTEGER rank, INTEGER n, REAL*8 a(n)
      SUBROUTINE RSR_RD( u, nm, n, a )
      IMPLICIT NONE
      INTEGER u, n
      CHARACTER*(*) nm
      Real*8 a(n)
      CHARACTER*16 got
      INTEGER rk, m
      READ( u ) got, rk, m
      IF ( got.NE.nm .OR. m.NE.n ) THEN
       WRITE(0,*) 'RSR_RD: want ', nm, n, ' got ', got, m
       STOP 'REPLAY RSTAR: bad input record'
      ENDIF
      READ( u ) a
      RETURN
      END

      SUBROUTINE RSR_WR( u, nm, rk, n, a )
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
