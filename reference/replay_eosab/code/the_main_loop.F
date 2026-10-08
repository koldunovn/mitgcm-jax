#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax EOSAB replay
C     | harness (reference/replay_eosab; M3 lane EOSAB, a copy
C     | of the COLMIX harness reference/replay_colmix). Not an
C     | MITgcm routine: it replaces model/src/the_main_loop.F in
C     | a genmake2 build of vermix/code. THE_MODEL_MAIN has run
C     | INITIALISE_FIXED (grid, parameters, INI_EOS). This
C     | routine
C     |  0. writes the namelist EOS's coefficients (cm_orig.bin),
C     |  1. reads nSamp synthetic samples (cm_in.bin) of theta,
C     |     salt, totPhiHyd and the priors of alphaLoc, betaLoc,
C     |  2. for each pass p = 1..nPass (kRef and index range:
C     |     1: kRef = k, full tile; 2: kRef = 1, full tile;
C     |     3: kRef = Nr+1-k, iMin..iMax = 2-OLx..sNx+OLx-1,
C     |        jMin..jMax = 1-OLy..sNy), each sample and each
C     |     level k = 1..Nr: FIND_ALPHA and FIND_BETA on the
C     |     prior (eosType of the namelist: MDJWF in vermix),
C     |  3. writes every output, all points incl. halos
C     |     (cm_out.bin) and the parameters the routines read
C     |     (cm_grid.bin).
C     | No time stepping.
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "EOS.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
      INTEGER nSamp, nPass, nIn3
      PARAMETER ( nSamp = 64, nPass = 3, nIn3 = 5 )
      INTEGER sz2, sz3
      PARAMETER ( sz2 = (sNx+2*OLx)*(sNy+2*OLy)*nSx*nSy )
      PARAMETER ( sz3 = sz2*Nr )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nSamp,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nSamp,nPass,2)
      COMMON / EA_BUF / in3, out3
      _RL aLoc(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL bLoc(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL scal(1)
      INTEGER iLo(nPass), iHi(nPass), jLo(nPass), jHi(nPass)
      INTEGER uIn, uOut, uGrd
      INTEGER bi, bj, i, j, k, n, p, kR
      CHARACTER*16 nm
      CHARACTER*2 sfx(3)
      DATA sfx / 'p1', 'p2', 'p3' /
      LOGICAL firstCall
      SAVE firstCall
      DATA firstCall / .TRUE. /

      IF ( .NOT.firstCall ) RETURN
      firstCall = .FALSE.
      uIn  = 91
      uOut = 92
      uGrd = 93
      IF ( equationOfState.NE.'MDJWF' )
     &  STOP 'REPLAY EOSAB: the namelist eosType is not MDJWF'
      iLo(1) = 1-OLx
      iHi(1) = sNx+OLx
      jLo(1) = 1-OLy
      jHi(1) = sNy+OLy
      iLo(2) = 1-OLx
      iHi(2) = sNx+OLx
      jLo(2) = 1-OLy
      jHi(2) = sNy+OLy
      iLo(3) = 2-OLx
      iHi(3) = sNx+OLx-1
      jLo(3) = 1-OLy
      jHi(3) = sNy

C--   0. the namelist EOS's coefficients (cm_orig.bin)
      OPEN( uGrd, FILE='cm_orig.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL CM_WR( uGrd, 'eosMDJWFnum', 1, 12, eosMDJWFnum )
      CALL CM_WR( uGrd, 'eosMDJWFden', 1, 13, eosMDJWFden )
      CALL CM_SC( uGrd, 'eosRefP0', eosRefP0 )
      CLOSE( uGrd )

C--   1. inputs (order: reference/replay_eosab/replay_io.py IN3)
      OPEN( uIn, FILE='cm_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL CM_RD( uIn, 'theta', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 1))
      CALL CM_RD( uIn, 'salt', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 2) )
      CALL CM_RD( uIn, 'totPhiHyd', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1, 3) )
      CALL CM_RD( uIn, 'alphaPrior', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1, 4) )
      CALL CM_RD( uIn, 'betaPrior', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1, 5) )
      CLOSE( uIn )

C--   2. passes x samples x levels
      DO p = 1, nPass
       DO n = 1, nSamp
        DO bj = 1, nSy
         DO bi = 1, nSx
          DO k = 1, Nr
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             theta(i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 1)
             salt (i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 2)
             totPhiHyd(i,j,k,bi,bj) = in3(i,j,k,bi,bj,n, 3)
            ENDDO
           ENDDO
          ENDDO
          DO k = 1, Nr
           kR = k
           IF ( p.EQ.2 ) kR = 1
           IF ( p.EQ.3 ) kR = Nr+1-k
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             aLoc(i,j) = in3(i,j,k,bi,bj,n, 4)
             bLoc(i,j) = in3(i,j,k,bi,bj,n, 5)
            ENDDO
           ENDDO
           CALL FIND_ALPHA( bi, bj, iLo(p), iHi(p), jLo(p), jHi(p),
     &                      k, kR, aLoc, myThid )
           CALL FIND_BETA(  bi, bj, iLo(p), iHi(p), jLo(p), jHi(p),
     &                      k, kR, bLoc, myThid )
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             out3(i,j,k,bi,bj,n,p,1) = aLoc(i,j)
             out3(i,j,k,bi,bj,n,p,2) = bLoc(i,j)
            ENDDO
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   3. outputs (all points incl. halos)
      OPEN( uOut, FILE='cm_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      DO p = 1, nPass
       nm = 'alpha_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,1) )
       nm = 'beta_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,2) )
      ENDDO
      CLOSE( uOut )

C--   parameters the routines read (cm_grid.bin)
      OPEN( uGrd, FILE='cm_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL CM_WR( uGrd, 'rC', 1, Nr, rC )
      CALL CM_WR( uGrd, 'pRef4EOS', 1, Nr, pRef4EOS )
      CALL CM_WR( uGrd, 'phiRef', 1, 2*Nr+1, phiRef )
      CALL CM_SC( uGrd, 'eosRefP0', eosRefP0 )
      CALL CM_SC( uGrd, 'rhoNil', rhoNil )
      CALL CM_SC( uGrd, 'rhoConst', rhoConst )
      CALL CM_SC( uGrd, 'surf_pRef', surf_pRef )
      scal(1) = selectP_inEOS_Zc
      CALL CM_SC( uGrd, 'selectP_inEOS_Zc', scal(1) )
      scal(1) = 0.
      IF ( usingPCoords ) scal(1) = 1.
      CALL CM_SC( uGrd, 'usingPCoords', scal(1) )
      scal(1) = nSamp
      CALL CM_SC( uGrd, 'nSamp', scal(1) )
      scal(1) = nPass
      CALL CM_SC( uGrd, 'nPass', scal(1) )
      DO p = 1, nPass
       scal(1) = iLo(p)
       CALL CM_SC( uGrd, 'iMin_'//sfx(p), scal(1) )
       scal(1) = iHi(p)
       CALL CM_SC( uGrd, 'iMax_'//sfx(p), scal(1) )
       scal(1) = jLo(p)
       CALL CM_SC( uGrd, 'jMin_'//sfx(p), scal(1) )
       scal(1) = jHi(p)
       CALL CM_SC( uGrd, 'jMax_'//sfx(p), scal(1) )
      ENDDO
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A,I3,A,A)')
     &  'REPLAY EOSAB harness: ', nSamp, ' samples x', Nr,
     &  ' levels x', nPass, ' passes; eos ', equationOfState
      RETURN
      END

C     named records (big-endian stream): CHARACTER*16 name,
C     INTEGER rank, INTEGER n, REAL*8 a(n)
      SUBROUTINE CM_RD( u, nm, n, a )
      IMPLICIT NONE
      INTEGER u, n
      CHARACTER*(*) nm
      Real*8 a(n)
      CHARACTER*16 got
      INTEGER rk, m
      READ( u ) got, rk, m
      IF ( got.NE.nm .OR. m.NE.n ) THEN
       WRITE(0,*) 'CM_RD: want ', nm, n, ' got ', got, m
       STOP 'REPLAY EOSAB: bad input record'
      ENDIF
      READ( u ) a
      RETURN
      END

      SUBROUTINE CM_WR( u, nm, rk, n, a )
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

      SUBROUTINE CM_SC( u, nm, x )
      IMPLICIT NONE
      INTEGER u
      CHARACTER*(*) nm
      Real*8 x, a(1)
      a(1) = x
      CALL CM_WR( u, nm, 0, 1, a )
      RETURN
      END
