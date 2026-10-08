#include "GGL90_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax GGL90 replay
C     | harness (reference/replay_ggl90; M3 sub-lane GGL90, a copy
C     | of the COL harness reference/replay_col). Not an MITgcm
C     | routine: it replaces model/src/the_main_loop.F in a
C     | genmake2 build of a verification experiment's code dir.
C     | THE_MODEL_MAIN has already called INITIALISE_FIXED, so
C     | GRID.h, PARAMS.h and GGL90.h (GGL90_READPARMS) hold the
C     | experiment's real grid and parameters. This routine
C     |  1. writes the GGL90.h parameters as read (ggl_parm.bin),
C     |     calls GGL90_INIT_VARIA and writes its fields,
C     |  2. reads synthetic inputs written by Python
C     |     (ggl_in.bin: named big-endian stream records),
C     |  3. for every case (GGL90.h flags mxlMaxFlag, useLANGMUIR,
C     |     calcMeanVertShear, GGL90_dirichlet, mxlSurfFlag,
C     |     useIDEMIX set from the case table) sets GGL90TKE,
C     |     IDEMIX_E, GGL90viscArU/V, GGL90diffKr, uVel, vVel,
C     |     surfaceForcingU/V to the inputs and calls GGL90_CALC
C     |     for every tile as DO_OCEANIC_PHYS does,
C     |  4. calls GGL90_CALC_DIFF, GGL90_CALC_VISC,
C     |     GGL90_ADD_STOKESDRIFT (Langmuir builds) on priors and
C     |     GGL90_EXCHANGES after the last case,
C     |  5. writes every output (all points incl. halos) and the
C     |     grid/parameters the routines read (ggl_out.bin,
C     |     ggl_grid.bin).
C     | No time stepping. A second call returns at once.
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "FFIELDS.h"
#include "GGL90.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
      INTEGER nIn3, nCaseMax, nCaseVal
      PARAMETER ( nIn3 = 11, nCaseMax = 8, nCaseVal = 6 )
      INTEGER sz2, sz3, sz3p
      PARAMETER ( sz2 = (sNx+2*OLx)*(sNy+2*OLy)*nSx*nSy )
      PARAMETER ( sz3 = sz2*Nr, sz3p = sz2*(Nr+1) )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL in2 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,2)
      _RL o3  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      _RL kapU(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1,nSx,nSy)
      _RL kapV(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1,nSx,nSy)
      COMMON / GGLR_BUF / in3, in2, o3, kapU, kapV
      _RL sigR(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kap3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kRU(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1)
      _RL kRV(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1)
      _RL uR(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vR(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL uF(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vF(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL cases(nCaseVal,nCaseMax), scal(1)
      INTEGER nCase, ic
      INTEGER uIn, uOut, uGrd, uPar
      INTEGER bi, bj, i, j, k
      CHARACTER*16 nm
      LOGICAL firstCall
      SAVE firstCall
      DATA firstCall / .TRUE. /

      IF ( .NOT.firstCall ) RETURN
      firstCall = .FALSE.
      uIn  = 91
      uOut = 92
      uGrd = 93
      uPar = 94

C--   1. GGL90.h parameters as GGL90_READPARMS left them, GGL90_INIT_VARIA
      OPEN( uPar, FILE='ggl_parm.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL GGLR_W0( uPar, 'GGL90ck', GGL90ck )
      CALL GGLR_W0( uPar, 'GGL90ceps', GGL90ceps )
      CALL GGLR_W0( uPar, 'GGL90alpha', GGL90alpha )
      CALL GGLR_W0( uPar, 'GGL90m2', GGL90m2 )
      CALL GGLR_W0( uPar, 'GGL90diffTKEh', GGL90diffTKEh )
      CALL GGLR_W0( uPar, 'GGL90mixingLengt', GGL90mixingLengthMin )
      CALL GGLR_W0( uPar, 'GGL90TKEmin', GGL90TKEmin )
      CALL GGLR_W0( uPar, 'GGL90TKEsurfMin', GGL90TKEsurfMin )
      CALL GGLR_W0( uPar, 'GGL90TKEbottom', GGL90TKEbottom )
      CALL GGLR_W0( uPar, 'GGL90viscMax', GGL90viscMax )
      CALL GGLR_W0( uPar, 'GGL90diffMax', GGL90diffMax )
      CALL GGLR_W0( uPar, 'mxlMaxFlag', DBLE(mxlMaxFlag) )
      CALL GGLR_W0( uPar, 'adMxlMaxFlag', DBLE(adMxlMaxFlag) )
      CALL GGLR_WL( uPar, 'GGL90_dirichlet', GGL90_dirichlet )
      CALL GGLR_WL( uPar, 'mxlSurfFlag', mxlSurfFlag )
      CALL GGLR_WL( uPar, 'calcMeanVertShea', calcMeanVertShear )
      CALL GGLR_WL( uPar, 'useIDEMIX', useIDEMIX )
      CALL GGLR_WL( uPar, 'useLANGMUIR', useLANGMUIR )
#ifdef ALLOW_GGL90_IDEMIX
      CALL GGLR_W0( uPar, 'IDEMIX_tau_v', IDEMIX_tau_v )
      CALL GGLR_W0( uPar, 'IDEMIX_tau_h', IDEMIX_tau_h )
      CALL GGLR_W0( uPar, 'IDEMIX_gamma', IDEMIX_gamma )
      CALL GGLR_W0( uPar, 'IDEMIX_jstar', IDEMIX_jstar )
      CALL GGLR_W0( uPar, 'IDEMIX_mu0', IDEMIX_mu0 )
      CALL GGLR_W0( uPar, 'IDEMIX_frac_F_b', IDEMIX_frac_F_b )
      CALL GGLR_W0( uPar, 'IDEMIX_frac_F_s', IDEMIX_frac_F_s )
#endif
#ifdef ALLOW_GGL90_LANGMUIR
      CALL GGLR_W0( uPar, 'LC_Gamma', LC_Gamma )
      CALL GGLR_W0( uPar, 'LC_num', LC_num )
      CALL GGLR_W0( uPar, 'LC_lambda', LC_lambda )
#endif
      CALL GGL90_INIT_VARIA( myThid )
      CALL GGLR_WR( uPar, 'init_TKE', 3, sz3, GGL90TKE )
#ifdef ALLOW_GGL90_IDEMIX
      CALL GGLR_WR( uPar, 'init_E', 3, sz3, IDEMIX_E )
      CALL GGLR_WR( uPar, 'init_F_B', 2, sz2, IDEMIX_F_B )
      CALL GGLR_WR( uPar, 'init_F_S', 2, sz2, IDEMIX_F_S )
#endif
      CLOSE( uPar )

C--   2. inputs
      OPEN( uIn, FILE='ggl_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL GGLR_RD( uIn, 'TKE', sz3, in3(1-OLx,1-OLy,1,1,1, 1) )
      CALL GGLR_RD( uIn, 'IDEMIX_E', sz3, in3(1-OLx,1-OLy,1,1,1, 2) )
      CALL GGLR_RD( uIn, 'uVel', sz3, in3(1-OLx,1-OLy,1,1,1, 3) )
      CALL GGLR_RD( uIn, 'vVel', sz3, in3(1-OLx,1-OLy,1,1,1, 4) )
      CALL GGLR_RD( uIn, 'sigmaR', sz3, in3(1-OLx,1-OLy,1,1,1, 5) )
      CALL GGLR_RD( uIn, 'viscArUPrior', sz3, in3(1-OLx,1-OLy,1,1,1,6) )
      CALL GGLR_RD( uIn, 'viscArVPrior', sz3, in3(1-OLx,1-OLy,1,1,1,7) )
      CALL GGLR_RD( uIn, 'diffKrPrior', sz3, in3(1-OLx,1-OLy,1,1,1, 8) )
      CALL GGLR_RD( uIn, 'kappaRx', sz3, in3(1-OLx,1-OLy,1,1,1, 9) )
      CALL GGLR_RD( uIn, 'kappaRU', sz3, in3(1-OLx,1-OLy,1,1,1,10) )
      CALL GGLR_RD( uIn, 'kappaRV', sz3, in3(1-OLx,1-OLy,1,1,1,11) )
      CALL GGLR_RD( uIn, 'sfU', sz2, in2(1-OLx,1-OLy,1,1,1) )
      CALL GGLR_RD( uIn, 'sfV', sz2, in2(1-OLx,1-OLy,1,1,2) )
      CALL GGLR_RD( uIn, 'nCase', 1, scal )
      nCase = NINT(scal(1))
      IF ( nCase.GT.nCaseMax ) STOP 'REPLAY GGL90: too many cases'
      CALL GGLR_RD( uIn, 'cases', nCaseVal*nCase, cases )
      CLOSE( uIn )

      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uVel(i,j,k,bi,bj) = in3(i,j,k,bi,bj,3)
           vVel(i,j,k,bi,bj) = in3(i,j,k,bi,bj,4)
          ENDDO
         ENDDO
        ENDDO
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          surfaceForcingU(i,j,bi,bj) = in2(i,j,bi,bj,1)
          surfaceForcingV(i,j,bi,bj) = in2(i,j,bi,bj,2)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
#ifdef ALLOW_DIAGNOSTICS
      IF ( useDiagnostics )
     &  CALL DIAGNOSTICS_SWITCH_ONOFF( 1, myTime, myIter, myThid )
#endif

      OPEN( uOut, FILE='ggl_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
C--   3. the cases
      DO ic = 1, nCase
       mxlMaxFlag        = NINT(cases(1,ic))
       adMxlMaxFlag      = mxlMaxFlag
       useLANGMUIR       = cases(2,ic).NE.0.
       calcMeanVertShear = cases(3,ic).NE.0.
       GGL90_dirichlet   = cases(4,ic).NE.0.
       mxlSurfFlag       = cases(5,ic).NE.0.
       useIDEMIX         = cases(6,ic).NE.0.
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            GGL90TKE(i,j,k,bi,bj)     = in3(i,j,k,bi,bj,1)
#ifdef ALLOW_GGL90_IDEMIX
            IDEMIX_E(i,j,k,bi,bj)     = in3(i,j,k,bi,bj,2)
#endif
            GGL90viscArU(i,j,k,bi,bj) = in3(i,j,k,bi,bj,6)
            GGL90viscArV(i,j,k,bi,bj) = in3(i,j,k,bi,bj,7)
            GGL90diffKr(i,j,k,bi,bj)  = in3(i,j,k,bi,bj,8)
            sigR(i,j,k) = in3(i,j,k,bi,bj,5)
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
            sigR(i,j,k) = in3(i,j,k,bi,bj,5)
           ENDDO
          ENDDO
         ENDDO
         CALL GGL90_CALC( bi, bj, sigR, myTime, myIter, myThid )
        ENDDO
       ENDDO
       WRITE(nm,'(A,I1)') 'TKE_c', ic
       CALL GGLR_WR( uOut, nm, 3, sz3, GGL90TKE )
       WRITE(nm,'(A,I1)') 'viscArU_c', ic
       CALL GGLR_WR( uOut, nm, 3, sz3, GGL90viscArU )
       WRITE(nm,'(A,I1)') 'viscArV_c', ic
       CALL GGLR_WR( uOut, nm, 3, sz3, GGL90viscArV )
       WRITE(nm,'(A,I1)') 'diffKr_c', ic
       CALL GGLR_WR( uOut, nm, 3, sz3, GGL90diffKr )
#ifdef ALLOW_GGL90_IDEMIX
       WRITE(nm,'(A,I1)') 'E_c', ic
       CALL GGLR_WR( uOut, nm, 3, sz3, IDEMIX_E )
#endif
      ENDDO

C--   4. GGL90_EXCHANGES on the last case's fields, then the
C     per-level routines on the input priors
      CALL GGL90_EXCHANGES( myThid )
      CALL GGLR_WR( uOut, 'TKE_exch', 3, sz3, GGL90TKE )
#ifdef ALLOW_GGL90_IDEMIX
      CALL GGLR_WR( uOut, 'E_exch', 3, sz3, IDEMIX_E )
#endif
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           GGL90viscArU(i,j,k,bi,bj) = in3(i,j,k,bi,bj,6)
           GGL90viscArV(i,j,k,bi,bj) = in3(i,j,k,bi,bj,7)
           GGL90diffKr(i,j,k,bi,bj)  = in3(i,j,k,bi,bj,8)
           kap3(i,j,k) = in3(i,j,k,bi,bj,9)
           kRU(i,j,k) = in3(i,j,k,bi,bj,10)
           kRV(i,j,k) = in3(i,j,k,bi,bj,11)
          ENDDO
         ENDDO
        ENDDO
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          kRU(i,j,Nr+1) = 0.
          kRV(i,j,Nr+1) = 0.
         ENDDO
        ENDDO
C-    GGL90_CALC_DIFF: calc_3d_diffusivity.F:240-243 (kArg = 0, kSize = Nr)
        CALL GGL90_CALC_DIFF( bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     &                        0, Nr, kap3, myThid )
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           o3(i,j,k,bi,bj) = kap3(i,j,k)
          ENDDO
         ENDDO
        ENDDO
C-    GGL90_CALC_VISC: calc_viscosity.F:113-118, k = 1..Nr
        DO k = 1, Nr
         CALL GGL90_CALC_VISC( bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     &                         k, kRU, kRV, myThid )
        ENDDO
        DO k = 1, Nr+1
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           kapU(i,j,k,bi,bj) = kRU(i,j,k)
           kapV(i,j,k,bi,bj) = kRV(i,j,k)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL GGLR_WR( uOut, 'calcDiff', 3, sz3, o3 )
      CALL GGLR_WR( uOut, 'calcViscU', 4, sz3p, kapU )
      CALL GGLR_WR( uOut, 'calcViscV', 4, sz3p, kapV )
#ifdef ALLOW_GGL90_LANGMUIR
C-    GGL90_ADD_STOKESDRIFT: mom_fluxform.F:1085-1087 per level, with
C     the last case's useLANGMUIR set to .TRUE.; uRes/vRes priors = -1
      useLANGMUIR = .TRUE.
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uF(i,j) = uVel(i,j,k,bi,bj)
           vF(i,j) = vVel(i,j,k,bi,bj)
           uR(i,j) = -1.
           vR(i,j) = -1.
          ENDDO
         ENDDO
         CALL GGL90_ADD_STOKESDRIFT( uR, vR, uF, vF, k, bi, bj, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           o3(i,j,k,bi,bj)   = uR(i,j)
           in3(i,j,k,bi,bj,9) = vR(i,j)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL GGLR_WR( uOut, 'stokesU', 3, sz3, o3 )
      CALL GGLR_WR( uOut, 'stokesV', 3, sz3, in3(1-OLx,1-OLy,1,1,1,9) )
#endif
      CLOSE( uOut )

C--   5. grid and parameters as the routines read them
      OPEN( uGrd, FILE='ggl_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL GGLR_WR( uGrd, 'hFacC', 3, sz3, hFacC )
      CALL GGLR_WR( uGrd, 'hFacW', 3, sz3, hFacW )
      CALL GGLR_WR( uGrd, 'hFacS', 3, sz3, hFacS )
      CALL GGLR_WR( uGrd, 'recip_hFacC', 3, sz3, recip_hFacC )
      CALL GGLR_WR( uGrd, 'maskC', 3, sz3, maskC )
      CALL GGLR_WR( uGrd, 'maskW', 3, sz3, maskW )
      CALL GGLR_WR( uGrd, 'maskS', 3, sz3, maskS )
      CALL GGLR_WR( uGrd, 'dxF', 2, sz2, dxF )
      CALL GGLR_WR( uGrd, 'dyF', 2, sz2, dyF )
      CALL GGLR_WR( uGrd, 'dxG', 2, sz2, dxG )
      CALL GGLR_WR( uGrd, 'dyG', 2, sz2, dyG )
      CALL GGLR_WR( uGrd, 'recip_dxC', 2, sz2, recip_dxC )
      CALL GGLR_WR( uGrd, 'recip_dyC', 2, sz2, recip_dyC )
      CALL GGLR_WR( uGrd, 'recip_rA', 2, sz2, recip_rA )
      CALL GGLR_WR( uGrd, 'Ro_surf', 2, sz2, Ro_surf )
      CALL GGLR_WR( uGrd, 'R_low', 2, sz2, R_low )
      CALL GGLR_WR( uGrd, 'fCori', 2, sz2, fCori )
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          in2(i,j,bi,bj,1) = kLowC(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL GGLR_WR( uGrd, 'kLowC', 2, sz2, in2(1-OLx,1-OLy,1,1,1) )
      CALL GGLR_WR( uGrd, 'rC', 1, Nr, rC )
      CALL GGLR_WR( uGrd, 'rF', 1, Nr+1, rF )
      CALL GGLR_WR( uGrd, 'drC', 1, Nr+1, drC )
      CALL GGLR_WR( uGrd, 'drF', 1, Nr, drF )
      CALL GGLR_WR( uGrd, 'recip_drC', 1, Nr+1, recip_drC )
      CALL GGLR_WR( uGrd, 'recip_drF', 1, Nr, recip_drF )
      CALL GGLR_WR( uGrd, 'viscArNr', 1, Nr, viscArNr )
      CALL GGLR_WR( uGrd, 'diffKrNrS', 1, Nr, diffKrNrS )
      CALL GGLR_WR( uGrd, 'dTtracerLev', 1, Nr, dTtracerLev )
      CALL GGLR_W0( uGrd, 'gravity', gravity )
      CALL GGLR_W0( uGrd, 'gravitySign', gravitySign )
      CALL GGLR_W0( uGrd, 'recip_rhoConst', recip_rhoConst )
      CALL GGLR_WL( uGrd, 'usingPCoords', usingPCoords )
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A,I2,A)')
     &  'REPLAY GGL90 harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels, ', nCase, ' cases replayed'
      RETURN
      END

C     named records (big-endian stream): CHARACTER*16 name,
C     INTEGER rank, INTEGER n, REAL*8 a(n)
      SUBROUTINE GGLR_RD( u, nm, n, a )
      IMPLICIT NONE
      INTEGER u, n
      CHARACTER*(*) nm
      Real*8 a(n)
      CHARACTER*16 got
      INTEGER rk, m
      READ( u ) got, rk, m
      IF ( got.NE.nm .OR. m.NE.n ) THEN
       WRITE(0,*) 'GGLR_RD: want ', nm, n, ' got ', got, m
       STOP 'REPLAY GGL90: bad input record'
      ENDIF
      READ( u ) a
      RETURN
      END

      SUBROUTINE GGLR_WR( u, nm, rk, n, a )
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

      SUBROUTINE GGLR_W0( u, nm, v )
      IMPLICIT NONE
      INTEGER u
      CHARACTER*(*) nm
      Real*8 v, a(1)
      a(1) = v
      CALL GGLR_WR( u, nm, 0, 1, a )
      RETURN
      END

      SUBROUTINE GGLR_WL( u, nm, l )
      IMPLICIT NONE
      INTEGER u
      CHARACTER*(*) nm
      LOGICAL l
      Real*8 a(1)
      a(1) = 0.
      IF ( l ) a(1) = 1.
      CALL GGLR_WR( u, nm, 0, 1, a )
      RETURN
      END
