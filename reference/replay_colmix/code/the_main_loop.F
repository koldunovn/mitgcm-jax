#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax COLMIX replay
C     | harness (reference/replay_colmix; M3 sub-lane COLMIX, a
C     | copy of the COL harness reference/replay_col). Not an
C     | MITgcm routine: it replaces model/src/the_main_loop.F in
C     | a genmake2 build of vermix/code. THE_MODEL_MAIN has run
C     | INITIALISE_FIXED (grid, parameters, INI_EOS, the
C     | packages' READPARMS). This routine
C     |  0. writes what INITIALISE_FIXED left (cm_orig.bin:
C     |     masks, kLowC, kSurfC, viscArNr, diffKrNrS, the
C     |     namelist EOS's coefficients),
C     |  1. reads nSamp synthetic samples (cm_in.bin) of the
C     |     state and of the grid fields the routines read
C     |     (maskC/W/S, kLowC, kSurfC) and the priors of every
C     |     output, and overrides viscArNr, diffKrNrS,
C     |  2. calls the INIT_VARIA of the active package (output
C     |     records *_ini), then for each pass p = 1..nPass
C     |     (1: eosType of the namelist, MDJWF in vermix;
C     |      2: eosType = 'JMD95Z' + INI_EOS; 3: as 2 with
C     |      useGCMwVel = .TRUE. (OPPS))
C     |     and each sample: PP81_CALC, PP81_CALC_DIFF (kArg 0
C     |     and kArg = k), PP81_CALC_VISC (k = 1..Nr) if usePP81;
C     |     the MY82 routines alike if useMY82; OPPS_INTERFACE on
C     |     the full tile and OPPS_CALC per column (the
C     |     convection count) if useOPPS,
C     |  3. writes every output, all points incl. halos
C     |     (cm_out.bin) and the parameters and grid the routines
C     |     read (cm_grid.bin).
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
#ifdef ALLOW_PP81
#include "PP81.h"
#endif
#ifdef ALLOW_MY82
#include "MY82.h"
#endif
#ifdef ALLOW_OPPS
#include "OPPS.h"
#endif
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
      INTEGER nSamp, nPass, nIn3, nIn2, nOut3, nOut2
      PARAMETER ( nSamp = 48, nPass = 3 )
      PARAMETER ( nIn3 = 16, nIn2 = 3, nOut3 = 9, nOut2 = 1 )
      INTEGER sz2, sz3
      PARAMETER ( sz2 = (sNx+2*OLx)*(sNy+2*OLy)*nSx*nSy )
      PARAMETER ( sz3 = sz2*Nr )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nSamp,nIn3)
      _RL in2 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,nSamp,nIn2)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nSamp,nPass,
     &         nOut3)
      _RL out2(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,nSamp,nPass,nOut2)
      _RL ini3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,2)
      _RL ini2(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      COMMON / CM_BUF / in3, in2, out3, out2, ini3, ini2
      _RL sigR(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kX0 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kXk (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kU  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kV  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL cnt (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL trLoc(Nr,2), wLoc(Nr)
      _RL vOrig(Nr), scal(1)
      INTEGER uIn, uOut, uGrd
      INTEGER bi, bj, i, j, k, n, p, nPassRun
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
      nPassRun = 2
#ifdef ALLOW_OPPS
      IF ( useOPPS ) nPassRun = 3
#endif

C--   0. what INITIALISE_FIXED left, before any override (cm_orig.bin):
C     masks, kLowC, kSurfC, viscArNr, diffKrNrS, EOS.h (namelist eosType)
      OPEN( uGrd, FILE='cm_orig.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL CM_WR( uGrd, 'maskC', 3, sz3, maskC )
      CALL CM_WR( uGrd, 'maskW', 3, sz3, maskW )
      CALL CM_WR( uGrd, 'maskS', 3, sz3, maskS )
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          in2(i,j,bi,bj,1,1) = kLowC(i,j,bi,bj)
          in2(i,j,bi,bj,1,2) = kSurfC(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL CM_WR( uGrd, 'kLowC', 2, sz2, in2(1-OLx,1-OLy,1,1,1,1) )
      CALL CM_WR( uGrd, 'kSurfC', 2, sz2, in2(1-OLx,1-OLy,1,1,1,2) )
      CALL CM_WR( uGrd, 'viscArNr', 1, Nr, viscArNr )
      CALL CM_WR( uGrd, 'diffKrNrS', 1, Nr, diffKrNrS )
      CALL CM_WR( uGrd, 'eosMDJWFnum', 1, 12, eosMDJWFnum )
      CALL CM_WR( uGrd, 'eosMDJWFden', 1, 13, eosMDJWFden )
      CALL CM_SC( uGrd, 'eosRefP0', eosRefP0 )
      CLOSE( uGrd )

C--   1. inputs (order: reference/replay_colmix/replay_io.py IN3, IN2)
      OPEN( uIn, FILE='cm_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL CM_RD( uIn, 'theta', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 1))
      CALL CM_RD( uIn, 'salt', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 2) )
      CALL CM_RD( uIn, 'uVel', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 3) )
      CALL CM_RD( uIn, 'vVel', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 4) )
      CALL CM_RD( uIn, 'wVel', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 5) )
      CALL CM_RD( uIn, 'totPhiHyd', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1, 6) )
      CALL CM_RD( uIn, 'maskC', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 7) )
      CALL CM_RD( uIn, 'maskW', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 8) )
      CALL CM_RD( uIn, 'maskS', sz3*nSamp, in3(1-OLx,1-OLy,1,1,1,1, 9) )
      CALL CM_RD( uIn, 'viscAr0', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1,10) )
      CALL CM_RD( uIn, 'diffKr0', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1,11) )
      CALL CM_RD( uIn, 'kappaRx0', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1,12) )
      CALL CM_RD( uIn, 'kappaRU0', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1,13) )
      CALL CM_RD( uIn, 'kappaRV0', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1,14) )
      CALL CM_RD( uIn, 'viscAr1', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1,15) )
      CALL CM_RD( uIn, 'diffKr1', sz3*nSamp,
     &            in3(1-OLx,1-OLy,1,1,1,1,16) )
      CALL CM_RD( uIn, 'kLowC', sz2*nSamp, in2(1-OLx,1-OLy,1,1,1, 1) )
      CALL CM_RD( uIn, 'kSurfC', sz2*nSamp, in2(1-OLx,1-OLy,1,1,1, 2) )
      CALL CM_RD( uIn, 'MYhbl0', sz2*nSamp, in2(1-OLx,1-OLy,1,1,1, 3) )
C     overrides of PARAMS.h vectors (the readparms values are kept in
C     the dump: viscArNr_orig, RiLimit)
      DO k = 1, Nr
       vOrig(k) = viscArNr(k)
      ENDDO
      CALL CM_RD( uIn, 'viscArNr', Nr, viscArNr )
      CALL CM_RD( uIn, 'diffKrNrS', Nr, diffKrNrS )
      CLOSE( uIn )

#ifdef ALLOW_DIAGNOSTICS
C     pkg/diagnostics as at the first time step: fills are allowed
      IF ( useDiagnostics )
     &  CALL DIAGNOSTICS_SWITCH_ONOFF( 1, myTime, myIter, myThid )
#endif

C--   2a. the active package's INIT_VARIA (after the overrides)
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           ini3(i,j,k,bi,bj,1) = 0.
           ini3(i,j,k,bi,bj,2) = 0.
          ENDDO
         ENDDO
        ENDDO
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          ini2(i,j,bi,bj) = 0.
         ENDDO
        ENDDO
       ENDDO
      ENDDO
#ifdef ALLOW_PP81
      IF ( usePP81 ) THEN
       CALL PP81_INIT_VARIA( myThid )
       CALL CM_CP3( PPviscAr, ini3(1-OLx,1-OLy,1,1,1,1) )
       CALL CM_CP3( PPdiffKr, ini3(1-OLx,1-OLy,1,1,1,2) )
      ENDIF
#endif
#ifdef ALLOW_MY82
      IF ( useMY82 ) THEN
       CALL MY82_INIT_VARIA( myThid )
       CALL CM_CP3( MYviscAr, ini3(1-OLx,1-OLy,1,1,1,1) )
       CALL CM_CP3( MYdiffKr, ini3(1-OLx,1-OLy,1,1,1,2) )
       CALL CM_CP2( MYhbl, ini2 )
      ENDIF
#endif

C--   2b. replay
      DO p = 1, nPassRun
       IF ( p.EQ.2 ) THEN
        eosType = 'JMD95Z'
        CALL INI_EOS( myThid )
       ENDIF
#ifdef ALLOW_OPPS
       IF ( p.EQ.3 ) useGCMwVel = .TRUE.
#endif
       DO n = 1, nSamp
        DO bj = 1, nSy
         DO bi = 1, nSx
C     state and grid of sample n
          DO k = 1, Nr
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             theta(i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 1)
             salt (i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 2)
             uVel (i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 3)
             vVel (i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 4)
             wVel (i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 5)
             totPhiHyd(i,j,k,bi,bj) = in3(i,j,k,bi,bj,n, 6)
             maskC(i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 7)
             maskW(i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 8)
             maskS(i,j,k,bi,bj)     = in3(i,j,k,bi,bj,n, 9)
             sigR(i,j,k) = 0.
             kX0(i,j,k) = in3(i,j,k,bi,bj,n,12)
             kXk(i,j,k) = in3(i,j,k,bi,bj,n,12)
             kU (i,j,k) = in3(i,j,k,bi,bj,n,13)
             kV (i,j,k) = in3(i,j,k,bi,bj,n,14)
            ENDDO
           ENDDO
          ENDDO
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            kLowC (i,j,bi,bj) = NINT( in2(i,j,bi,bj,n,1) )
            kSurfC(i,j,bi,bj) = NINT( in2(i,j,bi,bj,n,2) )
           ENDDO
          ENDDO
#ifdef ALLOW_PP81
          IF ( usePP81 ) THEN
           DO k = 1, Nr
            DO j = 1-OLy, sNy+OLy
             DO i = 1-OLx, sNx+OLx
              PPviscAr(i,j,k,bi,bj) = in3(i,j,k,bi,bj,n,10)
              PPdiffKr(i,j,k,bi,bj) = in3(i,j,k,bi,bj,n,11)
             ENDDO
            ENDDO
           ENDDO
           CALL PP81_CALC( bi, bj, sigR, myTime, myIter, myThid )
           CALL PP81_CALC_DIFF( bi,bj, 1-OLx,sNx+OLx,1-OLy,sNy+OLy,
     &                          0, Nr, kX0, myThid )
           DO k = 1, Nr
            CALL PP81_CALC_DIFF( bi,bj, 1-OLx,sNx+OLx,1-OLy,sNy+OLy,
     &                           k, Nr, kXk, myThid )
            CALL PP81_CALC_VISC( bi,bj, 2-OLx,sNx+OLx,2-OLy,sNy+OLy,
     &                           k, kU, kV, myThid )
           ENDDO
           CALL CM_CP3( PPviscAr(1-OLx,1-OLy,1,bi,bj),
     &                  out3(1-OLx,1-OLy,1,bi,bj,n,p,1) )
           CALL CM_CP3( PPdiffKr(1-OLx,1-OLy,1,bi,bj),
     &                  out3(1-OLx,1-OLy,1,bi,bj,n,p,2) )
          ENDIF
#endif
#ifdef ALLOW_MY82
          IF ( useMY82 ) THEN
           DO k = 1, Nr
            DO j = 1-OLy, sNy+OLy
             DO i = 1-OLx, sNx+OLx
              MYviscAr(i,j,k,bi,bj) = in3(i,j,k,bi,bj,n,15)
              MYdiffKr(i,j,k,bi,bj) = in3(i,j,k,bi,bj,n,16)
             ENDDO
            ENDDO
           ENDDO
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             MYhbl(i,j,bi,bj) = in2(i,j,bi,bj,n,3)
            ENDDO
           ENDDO
           CALL MY82_CALC( bi, bj, sigR, myTime, myIter, myThid )
           CALL MY82_CALC_DIFF( bi,bj, 1-OLx,sNx+OLx,1-OLy,sNy+OLy,
     &                          0, Nr, kX0, myThid )
           DO k = 1, Nr
            CALL MY82_CALC_DIFF( bi,bj, 1-OLx,sNx+OLx,1-OLy,sNy+OLy,
     &                           k, Nr, kXk, myThid )
            CALL MY82_CALC_VISC( bi,bj, 2-OLx,sNx+OLx,2-OLy,sNy+OLy,
     &                           k, kU, kV, myThid )
           ENDDO
           CALL CM_CP3( MYviscAr(1-OLx,1-OLy,1,bi,bj),
     &                  out3(1-OLx,1-OLy,1,bi,bj,n,p,1) )
           CALL CM_CP3( MYdiffKr(1-OLx,1-OLy,1,bi,bj),
     &                  out3(1-OLx,1-OLy,1,bi,bj,n,p,2) )
           CALL CM_CP2( MYhbl(1-OLx,1-OLy,bi,bj),
     &                  out2(1-OLx,1-OLy,bi,bj,n,p,1) )
          ENDIF
#endif
          CALL CM_CP3( kX0, out3(1-OLx,1-OLy,1,bi,bj,n,p,3) )
          CALL CM_CP3( kXk, out3(1-OLx,1-OLy,1,bi,bj,n,p,4) )
          CALL CM_CP3( kU,  out3(1-OLx,1-OLy,1,bi,bj,n,p,5) )
          CALL CM_CP3( kV,  out3(1-OLx,1-OLy,1,bi,bj,n,p,6) )
#ifdef ALLOW_OPPS
          IF ( useOPPS ) THEN
           CALL OPPS_INTERFACE( bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     &                          myTime, myIter, myThid )
           CALL CM_CP3( theta(1-OLx,1-OLy,1,bi,bj),
     &                  out3(1-OLx,1-OLy,1,bi,bj,n,p,7) )
           CALL CM_CP3( salt(1-OLx,1-OLy,1,bi,bj),
     &                  out3(1-OLx,1-OLy,1,bi,bj,n,p,8) )
C     OPPS_CALC column by column as OPPS_INTERFACE calls it (its local
C     OPPSconvectCount goes to DIAGNOSTICS_FILL only)
           DO k = 1, Nr
            DO j = 1-OLy, sNy+OLy
             DO i = 1-OLx, sNx+OLx
              cnt(i,j,k) = 0. _d 0
             ENDDO
            ENDDO
           ENDDO
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             IF ( kSurfC(i,j,bi,bj) .LE. Nr ) THEN
              DO k = 1, Nr
               trLoc(k,1) = in3(i,j,k,bi,bj,n,1)
               trLoc(k,2) = in3(i,j,k,bi,bj,n,2)
               IF ( useGCMwVel ) THEN
                wLoc(k) = in3(i,j,k,bi,bj,n,5)
               ELSE
                wLoc(k) = - VERTICAL_VELOCITY
               ENDIF
              ENDDO
              CALL OPPS_CALC( trLoc, cnt, wLoc, kLowC(i,j,bi,bj), 2, 2,
     &                        i, j, bi, bj, myTime, myIter, myThid )
             ENDIF
            ENDDO
           ENDDO
           CALL CM_CP3( cnt, out3(1-OLx,1-OLy,1,bi,bj,n,p,9) )
          ENDIF
#endif
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   3a. outputs: out3 fields x pass (n, p order: [pass][sample])
      OPEN( uOut, FILE='cm_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL CM_WR( uOut, 'ini_vis', 3, sz3, ini3(1-OLx,1-OLy,1,1,1,1) )
      CALL CM_WR( uOut, 'ini_dif', 3, sz3, ini3(1-OLx,1-OLy,1,1,1,2) )
      CALL CM_WR( uOut, 'ini_hbl', 2, sz2, ini2 )
      DO p = 1, nPassRun
       nm = 'vis_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,1) )
       nm = 'dif_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,2) )
       nm = 'kapX0_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,3) )
       nm = 'kapXk_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,4) )
       nm = 'kapU_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,5) )
       nm = 'kapV_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,6) )
       nm = 'theta_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,7) )
       nm = 'salt_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,8) )
       nm = 'cnt_'//sfx(p)
       CALL CM_WR( uOut, nm, 4, sz3*nSamp,
     &             out3(1-OLx,1-OLy,1,1,1,1,p,9) )
       nm = 'hbl_'//sfx(p)
       CALL CM_WR( uOut, nm, 3, sz2*nSamp,
     &             out2(1-OLx,1-OLy,1,1,1,p,1) )
      ENDDO
      CLOSE( uOut )

C--   3b. parameters and grid as the routines read them (EOS.h after
C     the last pass: JMD95Z)
      OPEN( uGrd, FILE='cm_grid.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL CM_WR( uGrd, 'rC', 1, Nr, rC )
      CALL CM_WR( uGrd, 'rF', 1, Nr+1, rF )
      CALL CM_WR( uGrd, 'drC', 1, Nr+1, drC )
      CALL CM_WR( uGrd, 'drF', 1, Nr, drF )
      CALL CM_WR( uGrd, 'recip_drC', 1, Nr+1, recip_drC )
      CALL CM_WR( uGrd, 'recip_drF', 1, Nr, recip_drF )
      CALL CM_WR( uGrd, 'viscArNr', 1, Nr, viscArNr )
      CALL CM_WR( uGrd, 'viscArNr_orig', 1, Nr, vOrig )
      CALL CM_WR( uGrd, 'diffKrNrS', 1, Nr, diffKrNrS )
      CALL CM_WR( uGrd, 'dTtracerLev', 1, Nr, dTtracerLev )
      CALL CM_WR( uGrd, 'tRef', 1, Nr, tRef )
      CALL CM_WR( uGrd, 'sRef', 1, Nr, sRef )
      CALL CM_WR( uGrd, 'pRef4EOS', 1, Nr, pRef4EOS )
      CALL CM_WR( uGrd, 'phiRef', 1, 2*Nr+1, phiRef )
      CALL CM_WR( uGrd, 'eosJMDCFw', 1, 6, eosJMDCFw )
      CALL CM_WR( uGrd, 'eosJMDCSw', 1, 9, eosJMDCSw )
      CALL CM_WR( uGrd, 'eosJMDCKFw', 1, 5, eosJMDCKFw )
      CALL CM_WR( uGrd, 'eosJMDCKSw', 1, 7, eosJMDCKSw )
      CALL CM_WR( uGrd, 'eosJMDCKP', 1, 14, eosJMDCKP )
      CALL CM_SC( uGrd, 'eosRefP0', eosRefP0 )
      CALL CM_SC( uGrd, 'tAlpha', tAlpha )
      CALL CM_SC( uGrd, 'sBeta', sBeta )
      CALL CM_SC( uGrd, 'rhoNil', rhoNil )
      CALL CM_SC( uGrd, 'rhoConst', rhoConst )
      CALL CM_SC( uGrd, 'gravity', gravity )
      CALL CM_SC( uGrd, 'mass2rUnit', mass2rUnit )
      CALL CM_SC( uGrd, 'surf_pRef', surf_pRef )
      scal(1) = selectP_inEOS_Zc
      CALL CM_SC( uGrd, 'selectP_inEOS_Zc', scal(1) )
      scal(1) = 0.
      IF ( usingPCoords ) scal(1) = 1.
      CALL CM_SC( uGrd, 'usingPCoords', scal(1) )
      scal(1) = nSamp
      CALL CM_SC( uGrd, 'nSamp', scal(1) )
      scal(1) = nPassRun
      CALL CM_SC( uGrd, 'nPass', scal(1) )
#ifdef ALLOW_PP81
      IF ( usePP81 ) THEN
       scal(1) = PPnRi
       CALL CM_SC( uGrd, 'PPnRi', scal(1) )
       CALL CM_SC( uGrd, 'PPviscMin', PPviscMin )
       CALL CM_SC( uGrd, 'PPdiffMin', PPdiffMin )
       CALL CM_SC( uGrd, 'PPviscMax', PPviscMax )
       CALL CM_SC( uGrd, 'PPnu0', PPnu0 )
       CALL CM_SC( uGrd, 'PPalpha', PPalpha )
       CALL CM_SC( uGrd, 'RiLimit', RiLimit )
      ENDIF
#endif
#ifdef ALLOW_MY82
      IF ( useMY82 ) THEN
       CALL CM_SC( uGrd, 'alpha1', alpha1 )
       CALL CM_SC( uGrd, 'alpha2', alpha2 )
       CALL CM_SC( uGrd, 'beta1', beta1 )
       CALL CM_SC( uGrd, 'beta2', beta2 )
       CALL CM_SC( uGrd, 'beta3', beta3 )
       CALL CM_SC( uGrd, 'beta4', beta4 )
       CALL CM_SC( uGrd, 'RiMax', RiMax )
       CALL CM_SC( uGrd, 'MYhblScale', MYhblScale )
       CALL CM_SC( uGrd, 'MYviscMax', MYviscMax )
       CALL CM_SC( uGrd, 'MYdiffMax', MYdiffMax )
      ENDIF
#endif
#ifdef ALLOW_OPPS
      IF ( useOPPS ) THEN
       scal(1) = MAX_ABE_ITERATIONS
       CALL CM_SC( uGrd, 'MAX_ABE_ITERATIONS', scal(1) )
       scal(1) = OPPSdebugLevel
       CALL CM_SC( uGrd, 'OPPSdebugLevel', scal(1) )
       CALL CM_SC( uGrd, 'PlumeRadius', PlumeRadius )
       CALL CM_SC( uGrd, 'STABILITY_THRESHOLD', STABILITY_THRESHOLD )
       CALL CM_SC( uGrd, 'FRACTIONAL_AREA', FRACTIONAL_AREA )
       CALL CM_SC( uGrd, 'MAX_FRACTIONAL_AREA', MAX_FRACTIONAL_AREA )
       CALL CM_SC( uGrd, 'VERTICAL_VELOCITY', VERTICAL_VELOCITY )
       CALL CM_SC( uGrd, 'ENTRAINMENT_RATE', ENTRAINMENT_RATE )
       CALL CM_SC( uGrd, 'e2', e2 )
      ENDIF
#endif
      CLOSE( uGrd )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A,I3,A,A)')
     &  'REPLAY COLMIX harness: ', nSamp, ' samples x', Nr,
     &  ' levels x', nPassRun, ' passes; last eos ', equationOfState
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
       STOP 'REPLAY COLMIX: bad input record'
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

      SUBROUTINE CM_CP3( a, b )
      IMPLICIT NONE
#include "SIZE.h"
      Real*8 a(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      Real*8 b(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      INTEGER i, j, k
      DO k = 1, Nr
       DO j = 1-OLy, sNy+OLy
        DO i = 1-OLx, sNx+OLx
         b(i,j,k) = a(i,j,k)
        ENDDO
       ENDDO
      ENDDO
      RETURN
      END

      SUBROUTINE CM_CP2( a, b )
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
