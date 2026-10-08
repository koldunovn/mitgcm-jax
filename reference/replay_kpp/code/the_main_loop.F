#include "KPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax KPP replay harness
C     | (reference/replay_kpp; M3 sub-lane KPP, a copy of the
C     | Task 8 harness pattern reference/replay_col). Not an
C     | MITgcm routine: it replaces model/src/the_main_loop.F in
C     | a genmake2 build of vermix/code. THE_MODEL_MAIN has run
C     | INITIALISE_FIXED (real grid, INI_EOS, KPP_READPARMS,
C     | KPP_INIT_FIXED). This routine
C     |  1. calls KPP_INIT_VARIA and writes the KPP_PARAMS.h
C     |     values, zgrid, hwide, wmt, wst, nzmax and the KPP.h
C     |     fields it set (kpp_fixed.bin),
C     |  2. for every case of kpp_in.bin (synthetic columns):
C     |     sets DYNVARS.h theta, salt, uVel, vVel, IVDConvCount,
C     |     FFIELDS.h surfaceForcingU/V/T/S, adjustColdSST_diag,
C     |     Qsw, GRID.h maskC, maskW, maskS, fCori, the KPP_PARAMS
C     |     nzmax (land and shallow columns) and the KPP.h fields
C     |     (priors), then calls, for every tile, STATEKPP,
C     |     KPP_FORCING_SURF, KPPMIX, KPP_DOUBLEDIFF on synthetic
C     |     intermediate inputs (unit replays), then KPP_CALC
C     |     (the whole routine), KPP_DO_EXCH, KPP_CALC_DIFF_T/S
C     |     (kArg = 0 and kArg = k), KPP_CALC_VISC and
C     |     KPP_TRANSPORT_T/S for every level, and writes every
C     |     output, all points incl. halos (kpp_out.bin).
C     | Files: named big-endian stream records (CHARACTER*16
C     | name, INTEGER rank, INTEGER n, REAL*8 a(n)); per-case
C     | names are 'cNN:<name>'. Layout: replay_io.py.
C     | useDiagnostics is set .FALSE. (output only).
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "FFIELDS.h"
#include "KPP.h"
#include "KPP_PARAMS.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
      INTEGER nI3, nI3p, nI2, nO3, nO3p, nO2, nSc
      PARAMETER ( nI3 = 22, nI3p = 2, nI2 = 14 )
      PARAMETER ( nO3 = 19, nO3p = 2, nO2 = 7, nSc = 47 )
      INTEGER sz2, sz3, sz3p, szv
      PARAMETER ( sz2 = (sNx+2*OLx)*(sNy+2*OLy)*nSx*nSy )
      PARAMETER ( sz3 = sz2*Nr, sz3p = sz2*(Nr+1) )
      PARAMETER ( szv = sz2*(Nr+2)*mdiff )
      _RL i3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nI3)
      _RL i3p(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1,nSx,nSy,nI3p)
      _RL i2 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,nI2)
      _RL o3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nO3)
      _RL o3p(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1,nSx,nSy,nO3p)
      _RL o2 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy,nO2)
      _RL ovd(1-OLx:sNx+OLx,1-OLy:sNy+OLy,0:Nr+1,mdiff,nSx,nSy)
      COMMON / KPP_REPLAY_BUF / i3, i3p, i2, o3, o3p, o2, ovd
      _RL rho1 (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dbl  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL dbs  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL alp  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nrp1)
      _RL bet  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nrp1)
      _RL ust  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL bo   (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL bosol(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dvs  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL rhoS (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dbI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL rtI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL shI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL dvI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL ktI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL ksI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL ghI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL alI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nrp1)
      _RL beI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nrp1)
      _RL usI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL boI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL bsI  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL hbl  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vdd  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,0:Nrp1,mdiff)
      _RL kRT  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kRS  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kRU  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL kRV  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr)
      _RL df   (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL sc(nSc), hdr(4)
      INTEGER uIn, uOut, uFix, nCase, iCase, ikey
      INTEGER bi, bj, i, j, k, km1, md
      LOGICAL firstCall
      SAVE firstCall
      DATA firstCall / .TRUE. /

      IF ( .NOT.firstCall ) RETURN
      firstCall = .FALSE.
#if (defined ALLOW_SALT_PLUME || defined SHORTWAVE_HEATING)
      STOP 'REPLAY KPP: the calls below are written for builds without
     & ALLOW_SALT_PLUME and SHORTWAVE_HEATING (vermix/code)'
#endif
#ifndef KPP_ESTIMATE_UREF
      STOP 'REPLAY KPP: KPP_FORCING_SURF call needs KPP_ESTIMATE_UREF'
#endif
      uIn  = 91
      uOut = 92
      uFix = 93
      ikey = 1
      useDiagnostics = .FALSE.

C--   1. KPP_INIT_VARIA, then the fixed values (kpp_fixed.bin)
      CALL KPP_INIT_VARIA( myThid )
      sc( 1) = kpp_freq
      sc( 2) = kpp_dumpFreq
      sc( 3) = minKPPhbl
      sc( 4) = epsln
      sc( 5) = phepsi
      sc( 6) = epsilon
      sc( 7) = vonk
      sc( 8) = dB_dz
      sc( 9) = conc1
      sc(10) = conam
      sc(11) = concm
      sc(12) = conc2
      sc(13) = zetam
      sc(14) = conas
      sc(15) = concs
      sc(16) = conc3
      sc(17) = zetas
      sc(18) = Ricr
      sc(19) = cekman
      sc(20) = cmonob
      sc(21) = concv
      sc(22) = Vtc
      sc(23) = hbf
      sc(24) = deltaz
      sc(25) = deltau
      sc(26) = zmin
      sc(27) = zmax
      sc(28) = umin
      sc(29) = umax
      sc(30) = Riinfty
      sc(31) = BVSQcon
      sc(32) = difm0
      sc(33) = difs0
      sc(34) = dift0
      sc(35) = difmcon
      sc(36) = difscon
      sc(37) = diftcon
      sc(38) = Rrho0
      sc(39) = dsfmax
      sc(40) = cstar
      sc(41) = cg
      sc(42) = 0. _d 0
      IF ( KPPwriteState ) sc(42) = 1. _d 0
      sc(43) = 0. _d 0
      IF ( KPP_ghatUseTotalDiffus ) sc(43) = 1. _d 0
      sc(44) = 0. _d 0
      IF ( KPPuseDoubleDiff ) sc(44) = 1. _d 0
      sc(45) = 0. _d 0
      IF ( LimitHblStable ) sc(45) = 1. _d 0
      sc(46) = 0. _d 0
      IF ( KPPuseSWfrac3D ) sc(46) = 1. _d 0
      sc(47) = num_v_smooth_Ri
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          o2(i,j,bi,bj,1) = nzmax(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      OPEN( uFix, FILE='kpp_fixed.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL KPPR_WR( uFix, 0, 'scalars', 1, nSc, sc )
      CALL KPPR_WR( uFix, 0, 'zgrid', 1, Nr+2, zgrid )
      CALL KPPR_WR( uFix, 0, 'hwide', 1, Nr+2, hwide )
      CALL KPPR_WR( uFix, 0, 'wmt', 1, (nni+2)*(nnj+2), wmt )
      CALL KPPR_WR( uFix, 0, 'wst', 1, (nni+2)*(nnj+2), wst )
      CALL KPPR_WR( uFix, 0, 'nzmax', 2, sz2, o2(1-OLx,1-OLy,1,1,1) )
      CALL KPPR_WR( uFix, 0, 'KPPviscAz', 3, sz3, KPPviscAz )
      CALL KPPR_WR( uFix, 0, 'KPPdiffKzS', 3, sz3, KPPdiffKzS )
      CALL KPPR_WR( uFix, 0, 'KPPdiffKzT', 3, sz3, KPPdiffKzT )
      CALL KPPR_WR( uFix, 0, 'KPPghat', 3, sz3, KPPghat )
      CALL KPPR_WR( uFix, 0, 'KPPhbl', 2, sz2, KPPhbl )
      CALL KPPR_WR( uFix, 0, 'viscArNr', 1, Nr, viscArNr )
      CALL KPPR_WR( uFix, 0, 'diffKrNrT', 1, Nr, diffKrNrT )
      CALL KPPR_WR( uFix, 0, 'diffKrNrS', 1, Nr, diffKrNrS )
      CLOSE( uFix )

C--   2. cases
      OPEN( uIn, FILE='kpp_in.bin', STATUS='OLD', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      CALL KPPR_RD( uIn, 0, 'header', 4, hdr )
      IF ( hdr(1).NE.sNx .OR. hdr(2).NE.sNy .OR. hdr(3).NE.Nr )
     &   STOP 'REPLAY KPP: kpp_in.bin does not match SIZE.h'
      nCase = NINT(hdr(4))
      OPEN( uOut, FILE='kpp_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )

      DO iCase = 1, nCase
C     2a. inputs (order: replay_io.py IN3, IN3P, IN2)
       CALL KPPR_RD( uIn, iCase, 'theta',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1, 1))
       CALL KPPR_RD( uIn, iCase, 'salt',     sz3,
     &               i3(1-OLx,1-OLy,1,1,1, 2))
       CALL KPPR_RD( uIn, iCase, 'uVel',     sz3,
     &               i3(1-OLx,1-OLy,1,1,1, 3))
       CALL KPPR_RD( uIn, iCase, 'vVel',     sz3,
     &               i3(1-OLx,1-OLy,1,1,1, 4))
       CALL KPPR_RD( uIn, iCase, 'IVDConv',  sz3,
     &               i3(1-OLx,1-OLy,1,1,1, 5))
       CALL KPPR_RD( uIn, iCase, 'maskC',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1, 6))
       CALL KPPR_RD( uIn, iCase, 'maskW',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1, 7))
       CALL KPPR_RD( uIn, iCase, 'maskS',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1, 8))
       CALL KPPR_RD( uIn, iCase, 'pViscAz',  sz3,
     &               i3(1-OLx,1-OLy,1,1,1, 9))
       CALL KPPR_RD( uIn, iCase, 'pDiffKzS', sz3,
     &               i3(1-OLx,1-OLy,1,1,1,10))
       CALL KPPR_RD( uIn, iCase, 'pDiffKzT', sz3,
     &               i3(1-OLx,1-OLy,1,1,1,11))
       CALL KPPR_RD( uIn, iCase, 'pGhat',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1,12))
       CALL KPPR_RD( uIn, iCase, 'dblocI',   sz3,
     &               i3(1-OLx,1-OLy,1,1,1,13))
       CALL KPPR_RD( uIn, iCase, 'RitopI',   sz3,
     &               i3(1-OLx,1-OLy,1,1,1,14))
       CALL KPPR_RD( uIn, iCase, 'shsqI',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1,15))
       CALL KPPR_RD( uIn, iCase, 'dVsqI',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1,16))
       CALL KPPR_RD( uIn, iCase, 'kapTI',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1,17))
       CALL KPPR_RD( uIn, iCase, 'kapSI',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1,18))
       CALL KPPR_RD( uIn, iCase, 'ghatI',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1,19))
       CALL KPPR_RD( uIn, iCase, 'kapRU',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1,20))
       CALL KPPR_RD( uIn, iCase, 'kapRV',    sz3,
     &               i3(1-OLx,1-OLy,1,1,1,21))
       CALL KPPR_RD( uIn, iCase, 'dfPrior',  sz3,
     &               i3(1-OLx,1-OLy,1,1,1,22))
       CALL KPPR_RD( uIn, iCase, 'ttalphaI', sz3p,
     &               i3p(1-OLx,1-OLy,1,1,1,1) )
       CALL KPPR_RD( uIn, iCase, 'ssbetaI',  sz3p,
     &               i3p(1-OLx,1-OLy,1,1,1,2) )
       CALL KPPR_RD( uIn, iCase, 'sfU',      sz2,
     &               i2(1-OLx,1-OLy,1,1, 1))
       CALL KPPR_RD( uIn, iCase, 'sfV',      sz2,
     &               i2(1-OLx,1-OLy,1,1, 2))
       CALL KPPR_RD( uIn, iCase, 'sfT',      sz2,
     &               i2(1-OLx,1-OLy,1,1, 3))
       CALL KPPR_RD( uIn, iCase, 'sfS',      sz2,
     &               i2(1-OLx,1-OLy,1,1, 4))
       CALL KPPR_RD( uIn, iCase, 'adjCold',  sz2,
     &               i2(1-OLx,1-OLy,1,1, 5))
       CALL KPPR_RD( uIn, iCase, 'Qsw',      sz2,
     &               i2(1-OLx,1-OLy,1,1, 6))
       CALL KPPR_RD( uIn, iCase, 'fCori',    sz2,
     &               i2(1-OLx,1-OLy,1,1, 7))
       CALL KPPR_RD( uIn, iCase, 'nzmax',    sz2,
     &               i2(1-OLx,1-OLy,1,1, 8))
       CALL KPPR_RD( uIn, iCase, 'pHbl',     sz2,
     &               i2(1-OLx,1-OLy,1,1, 9))
       CALL KPPR_RD( uIn, iCase, 'pFrac',    sz2,
     &               i2(1-OLx,1-OLy,1,1,10))
       CALL KPPR_RD( uIn, iCase, 'sdensI',   sz2,
     &               i2(1-OLx,1-OLy,1,1,11))
       CALL KPPR_RD( uIn, iCase, 'ustarI',   sz2,
     &               i2(1-OLx,1-OLy,1,1,12))
       CALL KPPR_RD( uIn, iCase, 'boI',      sz2,
     &               i2(1-OLx,1-OLy,1,1,13))
       CALL KPPR_RD( uIn, iCase, 'bosolI',   sz2,
     &               i2(1-OLx,1-OLy,1,1,14))

C     2b. model state, grid overrides and KPP.h priors
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            theta(i,j,k,bi,bj)        = i3(i,j,k,bi,bj, 1)
            salt (i,j,k,bi,bj)        = i3(i,j,k,bi,bj, 2)
            uVel (i,j,k,bi,bj)        = i3(i,j,k,bi,bj, 3)
            vVel (i,j,k,bi,bj)        = i3(i,j,k,bi,bj, 4)
            IVDConvCount(i,j,k,bi,bj) = i3(i,j,k,bi,bj, 5)
            maskC(i,j,k,bi,bj)        = i3(i,j,k,bi,bj, 6)
            maskW(i,j,k,bi,bj)        = i3(i,j,k,bi,bj, 7)
            maskS(i,j,k,bi,bj)        = i3(i,j,k,bi,bj, 8)
            KPPviscAz (i,j,k,bi,bj)   = i3(i,j,k,bi,bj, 9)
            KPPdiffKzS(i,j,k,bi,bj)   = i3(i,j,k,bi,bj,10)
            KPPdiffKzT(i,j,k,bi,bj)   = i3(i,j,k,bi,bj,11)
            KPPghat   (i,j,k,bi,bj)   = i3(i,j,k,bi,bj,12)
           ENDDO
          ENDDO
         ENDDO
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           surfaceForcingU(i,j,bi,bj)    = i2(i,j,bi,bj, 1)
           surfaceForcingV(i,j,bi,bj)    = i2(i,j,bi,bj, 2)
           surfaceForcingT(i,j,bi,bj)    = i2(i,j,bi,bj, 3)
           surfaceForcingS(i,j,bi,bj)    = i2(i,j,bi,bj, 4)
           adjustColdSST_diag(i,j,bi,bj) = i2(i,j,bi,bj, 5)
           Qsw(i,j,bi,bj)                = i2(i,j,bi,bj, 6)
           fCori(i,j,bi,bj)              = i2(i,j,bi,bj, 7)
           nzmax(i,j,bi,bj)              = NINT(i2(i,j,bi,bj, 8))
           KPPhbl (i,j,bi,bj)            = i2(i,j,bi,bj, 9)
           KPPfrac(i,j,bi,bj)            = i2(i,j,bi,bj,10)
          ENDDO
         ENDDO
        ENDDO
       ENDDO

C     2c. unit replays per tile: STATEKPP, KPP_FORCING_SURF, KPPMIX,
C     KPP_DOUBLEDIFF (KPP_CALC's call arguments, synthetic inputs)
       DO bj = 1, nSy
        DO bi = 1, nSx
         CALL STATEKPP( rho1, dbl, dbs, alp, bet, ikey, bi, bj, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           o2(i,j,bi,bj,1) = rho1(i,j)
           rhoS(i,j) = i2(i,j,bi,bj,11)
           usI(i,j)  = i2(i,j,bi,bj,12)
           boI(i,j)  = i2(i,j,bi,bj,13)
           bsI(i,j)  = i2(i,j,bi,bj,14)
          ENDDO
         ENDDO
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj,1) = dbl(i,j,k)
            o3(i,j,k,bi,bj,2) = dbs(i,j,k)
            dbI(i,j,k) = i3(i,j,k,bi,bj,13)
            rtI(i,j,k) = i3(i,j,k,bi,bj,14)
            shI(i,j,k) = i3(i,j,k,bi,bj,15)
            dvI(i,j,k) = i3(i,j,k,bi,bj,16)
            ktI(i,j,k) = i3(i,j,k,bi,bj,17)
            ksI(i,j,k) = i3(i,j,k,bi,bj,18)
            ghI(i,j,k) = i3(i,j,k,bi,bj,19)
           ENDDO
          ENDDO
         ENDDO
         DO k = 1, Nr+1
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3p(i,j,k,bi,bj,1) = alp(i,j,k)
            o3p(i,j,k,bi,bj,2) = bet(i,j,k)
            alI(i,j,k) = i3p(i,j,k,bi,bj,1)
            beI(i,j,k) = i3p(i,j,k,bi,bj,2)
           ENDDO
          ENDDO
         ENDDO
C     KPP_FORCING_SURF with KPP_CALC's bounds (kpp_calc.F:153)
         CALL KPP_FORCING_SURF(
     I     rhoS, surfaceForcingU, surfaceForcingV,
     I     surfaceForcingT, surfaceForcingS, adjustColdSST_diag,
     I     Qsw,
     I     dbI,
     I     alI, beI,
     O     ust, bo, bosol,
     O     dvs,
     I     ikey, 2-OLx, sNx+OLx-1, 2-OLy, sNy+OLy-1,
     I     bi, bj, myTime, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           o2(i,j,bi,bj,2) = ust(i,j)
           o2(i,j,bi,bj,3) = bo(i,j)
           o2(i,j,bi,bj,4) = bosol(i,j)
          ENDDO
         ENDDO
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj,3) = dvs(i,j,k)
           ENDDO
          ENDDO
         ENDDO
C     KPPMIX (prior of diffus: -1., every point is written)
         DO md = 1, mdiff
          DO k = 0, Nrp1
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             vdd(i,j,k,md) = -1. _d 0
            ENDDO
           ENDDO
          ENDDO
         ENDDO
         CALL KPPMIX (
     I       nzmax(1-OLx,1-OLy,bi,bj), shI, dvI, usI,
     I       maskC(1-OLx,1-OLy,1,bi,bj),
     I       boI, bsI,
     I       dbI, rtI, fCori(1-OLx,1-OLy,bi,bj),
     I       ksI, ktI,
     I       ikey,
     O       vdd,
     U       ghI,
     O       hbl,
     I       bi, bj, myTime, myIter, myThid )
         DO md = 1, mdiff
          DO k = 0, Nrp1
           DO j = 1-OLy, sNy+OLy
            DO i = 1-OLx, sNx+OLx
             ovd(i,j,k,md,bi,bj) = vdd(i,j,k,md)
            ENDDO
           ENDDO
          ENDDO
         ENDDO
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           o2(i,j,bi,bj,5) = hbl(i,j)
          ENDDO
         ENDDO
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj,4) = ghI(i,j,k)
            kRT(i,j,k) = i3(i,j,k,bi,bj,17)
            kRS(i,j,k) = i3(i,j,k,bi,bj,18)
           ENDDO
          ENDDO
         ENDDO
C     KPP_DOUBLEDIFF with KPP_CALC's bounds (kpp_calc.F:537)
         CALL KPP_DOUBLEDIFF(
     I      alI, beI,
     U      kRT,
     U      kRS,
     I      ikey, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy, bi, bj, myThid )
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj,5) = kRT(i,j,k)
            o3(i,j,k,bi,bj,6) = kRS(i,j,k)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO

C     2d. KPP_CALC (as DO_OCEANIC_PHYS calls it), then KPP_DO_EXCH
       DO bj = 1, nSy
        DO bi = 1, nSx
         CALL KPP_CALC( bi, bj, myTime, myIter, myThid )
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj, 7) = KPPviscAz (i,j,k,bi,bj)
            o3(i,j,k,bi,bj, 8) = KPPdiffKzS(i,j,k,bi,bj)
            o3(i,j,k,bi,bj, 9) = KPPdiffKzT(i,j,k,bi,bj)
            o3(i,j,k,bi,bj,10) = KPPghat   (i,j,k,bi,bj)
           ENDDO
          ENDDO
         ENDDO
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           o2(i,j,bi,bj,6) = KPPhbl (i,j,bi,bj)
           o2(i,j,bi,bj,7) = KPPfrac(i,j,bi,bj)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       CALL KPP_DO_EXCH( myThid )

C     2e. the routines that read KPP.h (after the exchange):
C     KPP_CALC_DIFF_T/S (CALC_3D_DIFFUSIVITY's call, kArg = 0, and
C     kArg = k), KPP_CALC_VISC and KPP_TRANSPORT_T/S for every level
C     with the bounds of DYNAMICS / TEMP_INTEGRATE (0, sNx+1, 0, sNy+1)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj,11) = KPPviscAz(i,j,k,bi,bj)
            kRT(i,j,k) = i3(i,j,k,bi,bj,17)
            kRS(i,j,k) = i3(i,j,k,bi,bj,18)
           ENDDO
          ENDDO
         ENDDO
         CALL KPP_CALC_DIFF_T( bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     &                         0, Nr, kRT, myThid )
         CALL KPP_CALC_DIFF_S( bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy,
     &                         0, Nr, kRS, myThid )
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj,12) = kRT(i,j,k)
            o3(i,j,k,bi,bj,13) = kRS(i,j,k)
            kRT(i,j,k) = i3(i,j,k,bi,bj,17)
            kRS(i,j,k) = i3(i,j,k,bi,bj,18)
            kRU(i,j,k) = i3(i,j,k,bi,bj,20)
            kRV(i,j,k) = i3(i,j,k,bi,bj,21)
           ENDDO
          ENDDO
         ENDDO
         DO k = 1, Nr
          CALL KPP_CALC_DIFF_T( bi, bj, 0, sNx+1, 0, sNy+1,
     &                          k, Nr, kRT, myThid )
          CALL KPP_CALC_DIFF_S( bi, bj, 0, sNx+1, 0, sNy+1,
     &                          k, Nr, kRS, myThid )
          CALL KPP_CALC_VISC( bi, bj, 0, sNx+1, 0, sNy+1, k,
     &                        kRU, kRV, myThid )
         ENDDO
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj,14) = kRT(i,j,k)
            o3(i,j,k,bi,bj,15) = kRS(i,j,k)
            o3(i,j,k,bi,bj,16) = kRU(i,j,k)
            o3(i,j,k,bi,bj,17) = kRV(i,j,k)
           ENDDO
          ENDDO
         ENDDO
         DO k = 1, Nr
          km1 = MAX(1,k-1)
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            df(i,j) = i3(i,j,k,bi,bj,22)
           ENDDO
          ENDDO
          CALL KPP_TRANSPORT_T( 0, sNx+1, 0, sNy+1, bi, bj, k, km1,
     &                          df, myTime, myIter, myThid )
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj,18) = df(i,j)
            df(i,j) = i3(i,j,k,bi,bj,22)
           ENDDO
          ENDDO
          CALL KPP_TRANSPORT_S( 0, sNx+1, 0, sNy+1, bi, bj, k, km1,
     &                          df, myTime, myIter, myThid )
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            o3(i,j,k,bi,bj,19) = df(i,j)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO

C     2f. outputs of the case (order: replay_io.py OUT3, OUT3P, OUT2)
       CALL KPPR_WR( uOut, iCase, 'sk_dbloc',  3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1, 1) )
       CALL KPPR_WR( uOut, iCase, 'sk_dbsfc',  3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1, 2) )
       CALL KPPR_WR( uOut, iCase, 'fs_dVsq',   3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1, 3) )
       CALL KPPR_WR( uOut, iCase, 'mx_ghat',   3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1, 4) )
       CALL KPPR_WR( uOut, iCase, 'dd_kapT',   3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1, 5) )
       CALL KPPR_WR( uOut, iCase, 'dd_kapS',   3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1, 6) )
       CALL KPPR_WR( uOut, iCase, 'KPPviscAz', 3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1, 7) )
       CALL KPPR_WR( uOut, iCase, 'KPPdiffKzS',3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1, 8) )
       CALL KPPR_WR( uOut, iCase, 'KPPdiffKzT',3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1, 9) )
       CALL KPPR_WR( uOut, iCase, 'KPPghat',   3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,10) )
       CALL KPPR_WR( uOut, iCase, 'ex_viscAz', 3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,11) )
       CALL KPPR_WR( uOut, iCase, 'cd_T0',     3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,12) )
       CALL KPPR_WR( uOut, iCase, 'cd_S0',     3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,13) )
       CALL KPPR_WR( uOut, iCase, 'cd_Tk',     3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,14) )
       CALL KPPR_WR( uOut, iCase, 'cd_Sk',     3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,15) )
       CALL KPPR_WR( uOut, iCase, 'cv_RU',     3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,16) )
       CALL KPPR_WR( uOut, iCase, 'cv_RV',     3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,17) )
       CALL KPPR_WR( uOut, iCase, 'tr_T',      3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,18) )
       CALL KPPR_WR( uOut, iCase, 'tr_S',      3, sz3,
     &               o3(1-OLx,1-OLy,1,1,1,19) )
       CALL KPPR_WR( uOut, iCase, 'sk_alpha',  3, sz3p,
     &               o3p(1-OLx,1-OLy,1,1,1,1) )
       CALL KPPR_WR( uOut, iCase, 'sk_beta',   3, sz3p,
     &               o3p(1-OLx,1-OLy,1,1,1,2) )
       CALL KPPR_WR( uOut, iCase, 'sk_rho1',   2, sz2,
     &               o2(1-OLx,1-OLy,1,1,1) )
       CALL KPPR_WR( uOut, iCase, 'fs_ustar',  2, sz2,
     &               o2(1-OLx,1-OLy,1,1,2) )
       CALL KPPR_WR( uOut, iCase, 'fs_bo',     2, sz2,
     &               o2(1-OLx,1-OLy,1,1,3) )
       CALL KPPR_WR( uOut, iCase, 'fs_bosol',  2, sz2,
     &               o2(1-OLx,1-OLy,1,1,4) )
       CALL KPPR_WR( uOut, iCase, 'mx_hbl',    2, sz2,
     &               o2(1-OLx,1-OLy,1,1,5) )
       CALL KPPR_WR( uOut, iCase, 'KPPhbl',    2, sz2,
     &               o2(1-OLx,1-OLy,1,1,6) )
       CALL KPPR_WR( uOut, iCase, 'KPPfrac',   2, sz2,
     &               o2(1-OLx,1-OLy,1,1,7) )
       CALL KPPR_WR( uOut, iCase, 'mx_vddiff', 4, szv, ovd )
      ENDDO
      CLOSE( uIn )
      CLOSE( uOut )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A)')
     &  'REPLAY KPP harness: ', nCase, ' cases x', nSx*nSy,
     &  ' tiles replayed'
      RETURN
      END

C     named records (big-endian stream): CHARACTER*16 name,
C     INTEGER rank, INTEGER n, REAL*8 a(n); name 'cNN:<nm>' for
C     case NN > 0, '<nm>' for case 0
      SUBROUTINE KPPR_NAME( iCase, nm, full )
      IMPLICIT NONE
      INTEGER iCase
      CHARACTER*(*) nm
      CHARACTER*16 full
      IF ( iCase.GT.0 ) THEN
       WRITE(full,'(A1,I2.2,A1,A)') 'c', iCase, ':', nm
      ELSE
       full = nm
      ENDIF
      RETURN
      END

      SUBROUTINE KPPR_RD( u, iCase, nm, n, a )
      IMPLICIT NONE
      INTEGER u, iCase, n
      CHARACTER*(*) nm
      Real*8 a(n)
      CHARACTER*16 got, want
      INTEGER rk, m
      CALL KPPR_NAME( iCase, nm, want )
      READ( u ) got, rk, m
      IF ( got.NE.want .OR. m.NE.n ) THEN
       WRITE(0,*) 'KPPR_RD: want ', want, n, ' got ', got, m
       STOP 'REPLAY KPP: bad input record'
      ENDIF
      READ( u ) a
      RETURN
      END

      SUBROUTINE KPPR_WR( u, iCase, nm, rk, n, a )
      IMPLICIT NONE
      INTEGER u, iCase, rk, n
      CHARACTER*(*) nm
      Real*8 a(n)
      CHARACTER*16 full
      CALL KPPR_NAME( iCase, nm, full )
      WRITE( u ) full, rk, n
      WRITE( u ) a
      RETURN
      END
