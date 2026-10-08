#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax VECINV replay
C     | harness (reference/replay_vecinv; M2 sub-lane VECINV,
C     | plan Task 23; a copy of reference/replay_mom). Not an
C     | MITgcm routine: it replaces model/src/the_main_loop.F in
C     | a genmake2 build of a verification experiment's code dir.
C     | THE_MODEL_MAIN has already called INITIALISE_FIXED, so
C     | GRID.h holds the experiment's real grid (land, partial
C     | cells, cube corners, halos filled by the exchanges) and
C     | MOM_VISC.h the viscosity lengths of MOM_INIT_FIXED.
C     |  1. reads synthetic inputs written by Python
C     |     (replay_vecinv/replay_io.py, big-endian stream),
C     |  2. overrides grid factors that are trivial in the
C     |     experiments (deep/rho factors, cosFac, rVel2wUnit,
C     |     fCoriCos, angleCos/SinC, deepFacAdv) with the input
C     |     values, so that a missing or swapped factor shows,
C     |  3. calls MOM_VECINV per tile and level with the run's
C     |     own PARAMS.h (m0) and with the vorticity selectors
C     |     flipped (m1), then the leaf routines of
C     |     pkg/mom_vecinv and pkg/mom_common with the selectors
C     |     set per case; every output is first set to a
C     |     Python-given prior field (points a routine does not
C     |     write keep it),
C     |  4. writes every output, all points incl. halos
C     |     (replay_out.bin), and the grid and parameter values
C     |     the routines read (replay_grid.bin).
C     | No time stepping. useDiagnostics and useGrdchk are set
C     | .FALSE. The case
C     | list and the file layout are mirrored by
C     | replay_vecinv/replay_io.py (OUT3).
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
#include "MOM_VISC.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
      INTEGER nIn3, nOut3, nSPar, nVPar, nVC
      PARAMETER ( nIn3 = 22, nOut3 = 91, nSPar = 4 )
      PARAMETER ( nVPar = 20, nVC = 4 )
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261003, VERSN = 2 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / VIRP_BUF / in3, out3
      _RL uF  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vF  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL hDi (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vo3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dSt (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL zSt (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL ten (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL stn (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL KEi (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vAhZ(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vAhD(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vA4Z(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL vA4D(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL om3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL kapU(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1)
      _RL kapV(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1)
      _RL wA  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL wB  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL wC  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL wD  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL wH  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL fUm (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL fVm (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL fUp (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL fVp (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL guD (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL gvD (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS hZ  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RS rhZ (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dFacA(Nr)
      _RL spar(nSPar), vpar(nVPar,nVC)
      INTEGER hdr(10), ohdr(4)
      INTEGER uIn, uOut, uGrd, ios
      INTEGER bi, bj, i, j, k, kk, iO, iC, iM
      LOGICAL vFul(nVC), vHar(nVC), vBih(nVC)
      LOGICAL dHar(0:6), dBih(0:6), dVar(0:6)
      INTEGER cSch(0:5), sKE(0:3)
      LOGICAL cJam(0:5), sUpw(0:3)
      LOGICAL sAbs, sCD, sUpS, sJam
      INTEGER sVrt
C     MOM_CALC_VISC cases v1..v4 (useFullLeith, useHarmonicVisc,
C     useBiharmonicVisc); REAL parameters vpar from the input file
      DATA vFul / .FALSE., .TRUE., .FALSE., .FALSE. /
      DATA vHar / .FALSE., .TRUE., .TRUE., .TRUE. /
      DATA vBih / .TRUE., .TRUE., .TRUE., .FALSE. /
C     MOM_VI_HDISSIP cases h0..h6 (harmonic, biharmonic, variable)
      DATA dHar / .TRUE., .TRUE., .TRUE., .TRUE.,
     &            .FALSE., .FALSE., .FALSE. /
      DATA dBih / .TRUE., .TRUE., .FALSE., .FALSE.,
     &            .TRUE., .TRUE., .FALSE. /
      DATA dVar / .TRUE., .FALSE., .TRUE., .FALSE.,
     &            .TRUE., .FALSE., .FALSE. /
C     MOM_VI_U/V_CORIOLIS cases (selectVortScheme, useJamartMomAdv)
      DATA cSch / 0, 1, 2, 3, 4, 1 /
      DATA cJam / .FALSE., .FALSE., .FALSE., .FALSE., .FALSE.,
     &            .TRUE. /
C     MOM_VI_U/V_VERTSHEAR cases (selectKEscheme, upwindShear)
      DATA sKE  / 0, 0, 1, 3 /
      DATA sUpw / .FALSE., .TRUE., .FALSE., .TRUE. /

      uIn  = 91
      uOut = 92
      uGrd = 93

C--   1. inputs (layout: replay_vecinv/replay_io.py)
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
      READ( uIn ) spar
      READ( uIn ) vpar
      READ( uIn ) recip_deepFacC, recip_deepFac2C, deepFacC, deepFac2C,
     &            recip_rhoFacC, dFacA
      READ( uIn ) deepFac2F, rhoFacF, rVel2wUnit
      READ( uIn ) cosFacU, cosFacV
      READ( uIn ) fCoriCos, angleCosC, angleSinC
      READ( uIn ) in3
      CLOSE( uIn )

C--   2. overrides (see header)
      useDiagnostics  = .FALSE.
      useGrdchk       = .FALSE.
      DO k = 1, Nr
       deepFacAdv(k) = dFacA(k)
      ENDDO
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uVel(i,j,k,bi,bj) = in3(i,j,k,bi,bj,1)
           vVel(i,j,k,bi,bj) = in3(i,j,k,bi,bj,2)
           wVel(i,j,k,bi,bj) = in3(i,j,k,bi,bj,3)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO

C--   2b. the run's parameters as MOM_VECINV (m0) reads them; the
C     W2 topology of each tile (cube corners of MOM_CALC_RELVORT3)
      CALL VIRP_WRITE_GRID( uGrd, dFacA, MAGIC, VERSN )
      CALL VIRP_WRITE_W2( uGrd, MAGIC, VERSN )

C--   3a. MOM_VECINV, m0 = the run's own PARAMS.h, m1 = vorticity
C     selectors flipped (useAbsVorticity=T, useCDscheme=F,
C     upwindShear=T, selectVortScheme=2, useJamartMomAdv=T)
      sAbs = useAbsVorticity
      sCD  = useCDscheme
      sUpS = upwindShear
      sVrt = selectVortScheme
      sJam = useJamartMomAdv
      DO iM = 0, 1
       IF ( iM.EQ.1 ) THEN
        useAbsVorticity  = .TRUE.
        useCDscheme      = .FALSE.
        upwindShear      = .TRUE.
        selectVortScheme = 2
        useJamartMomAdv  = .TRUE.
       ENDIF
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO kk = 1, Nr+1
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            IF ( kk.LE.Nr ) THEN
             kapU(i,j,kk) = in3(i,j,kk,bi,bj,16)
             kapV(i,j,kk) = in3(i,j,kk,bi,bj,17)
            ELSE
             kapU(i,j,kk) = in3(i,j,1,bi,bj,18)
             kapV(i,j,kk) = in3(i,j,1,bi,bj,19)
            ENDIF
           ENDDO
          ENDDO
         ENDDO
         DO k = 1, Nr
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            fUm(i,j) = in3(i,j,k,bi,bj,20)
            fVm(i,j) = in3(i,j,k,bi,bj,21)
            fUp(i,j) = in3(i,j,k,bi,bj,22)
            fVp(i,j) = in3(i,j,k,bi,bj,22)
            guD(i,j) = in3(i,j,k,bi,bj,22)
            gvD(i,j) = in3(i,j,k,bi,bj,22)
            gU(i,j,k,bi,bj) = in3(i,j,k,bi,bj,22)
            gV(i,j,k,bi,bj) = in3(i,j,k,bi,bj,22)
           ENDDO
          ENDDO
          CALL MOM_VECINV( bi, bj, k, 0, sNx+1, 0, sNy+1,
     &                     kapU, kapV, fUm, fVm, fUp, fVp, guD, gvD,
     &                     myTime, myIter, myThid )
          iO = 79 + 6*iM
          CALL VIRP_PUT( fUp, bi, bj, k, iO+1 )
          CALL VIRP_PUT( fVp, bi, bj, k, iO+2 )
          CALL VIRP_PUT( guD, bi, bj, k, iO+3 )
          CALL VIRP_PUT( gvD, bi, bj, k, iO+4 )
          CALL VIRP_PUT( gU(1-OLx,1-OLy,k,bi,bj), bi, bj, k, iO+5 )
          CALL VIRP_PUT( gV(1-OLx,1-OLy,k,bi,bj), bi, bj, k, iO+6 )
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      useAbsVorticity  = sAbs
      useCDscheme      = sCD
      upwindShear      = sUpS
      selectVortScheme = sVrt
      useJamartMomAdv  = sJam

C--   3b. leaf routines
      nonHydrostatic = .FALSE.
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO k = 1, Nr
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           uF  (i,j) = in3(i,j,k,bi,bj, 1)
           vF  (i,j) = in3(i,j,k,bi,bj, 2)
           vo3 (i,j) = in3(i,j,k,bi,bj, 5)
           dSt (i,j) = in3(i,j,k,bi,bj, 6)
           zSt (i,j) = in3(i,j,k,bi,bj, 7)
           ten (i,j) = in3(i,j,k,bi,bj, 8)
           stn (i,j) = in3(i,j,k,bi,bj, 9)
           KEi (i,j) = in3(i,j,k,bi,bj,10)
           vAhZ(i,j) = in3(i,j,k,bi,bj,11)
           vAhD(i,j) = in3(i,j,k,bi,bj,12)
           vA4Z(i,j) = in3(i,j,k,bi,bj,13)
           vA4D(i,j) = in3(i,j,k,bi,bj,14)
           om3 (i,j) = in3(i,j,k,bi,bj,15)
          ENDDO
         ENDDO

C     out 1-2: MOM_CALC_HFACZ -> hFacZ, r_hFacZ (the hFacZ of the
C     other routines)
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           hZ (i,j) = in3(i,j,k,bi,bj,22)
           rhZ(i,j) = in3(i,j,k,bi,bj,22)
          ENDDO
         ENDDO
         CALL MOM_CALC_HFACZ( bi, bj, k, hZ, rhZ, myThid )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           out3(i,j,k,bi,bj,1) = hZ(i,j)
           out3(i,j,k,bi,bj,2) = rhZ(i,j)
          ENDDO
         ENDDO
         iO = 2

C     out 3-4: MOM_CALC_HDIV, hDivScheme = 1, 2
         DO iC = 1, 2
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL MOM_CALC_HDIV( bi, bj, k, iC, uF, vF, wA, myThid )
          iO = iO + 1
          CALL VIRP_PUT( wA, bi, bj, k, iO )
         ENDDO

C     out 5: MOM_CALC_RELVORT3
         CALL VIRP_PRIOR( wA, bi, bj, k )
         CALL MOM_CALC_RELVORT3( bi, bj, k, uF, vF, hZ, wA, myThid )
         iO = iO + 1
         CALL VIRP_PUT( wA, bi, bj, k, iO )

C     out 6-8: MOM_CALC_ABSVORT3, (momAdvection, useCoriolis) =
C     (T,T), (F,T), (T,F)
         DO iC = 0, 2
          momAdvection = iC.NE.1
          useCoriolis  = iC.NE.2
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL MOM_CALC_ABSVORT3( bi, bj, k, vo3, wA, myThid )
          iO = iO + 1
          CALL VIRP_PUT( wA, bi, bj, k, iO )
         ENDDO
         momAdvection = .TRUE.
         useCoriolis  = .TRUE.

C     out 9: MOM_CALC_TENSION ; out 10: MOM_CALC_STRAIN
         CALL VIRP_PRIOR( wA, bi, bj, k )
         CALL MOM_CALC_TENSION( bi, bj, k, uF, vF, wA, myThid )
         iO = iO + 1
         CALL VIRP_PUT( wA, bi, bj, k, iO )
         CALL VIRP_PRIOR( wA, bi, bj, k )
         CALL MOM_CALC_STRAIN( bi, bj, k, uF, vF, hZ, wA, myThid )
         iO = iO + 1
         CALL VIRP_PUT( wA, bi, bj, k, iO )

C     out 11-30: MOM_CALC_VISC cases v1..v4 -> viscAh_Z, viscAh_D,
C     viscA4_Z, viscA4_D, hDiv after the call (FILL_CS_CORNER_TR_RL
C     writes the corner halos of hDiv on the cube)
         DO iC = 1, nVC
          CALL VIRP_SETVISC( vpar(1,iC), vFul(iC), vHar(iC), vBih(iC) )
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL VIRP_PRIOR( wB, bi, bj, k )
          CALL VIRP_PRIOR( wC, bi, bj, k )
          CALL VIRP_PRIOR( wD, bi, bj, k )
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            wH(i,j) = in3(i,j,k,bi,bj,4)
           ENDDO
          ENDDO
          CALL MOM_CALC_VISC( bi, bj, k, wA, wB, wC, wD,
     &         wH, vo3, ten, stn, zSt, KEi, hZ, myThid )
          CALL VIRP_PUT( wA, bi, bj, k, iO+1 )
          CALL VIRP_PUT( wB, bi, bj, k, iO+2 )
          CALL VIRP_PUT( wC, bi, bj, k, iO+3 )
          CALL VIRP_PUT( wD, bi, bj, k, iO+4 )
          CALL VIRP_PUT( wH, bi, bj, k, iO+5 )
          iO = iO + 5
         ENDDO
         viscAhD = spar(1)
         viscAhZ = spar(2)
         viscA4D = spar(3)
         viscA4Z = spar(4)

C     out 31-33: MOM_VI_DEL2UV -> del2u, del2v, hDiv after the call
         CALL VIRP_PRIOR( wA, bi, bj, k )
         CALL VIRP_PRIOR( wB, bi, bj, k )
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           wH(i,j) = in3(i,j,k,bi,bj,4)
          ENDDO
         ENDDO
         CALL MOM_VI_DEL2UV( bi, bj, k, wH, vo3, hZ, wA, wB, myThid )
         CALL VIRP_PUT( wA, bi, bj, k, iO+1 )
         CALL VIRP_PUT( wB, bi, bj, k, iO+2 )
         CALL VIRP_PUT( wH, bi, bj, k, iO+3 )
         iO = iO + 3

C     out 34-47: MOM_VI_HDISSIP cases h0..h6 -> uDissip, vDissip
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           hDi(i,j) = in3(i,j,k,bi,bj,4)
          ENDDO
         ENDDO
         DO iC = 0, 6
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL VIRP_PRIOR( wB, bi, bj, k )
          CALL MOM_VI_HDISSIP( bi, bj, k,
     &         hDi, vo3, dSt, zSt, hZ,
     &         vAhZ, vAhD, vA4Z, vA4D,
     &         dHar(iC), dBih(iC), dVar(iC),
     &         wA, wB, myThid )
          CALL VIRP_PUT( wA, bi, bj, k, iO+1 )
          CALL VIRP_PUT( wB, bi, bj, k, iO+2 )
          iO = iO + 2
         ENDDO

C     out 48-55: MOM_VI_CORIOLIS, selectCoriScheme = 0..3
         DO iC = 0, 3
          selectCoriScheme = iC
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL VIRP_PRIOR( wB, bi, bj, k )
          CALL MOM_VI_CORIOLIS( bi, bj, k, uF, vF, hZ, rhZ,
     &                          wA, wB, myThid )
          CALL VIRP_PUT( wA, bi, bj, k, iO+1 )
          CALL VIRP_PUT( wB, bi, bj, k, iO+2 )
          iO = iO + 2
         ENDDO

C     out 56-61: MOM_VI_U_CORIOLIS, out 62-67: MOM_VI_V_CORIOLIS,
C     (selectVortScheme, useJamartMomAdv) cases c0..c5
         DO iC = 0, 5
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL MOM_VI_U_CORIOLIS( bi, bj, k, cSch(iC), cJam(iC),
     &                            vF, om3, hZ, rhZ, wA, myThid )
          CALL VIRP_PUT( wA, bi, bj, k, iO+1+iC )
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL MOM_VI_V_CORIOLIS( bi, bj, k, cSch(iC), cJam(iC),
     &                            uF, om3, hZ, rhZ, wA, myThid )
          CALL VIRP_PUT( wA, bi, bj, k, iO+7+iC )
         ENDDO
         iO = iO + 12

C     out 68-71: MOM_VI_U_VERTSHEAR, out 72-75: MOM_VI_V_VERTSHEAR,
C     (selectKEscheme, upwindShear) cases s0..s3
         DO iC = 0, 3
          selectKEscheme = sKE(iC)
          upwindShear    = sUpw(iC)
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL MOM_VI_U_VERTSHEAR( bi, bj, k, dFacA,
     &         in3(1-OLx,1-OLy,1,1,1,1), in3(1-OLx,1-OLy,1,1,1,3),
     &         wA, myThid )
          CALL VIRP_PUT( wA, bi, bj, k, iO+1+iC )
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL MOM_VI_V_VERTSHEAR( bi, bj, k, dFacA,
     &         in3(1-OLx,1-OLy,1,1,1,2), in3(1-OLx,1-OLy,1,1,1,3),
     &         wA, myThid )
          CALL VIRP_PUT( wA, bi, bj, k, iO+5+iC )
         ENDDO
         iO = iO + 8

C     out 76-77: MOM_VI_U_GRAD_KE, MOM_VI_V_GRAD_KE
         CALL VIRP_PRIOR( wA, bi, bj, k )
         CALL MOM_VI_U_GRAD_KE( bi, bj, k, KEi, wA, myThid )
         CALL VIRP_PUT( wA, bi, bj, k, iO+1 )
         CALL VIRP_PRIOR( wA, bi, bj, k )
         CALL MOM_VI_V_GRAD_KE( bi, bj, k, KEi, wA, myThid )
         CALL VIRP_PUT( wA, bi, bj, k, iO+2 )
         iO = iO + 2

C     out 78-79: MOM_V_CORIOLIS_NH, select3dCoriScheme = 1, 2
         DO iC = 1, 2
          select3dCoriScheme = iC
          CALL VIRP_PRIOR( wA, bi, bj, k )
          CALL MOM_V_CORIOLIS_NH( bi, bj, k,
     &         in3(1-OLx,1-OLy,1,1,1,3), wA, myThid )
          iO = iO + 1
          CALL VIRP_PUT( wA, bi, bj, k, iO )
         ENDDO

         IF ( iO.NE.79 ) STOP 'REPLAY: output count mismatch'
        ENDDO
       ENDDO
      ENDDO

C--   4. outputs: header int32 (MAGIC, VERSN, nOut3, Nr); out3
      OPEN( uOut, FILE='replay_out.bin', STATUS='NEW', ACCESS='STREAM',
     &      FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(1) = MAGIC
      ohdr(2) = VERSN
      ohdr(3) = nOut3
      ohdr(4) = Nr
      WRITE( uOut ) ohdr
      WRITE( uOut ) out3
      CLOSE( uOut )

      WRITE(standardMessageUnit,'(A,I4,A,I4,A,I4,A)')
     &  'REPLAY VECINV harness: ', nSx*nSy, ' tiles x', Nr,
     &  ' levels replayed,', nOut3, ' outputs'
      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: VIRP_WRITE_GRID
C     !INTERFACE:
      SUBROUTINE VIRP_WRITE_GRID( uGrd, dFacA, MAGIC, VERSN )
C     !DESCRIPTION: the grid, MOM_VISC.h and the PARAMS.h values as
C     MOM_VECINV (case m0) reads them (order: replay_io.py read_grid)
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "SURFACE.h"
#include "MOM_VISC.h"
      INTEGER uGrd, MAGIC, VERSN
      _RL dFacA(Nr)
CEOP
      INTEGER ohdr(4), iflg(33)
      _RL piRL
      INTEGER n
      OPEN( uGrd, FILE='replay_grid.bin', STATUS='NEW',
     &      ACCESS='STREAM', FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(1) = MAGIC
      ohdr(2) = VERSN
      ohdr(3) = 11
      ohdr(4) = 32
      WRITE( uGrd ) ohdr
      WRITE( uGrd ) hFacC, hFacW, hFacS, recip_hFacC, recip_hFacW,
     &              recip_hFacS, maskC, maskW, maskS
#ifdef NONLIN_FRSURF
      WRITE( uGrd ) h0FacW, h0FacS
#else
      WRITE( uGrd ) hFacW, hFacS
#endif
      WRITE( uGrd ) rA, rAw, rAs, rAz, recip_rA, recip_rAw, recip_rAs,
     &              recip_rAz, dxC, dyC, dxG, dyG, dxV, dyU,
     &              recip_dxC, recip_dyC, recip_dxG, recip_dyG,
     &              recip_dxV, recip_dyU, recip_dxF, recip_dyF,
     &              fCoriG, fCoriCos, angleCosC, angleSinC,
     &              L2_D, L2_Z, L3_D, L3_Z, L4rdt_D, L4rdt_Z
      WRITE( uGrd ) cosFacU, cosFacV, sqCosFacU, sqCosFacV
      WRITE( uGrd ) drF, recip_drF, recip_deepFacC, recip_deepFac2C,
     &              deepFacC, deepFac2C, recip_rhoFacC, dFacA,
     &              deepFacAdv
      WRITE( uGrd ) recip_drC, deepFac2F, rhoFacF, rVel2wUnit
      piRL = PI
      WRITE( uGrd ) rkSign, gravitySign, piRL, recip_rSphere,
     &  deltaTMom, deltaTClock, diagFreq, vfFacMom, sideDragFactor,
     &  bottomDragLinear, bottomDragQuadratic, rUnit2mass,
     &  viscAh, viscAhD, viscAhZ, viscAhGrid, viscAhMax,
     &  viscAhGridMax, viscAhGridMin, viscAhReMax,
     &  viscA4, viscA4D, viscA4Z, viscA4Grid, viscA4Max,
     &  viscA4GridMax, viscA4GridMin, viscA4ReMax,
     &  viscC2leith, viscC2leithD, viscC2LeithQG, viscC4leith,
     &  viscC4leithD, viscC2smag, viscC4smag, viscAhW, viscA4W
      DO n = 1, 33
       iflg(n) = 0
      ENDDO
      IF ( momViscosity )          iflg( 1) = 1
      IF ( momAdvection )          iflg( 2) = 1
      IF ( useCoriolis )           iflg( 3) = 1
      IF ( useCDscheme )           iflg( 4) = 1
      IF ( useAbsVorticity )       iflg( 5) = 1
      IF ( highOrderVorticity )    iflg( 6) = 1
      IF ( upwindVorticity )       iflg( 7) = 1
      iflg( 8) = selectVortScheme
      IF ( useJamartMomAdv )       iflg( 9) = 1
      IF ( upwindShear )           iflg(10) = 1
      iflg(11) = selectKEscheme
      IF ( momImplVertAdv )        iflg(12) = 1
      IF ( implicitViscosity )     iflg(13) = 1
      IF ( no_slip_sides )         iflg(14) = 1
      IF ( no_slip_bottom )        iflg(15) = 1
      iflg(16) = selectBotDragQuadr
      IF ( bottomVisc_pCell )      iflg(17) = 1
      iflg(18) = selectImplicitDrag
      iflg(19) = select3dCoriScheme
      IF ( useNHMTerms )           iflg(20) = 1
      IF ( deepAtmosphere )        iflg(21) = 1
      IF ( usingCurvilinearGrid )  iflg(22) = 1
      IF ( rotateGrid )            iflg(23) = 1
      IF ( useStrainTensionVisc )  iflg(24) = 1
      IF ( useHarmonicVisc )       iflg(25) = 1
      IF ( useBiharmonicVisc )     iflg(26) = 1
      IF ( useVariableVisc )       iflg(27) = 1
      IF ( useFullLeith )          iflg(28) = 1
      iflg(29) = nonlinFreeSurf
      iflg(30) = selectCoriScheme
      IF ( useCubedSphereExchange ) iflg(31) = 1
      IF ( usingZCoords )          iflg(32) = 1
      IF ( useAreaViscLength )     iflg(33) = 1
      WRITE( uGrd ) iflg
      CLOSE( uGrd )
      RETURN
      END

CBOP
C     !ROUTINE: VIRP_WRITE_W2
C     !INTERFACE:
      SUBROUTINE VIRP_WRITE_W2( uW2, MAGIC, VERSN )
C     !DESCRIPTION: replay_w2.bin: int32 MAGIC, VERSN, nTl, then per
C     tile (bi fastest) myTile = W2_myTileList(bi,bj), exch2_myFace,
C     exch2_isWedge, exch2_isEedge, exch2_isSedge, exch2_isNedge of
C     myTile (W2_EXCH2_TOPOLOGY.h); nTl = 0 without ALLOW_EXCH2
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#ifdef ALLOW_EXCH2
#include "W2_EXCH2_SIZE.h"
#include "W2_EXCH2_TOPOLOGY.h"
#endif
      INTEGER uW2, MAGIC, VERSN
CEOP
      INTEGER ohdr(3), itp(6), bi, bj, myTile
      OPEN( uW2, FILE='replay_w2.bin', STATUS='NEW',
     &      ACCESS='STREAM', FORM='UNFORMATTED', CONVERT='BIG_ENDIAN' )
      ohdr(1) = MAGIC
      ohdr(2) = VERSN
      ohdr(3) = 0
#ifdef ALLOW_EXCH2
      ohdr(3) = nSx*nSy
#endif
      WRITE( uW2 ) ohdr
#ifdef ALLOW_EXCH2
      DO bj = 1, nSy
       DO bi = 1, nSx
        myTile = W2_myTileList(bi,bj)
        itp(1) = myTile
        itp(2) = exch2_myFace(myTile)
        itp(3) = exch2_isWedge(myTile)
        itp(4) = exch2_isEedge(myTile)
        itp(5) = exch2_isSedge(myTile)
        itp(6) = exch2_isNedge(myTile)
        WRITE( uW2 ) itp
       ENDDO
      ENDDO
#endif
      CLOSE( uW2 )
      RETURN
      END

CBOP
C     !ROUTINE: VIRP_SETVISC
C     !INTERFACE:
      SUBROUTINE VIRP_SETVISC( vp, ful, har, bih )
C     !DESCRIPTION: PARAMS.h / MOM_VISC.h values of one MOM_CALC_VISC
C     case (order: replay_io.py VPAR)
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "MOM_VISC.h"
      _RL vp(20)
      LOGICAL ful, har, bih
CEOP
      viscAhD       = vp( 1)
      viscAhZ       = vp( 2)
      viscA4D       = vp( 3)
      viscA4Z       = vp( 4)
      viscAhGrid    = vp( 5)
      viscA4Grid    = vp( 6)
      viscAhMax     = vp( 7)
      viscA4Max     = vp( 8)
      viscAhGridMax = vp( 9)
      viscAhGridMin = vp(10)
      viscA4GridMax = vp(11)
      viscA4GridMin = vp(12)
      viscAhReMax   = vp(13)
      viscA4ReMax   = vp(14)
      viscC2leith   = vp(15)
      viscC2leithD  = vp(16)
      viscC4leith   = vp(17)
      viscC4leithD  = vp(18)
      viscC2smag    = vp(19)
      viscC4smag    = vp(20)
      useFullLeith      = ful
      useHarmonicVisc   = har
      useBiharmonicVisc = bih
      RETURN
      END

CBOP
C     !ROUTINE: VIRP_PRIOR
C     !INTERFACE:
      SUBROUTINE VIRP_PRIOR( a, bi, bj, k )
C     !DESCRIPTION: a(:,:) = prior field (in3 field 22, level k)
      IMPLICIT NONE
#include "SIZE.h"
      INTEGER nIn3, nOut3
      PARAMETER ( nIn3 = 22, nOut3 = 91 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / VIRP_BUF / in3, out3
      _RL a(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER bi, bj, k
CEOP
      INTEGER i, j
      DO j = 1-OLy, sNy+OLy
       DO i = 1-OLx, sNx+OLx
        a(i,j) = in3(i,j,k,bi,bj,22)
       ENDDO
      ENDDO
      RETURN
      END

CBOP
C     !ROUTINE: VIRP_PUT
C     !INTERFACE:
      SUBROUTINE VIRP_PUT( a, bi, bj, k, iO )
C     !DESCRIPTION: out3(:,:,k,bi,bj,iO) = a(:,:)
      IMPLICIT NONE
#include "SIZE.h"
      INTEGER nIn3, nOut3
      PARAMETER ( nIn3 = 22, nOut3 = 91 )
      _RL in3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nIn3)
      _RL out3(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nOut3)
      COMMON / VIRP_BUF / in3, out3
      _RL a(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER bi, bj, k, iO
CEOP
      INTEGER i, j
      IF ( iO.LT.1 .OR. iO.GT.nOut3 ) STOP 'REPLAY: bad output index'
      DO j = 1-OLy, sNy+OLy
       DO i = 1-OLx, sNx+OLx
        out3(i,j,k,bi,bj,iO) = a(i,j)
       ENDDO
      ENDDO
      RETURN
      END
