#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

CBOP
C     !ROUTINE: THE_MAIN_LOOP
C     !INTERFACE:
      SUBROUTINE THE_MAIN_LOOP( myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE THE_MAIN_LOOP of the mitjax cube replay
C     | harness (reference/replay_cube; plan Task 22). Not an
C     | MITgcm routine: it replaces model/src/the_main_loop.F in
C     | a genmake2 build of a cubed-sphere experiment's code/.
C     | EEBOOT has run the W2 set-up (w2_tile_topology log and
C     | STDOUT) and THE_MODEL_MAIN has called INITIALISE_FIXED,
C     | so GRID.h holds the experiment's curvilinear grid. This
C     | routine writes (big-endian stream, layout in
C     | reference/replay_cube/replay_io.py):
C     |  cube_grid.bin  the GRID.h horizontal fields, all points
C     |  cube_exch.bin  every exchange routine on index-coded
C     |                 fields and on signed-zero fields
C     |  cube_fill.bin  FILL_CS_CORNER_TR_RL / UV_RS / UV_RL /
C     |                 AG_RL, every option, every tile, on
C     |                 index-coded and signed-zero fields
C     | No time stepping.
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
      _RL     myTime
      INTEGER myIter
      INTEGER myThid
CEOP
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261002, VERSN = 1 )
      INTEGER uOut, nRec
      CHARACTER*16 rName

      CALL PRINT_MESSAGE( 'REPLAY_CUBE harness: start',
     &     standardMessageUnit, SQUEEZE_RIGHT, myThid )

C--   1) grid
      uOut = 71
      OPEN( uOut, FILE='cube_grid.bin', ACCESS='STREAM',
     &      FORM='UNFORMATTED', STATUS='NEW' )
      nRec = 35
      WRITE(uOut) MAGIC, VERSN, sNx, sNy, OLx, OLy, nSx, nSy, nRec
      rName = 'xC'
      WRITE(uOut) rName, xC
      rName = 'yC'
      WRITE(uOut) rName, yC
      rName = 'xG'
      WRITE(uOut) rName, xG
      rName = 'yG'
      WRITE(uOut) rName, yG
      rName = 'dxC'
      WRITE(uOut) rName, dxC
      rName = 'dyC'
      WRITE(uOut) rName, dyC
      rName = 'dxF'
      WRITE(uOut) rName, dxF
      rName = 'dyF'
      WRITE(uOut) rName, dyF
      rName = 'dxG'
      WRITE(uOut) rName, dxG
      rName = 'dyG'
      WRITE(uOut) rName, dyG
      rName = 'dxV'
      WRITE(uOut) rName, dxV
      rName = 'dyU'
      WRITE(uOut) rName, dyU
      rName = 'rA'
      WRITE(uOut) rName, rA
      rName = 'rAw'
      WRITE(uOut) rName, rAw
      rName = 'rAs'
      WRITE(uOut) rName, rAs
      rName = 'rAz'
      WRITE(uOut) rName, rAz
      rName = 'angleCosC'
      WRITE(uOut) rName, angleCosC
      rName = 'angleSinC'
      WRITE(uOut) rName, angleSinC
      rName = 'u2zonDir'
      WRITE(uOut) rName, u2zonDir
      rName = 'v2zonDir'
      WRITE(uOut) rName, v2zonDir
      rName = 'recip_dxC'
      WRITE(uOut) rName, recip_dxC
      rName = 'recip_dyC'
      WRITE(uOut) rName, recip_dyC
      rName = 'recip_dxF'
      WRITE(uOut) rName, recip_dxF
      rName = 'recip_dyF'
      WRITE(uOut) rName, recip_dyF
      rName = 'recip_dxG'
      WRITE(uOut) rName, recip_dxG
      rName = 'recip_dyG'
      WRITE(uOut) rName, recip_dyG
      rName = 'recip_dxV'
      WRITE(uOut) rName, recip_dxV
      rName = 'recip_dyU'
      WRITE(uOut) rName, recip_dyU
      rName = 'recip_rA'
      WRITE(uOut) rName, recip_rA
      rName = 'recip_rAw'
      WRITE(uOut) rName, recip_rAw
      rName = 'recip_rAs'
      WRITE(uOut) rName, recip_rAs
      rName = 'recip_rAz'
      WRITE(uOut) rName, recip_rAz
      rName = 'fCori'
      WRITE(uOut) rName, fCori
      rName = 'fCoriG'
      WRITE(uOut) rName, fCoriG
      rName = 'fCoriCos'
      WRITE(uOut) rName, fCoriCos
      CLOSE( uOut )

C--   2) exchanges
      CALL REPLAY_CUBE_EXCH( myThid )
C--   3) corner fills
      CALL REPLAY_CUBE_FILL( myThid )

      CALL PRINT_MESSAGE( 'REPLAY_CUBE harness: done',
     &     standardMessageUnit, SQUEEZE_RIGHT, myThid )
      RETURN
      END

C---------------------------------------------------------------------
C     Index codes as lane A's JAXDUMP_PROBE_FILL (jaxdump.F): value
C     comp*cbase + ((tile-1)*nyp + j+OLy-1)*nxp + i+OLx-1 + 1, comp 1 in
C     pu, 2 in pv, tile = W2_myTileList(bi,bj); cbase the smallest power
C     of 10 above nTiles*nyp*nxp. iz > 0: signed zeros instead (iz = 1:
C     u,v +0; 2: u,v -0; 3: u -0, v +0; 4: u +0, v -0).
      SUBROUTINE REPLAY_CUBE_PFILL( pu, pv, cbase, iz, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "W2_EXCH2_SIZE.h"
#include "W2_EXCH2_TOPOLOGY.h"
      INTEGER iz, myThid
      _RL pu(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL pv(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL cbase, idx, zu, zv
      INTEGER i, j, bi, bj, tile, nxp, nyp
      nxp = sNx + 2*OLx
      nyp = sNy + 2*OLy
      zu = 0. _d 0
      zv = 0. _d 0
      IF ( iz .EQ. 2 .OR. iz .EQ. 3 ) zu = SIGN( 0. _d 0, -1. _d 0 )
      IF ( iz .EQ. 2 .OR. iz .EQ. 4 ) zv = SIGN( 0. _d 0, -1. _d 0 )
      DO bj = 1, nSy
       DO bi = 1, nSx
        tile = W2_myTileList(bi,bj)
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          IF ( iz .EQ. 0 ) THEN
           idx = DBLE( (tile-1)*nyp + (j+OLy-1) )*DBLE( nxp )
     &         + DBLE( i+OLx-1 ) + 1. _d 0
           pu(i,j,bi,bj) = cbase + idx
           pv(i,j,bi,bj) = 2. _d 0*cbase + idx
          ELSE
           pu(i,j,bi,bj) = zu
           pv(i,j,bi,bj) = zv
          ENDIF
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      RETURN
      END

      SUBROUTINE REPLAY_CUBE_CBASE( cbase, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "W2_EXCH2_SIZE.h"
#include "W2_EXCH2_TOPOLOGY.h"
      _RL cbase, ncode
      INTEGER myThid
      ncode = DBLE(exch2_nTiles)*DBLE(sNy+2*OLy)*DBLE(sNx+2*OLx)
      cbase = 1. _d 0
      DO WHILE ( cbase .LE. ncode )
       cbase = cbase*10. _d 0
      ENDDO
      IF ( cbase .GT. 1. _d 15 ) STOP 'REPLAY_CUBE: code overflow'
      RETURN
      END

C---------------------------------------------------------------------
      SUBROUTINE REPLAY_CUBE_EXCH( myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
      INTEGER myThid
      INTEGER MAGIC, VERSN, NKIND
      PARAMETER ( MAGIC = 20261002, VERSN = 1, NKIND = 16 )
      COMMON /REPLAY_CUBE_P/ pu, pv, ps
      _RL pu(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL pv(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL ps(0:sNx+1,0:sNy+1,nSx,nSy)
      _RL cbase
      INTEGER uOut, m, iz, i, j, bi, bj, nRec
      CHARACTER*16 rName
      CHARACTER*6 kind(NKIND)
      LOGICAL isVec(NKIND)
      DATA kind / 'XY', 'UVs', 'UVn', 'Z', 'As', 'An', 'Bs', 'Bn',
     &            '3D', 'SMs', 'SMn', 'Ds', 'Dn', 'UV3s', 'UV3n',
     &            'S3D' /
      DATA isVec / .FALSE., .TRUE., .TRUE., .FALSE., .TRUE., .TRUE.,
     &             .TRUE., .TRUE., .FALSE., .FALSE., .FALSE., .TRUE.,
     &             .TRUE., .TRUE., .TRUE., .FALSE. /

      CALL REPLAY_CUBE_CBASE( cbase, myThid )
      uOut = 72
      OPEN( uOut, FILE='cube_exch.bin', ACCESS='STREAM',
     &      FORM='UNFORMATTED', STATUS='NEW' )
C     records: per kind code probe (u [, v]); per iz 1..4 and kind the
C     signed-zero probe (u [, v]); every record named
      nRec = 0
      DO m = 1, NKIND
       nRec = nRec + 1
       IF ( isVec(m) ) nRec = nRec + 1
      ENDDO
      nRec = nRec*5
      WRITE(uOut) MAGIC, VERSN, sNx, sNy, OLx, OLy, nSx, nSy, nRec
      WRITE(uOut) cbase
      DO iz = 0, 4
       DO m = 1, NKIND
        CALL REPLAY_CUBE_PFILL( pu, pv, cbase, iz, myThid )
        IF ( m .EQ. 1 ) THEN
         CALL EXCH_XY_RL( pu, myThid )
        ELSEIF ( m .EQ. 2 ) THEN
         CALL EXCH_UV_XY_RL( pu, pv, .TRUE., myThid )
        ELSEIF ( m .EQ. 3 ) THEN
         CALL EXCH_UV_XY_RL( pu, pv, .FALSE., myThid )
        ELSEIF ( m .EQ. 4 ) THEN
         CALL EXCH_Z_3D_RL( pu, 1, myThid )
        ELSEIF ( m .EQ. 5 ) THEN
         CALL EXCH_UV_AGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
        ELSEIF ( m .EQ. 6 ) THEN
         CALL EXCH_UV_AGRID_3D_RL( pu, pv, .FALSE., 1, myThid )
        ELSEIF ( m .EQ. 7 ) THEN
         CALL EXCH_UV_BGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
        ELSEIF ( m .EQ. 8 ) THEN
         CALL EXCH_UV_BGRID_3D_RL( pu, pv, .FALSE., 1, myThid )
        ELSEIF ( m .EQ. 9 ) THEN
         CALL EXCH_3D_RL( pu, 1, myThid )
        ELSEIF ( m .EQ. 10 ) THEN
         CALL EXCH_SM_3D_RL( pu, .TRUE., 1, myThid )
        ELSEIF ( m .EQ. 11 ) THEN
         CALL EXCH_SM_3D_RL( pu, .FALSE., 1, myThid )
        ELSEIF ( m .EQ. 12 ) THEN
         CALL EXCH_UV_DGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
        ELSEIF ( m .EQ. 13 ) THEN
         CALL EXCH_UV_DGRID_3D_RL( pu, pv, .FALSE., 1, myThid )
        ELSEIF ( m .EQ. 14 ) THEN
         CALL EXCH_UV_3D_RL( pu, pv, .TRUE., 1, myThid )
        ELSEIF ( m .EQ. 15 ) THEN
         CALL EXCH_UV_3D_RL( pu, pv, .FALSE., 1, myThid )
        ELSEIF ( m .EQ. 16 ) THEN
C        EXCH_S3D_RL on (0:sNx+1,0:sNy+1) arrays, copied in and out of
C        the coded full array (as jaxdump.F's probe)
         DO bj = 1, nSy
          DO bi = 1, nSx
           DO j = 0, sNy+1
            DO i = 0, sNx+1
             ps(i,j,bi,bj) = pu(i,j,bi,bj)
            ENDDO
           ENDDO
          ENDDO
         ENDDO
         CALL EXCH_S3D_RL( ps, 1, myThid )
         DO bj = 1, nSy
          DO bi = 1, nSx
           DO j = 0, sNy+1
            DO i = 0, sNx+1
             pu(i,j,bi,bj) = ps(i,j,bi,bj)
            ENDDO
           ENDDO
          ENDDO
         ENDDO
        ENDIF
        WRITE(rName,'(A,I1,2A)') 'z', iz, TRIM(kind(m)), '_u'
        WRITE(uOut) rName, pu
        IF ( isVec(m) ) THEN
         WRITE(rName,'(A,I1,2A)') 'z', iz, TRIM(kind(m)), '_v'
         WRITE(uOut) rName, pv
        ENDIF
       ENDDO
      ENDDO
      CLOSE( uOut )
      RETURN
      END

C---------------------------------------------------------------------
      SUBROUTINE REPLAY_CUBE_FILL( myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
      INTEGER myThid
      INTEGER MAGIC, VERSN
      PARAMETER ( MAGIC = 20261002, VERSN = 1 )
      COMMON /REPLAY_CUBE_P/ pu, pv, ps
      _RL pu(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL pv(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL ps(0:sNx+1,0:sNy+1,nSx,nSy)
      _RS ru(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RS rv(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL cbase
      INTEGER uOut, m, iz, i, j, bi, bj, nRec, d, s
      LOGICAL wS, fX
      CHARACTER*16 rName

      CALL REPLAY_CUBE_CBASE( cbase, myThid )
      uOut = 73
      OPEN( uOut, FILE='cube_fill.bin', ACCESS='STREAM',
     &      FORM='UNFORMATTED', STATUS='NEW' )
C     per iz (0..4): TR 3 dirs x 2 signs (u only), UV_RS 2, UV_RL 2,
C     AG_RL 2 dirs x 2 signs (u and v): 6 + 2*(2+2+4) = 22 records
      nRec = 22*5
      WRITE(uOut) MAGIC, VERSN, sNx, sNy, OLx, OLy, nSx, nSy, nRec
      WRITE(uOut) cbase
      DO iz = 0, 4
C      FILL_CS_CORNER_TR_RL( fill4dir, withSigns )
       DO d = 0, 2
        DO s = 1, 2
         wS = s .EQ. 1
         CALL REPLAY_CUBE_PFILL( pu, pv, cbase, iz, myThid )
         DO bj = 1, nSy
          DO bi = 1, nSx
           CALL FILL_CS_CORNER_TR_RL( d, wS, pu(1-OLx,1-OLy,bi,bj),
     &                                bi, bj, myThid )
          ENDDO
         ENDDO
         WRITE(rName,'(A,I1,A,I1,A,I1,A)') 'z',iz,'TR',d,'s',s,'_u'
         WRITE(uOut) rName, pu
        ENDDO
       ENDDO
C      FILL_CS_CORNER_UV_RS( withSigns ) on _RS copies
       DO s = 1, 2
        wS = s .EQ. 1
        CALL REPLAY_CUBE_PFILL( pu, pv, cbase, iz, myThid )
        DO bj = 1, nSy
         DO bi = 1, nSx
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            ru(i,j,bi,bj) = pu(i,j,bi,bj)
            rv(i,j,bi,bj) = pv(i,j,bi,bj)
           ENDDO
          ENDDO
          CALL FILL_CS_CORNER_UV_RS( wS, ru(1-OLx,1-OLy,bi,bj),
     &                         rv(1-OLx,1-OLy,bi,bj), bi, bj, myThid )
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            pu(i,j,bi,bj) = ru(i,j,bi,bj)
            pv(i,j,bi,bj) = rv(i,j,bi,bj)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        WRITE(rName,'(A,I1,A,I1,A)') 'z',iz,'UVRSs',s,'_u'
        WRITE(uOut) rName, pu
        WRITE(rName,'(A,I1,A,I1,A)') 'z',iz,'UVRSs',s,'_v'
        WRITE(uOut) rName, pv
       ENDDO
C      FILL_CS_CORNER_UV_RL( withSigns )
       DO s = 1, 2
        wS = s .EQ. 1
        CALL REPLAY_CUBE_PFILL( pu, pv, cbase, iz, myThid )
        DO bj = 1, nSy
         DO bi = 1, nSx
          CALL FILL_CS_CORNER_UV_RL( wS, pu(1-OLx,1-OLy,bi,bj),
     &                         pv(1-OLx,1-OLy,bi,bj), bi, bj, myThid )
         ENDDO
        ENDDO
        WRITE(rName,'(A,I1,A,I1,A)') 'z',iz,'UVRLs',s,'_u'
        WRITE(uOut) rName, pu
        WRITE(rName,'(A,I1,A,I1,A)') 'z',iz,'UVRLs',s,'_v'
        WRITE(uOut) rName, pv
       ENDDO
C      FILL_CS_CORNER_AG_RL( fill4dirX, withSigns )
       DO d = 1, 2
        fX = d .EQ. 1
        DO s = 1, 2
         wS = s .EQ. 1
         CALL REPLAY_CUBE_PFILL( pu, pv, cbase, iz, myThid )
         DO bj = 1, nSy
          DO bi = 1, nSx
           CALL FILL_CS_CORNER_AG_RL( fX, wS, pu(1-OLx,1-OLy,bi,bj),
     &                         pv(1-OLx,1-OLy,bi,bj), bi, bj, myThid )
          ENDDO
         ENDDO
         WRITE(rName,'(A,I1,A,I1,A,I1,A)') 'z',iz,'AG',d,'s',s,'_u'
         WRITE(uOut) rName, pu
         WRITE(rName,'(A,I1,A,I1,A,I1,A)') 'z',iz,'AG',d,'s',s,'_v'
         WRITE(uOut) rName, pv
        ENDDO
       ENDDO
      ENDDO
      CLOSE( uOut )
      RETURN
      END
