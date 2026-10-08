C     In-situ check (lane MINMAX): the oracle's own mon_advcflw2.o, linked
C     with this driver, on a tie of -0 (K term) and +0 (K-1 term) at
C     mon_advcflw2.F:43-44. RTL source operand: the K term wins (-0);
C     GIMPLE first operand: the K-1 term (+0).
#include "CPP_EEOPTIONS.h"
      PROGRAM DRV
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
      _RL W(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      _RS rHFac(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      _RS rDrF(Nr)
      _RL dT, z
      INTEGER i,j,k,bi,bj,n
      z = 0.D0
      READ(*,*) n
      myBxLo(1) = 1
      myBxHi(1) = 1
      myByLo(1) = 1
      myByHi(1) = 1
      DO bj=1,nSy
      DO bi=1,nSx
      DO k=1,Nr
      DO j=1-OLy,sNy+OLy
      DO i=1-OLx,sNx+OLx
        W(i,j,k,bi,bj) = 0.D0
        rHFac(i,j,k,bi,bj) = 0.D0
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      DO k=1,Nr
        rDrF(k) = 1.D0
      ENDDO
C     last point of the loop: K=Nr term = rDrF(2)*rHFac(.,2) = -0 (n=1) or NaN (n=2),
C     K-1 term = +0 (n=1) or 1 (n=2); W = 1, dT = 1
      W(sNx,sNy,Nr,1,1) = 1.D0
      IF ( n.EQ.1 ) rHFac(sNx,sNy,Nr,1,1) = -z
      IF ( n.EQ.2 ) rHFac(sNx,sNy,Nr,1,1) = z/z
      IF ( n.EQ.2 ) rHFac(sNx,sNy,Nr-1,1,1) = 1.D0
      dT = 1.D0
      CALL MON_ADVCFLW2( 'w2', W, rHFac, rDrF, dT, 1 )
      END
      SUBROUTINE GLOBAL_MAX_R8( x, myThid )
      REAL*8 x
      INTEGER myThid
      RETURN
      END
      SUBROUTINE MON_OUT_RL( label, val, foot, myThid )
      CHARACTER*(*) label, foot
      REAL*8 val
      INTEGER myThid
      WRITE(*,'(A,1X,Z16.16)') 'theMax', val
      RETURN
      END
