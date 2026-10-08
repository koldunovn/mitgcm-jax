C     Statement-function MAX/MIN probe (lane MINMAX): GAD_FLUX_LIMITER.h's
C     Superbee limiter as gad_fluxlimit_adv_x compiles it, and inline.
      SUBROUTINE SF( a, r )
      IMPLICIT NONE
      Real*8 a(2), r(4)
      Real*8 Cr, Cr2
      Real*8 Limiter
      Limiter(Cr)=max(0.d0,max(min(1.d0,2.d0*Cr),
     &                         min(2.d0,Cr)))
      Cr = a(1)
      Cr = Limiter(Cr)
      r(1) = Cr
      Cr2 = a(1)
      Cr2 = max(0.d0,max(min(1.d0,2.d0*Cr2),
     &                         min(2.d0,Cr2)))
      r(2) = Cr2
      Cr2 = a(1)
      r(3) = min(2.d0,Cr2)
      r(4) = min(Cr2,2.d0)
      RETURN
      END
      PROGRAM SFP
      IMPLICIT NONE
      Real*8 v(6), a(2), r(4), z
      INTEGER i, k
      z = 0.D0
      v(1) = z/z
      v(2) = z
      v(3) = -z
      v(4) = 2.D0
      v(5) = 0.5D0
      v(6) = -1.D0
      DO i = 1, 6
       a(1) = v(i)
       CALL SF( a, r )
       DO k = 1, 4
        WRITE(*,'(A,I1,1X,Z16.16,1X,Z16.16)') 'R', k, v(i), r(k)
       ENDDO
      ENDDO
      END
