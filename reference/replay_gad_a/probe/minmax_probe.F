C     MAX/MIN probe of the GAD-A lane (reference/replay_gad_a/probe):
C     the values gfortran's REAL*8 MAX(a,b) and MIN(a,b) give on
C     ties (+0, -0) and NaN, compiled with the oracle's flags
C     (-O0 -ffp-contract=off, gfortran 11.2.0). Operands are run-time
C     values (read from stdin), so nothing is constant-folded.
C     Output: one line per (op, a, b): op a b result as Z16.16 bits.
      PROGRAM MMPROBE
      IMPLICIT NONE
      Real*8 v(5), r
      INTEGER i, j, n
      READ(*,*) n
      v(1) = 0.D0
      v(2) = -v(1)
      v(3) = 1.D0
      v(4) = -1.D0
      v(5) = v(1)
      IF ( n.EQ.1 ) v(5) = v(1)/v(1)
      DO i = 1, 5
       DO j = 1, 5
        r = MAX(v(i), v(j))
        WRITE(*,'(A,1X,Z16.16,1X,Z16.16,1X,Z16.16)')
     &        'MAX', v(i), v(j), r
        r = MIN(v(i), v(j))
        WRITE(*,'(A,1X,Z16.16,1X,Z16.16,1X,Z16.16)')
     &        'MIN', v(i), v(j), r
       ENDDO
      ENDDO
      END
