C     MIN/MAX operand-order probe (mon_intrinsics.py): build as make_probe.py says, plus -fdump-tree-optimized.
      PROGRAM PROBE2
      IMPLICIT NONE
      REAL*8 zeroRL
      PARAMETER ( zeroRL = 0.0 D 0 )
      REAL*8 R(4), RK(4), MX(3), T, V(4)
      INTEGER N
      R(1) = 0.D0
      R(2) = -0.D0
      R(3) = 1.D0
      R(4) = -1.D0
      DO N=1,4
       RK(N) = R(N)
       V(N) = R(N)
      ENDDO
      MX(1) = 0.
      MX(2) = 0.
      MX(3) = 0.
      DO N=1,4
       T = MAX( R(N), zeroRL )
       WRITE(*,'(A,2(1X,Z16.16))') 'MAXZ', R(N), T
       T = MIN( RK(N), zeroRL )
       WRITE(*,'(A,2(1X,Z16.16))') 'MINZ', RK(N), T
       T = ( + MAX( R(N), zeroRL ) - MIN( RK(1), zeroRL ) )*1.D0
       WRITE(*,'(A,2(1X,Z16.16))') 'TMPV', R(N), T
       MX(3) = MAX( MX(3), T )
       WRITE(*,'(A,2(1X,Z16.16))') 'CHAI', T, MX(3)
      ENDDO
      MX(1) = 0.
      MX(1) = MAX( MX(1), V(2) )
      WRITE(*,'(A,2(1X,Z16.16))') 'ARR0', V(2), MX(1)
      T = 0.
      T = MAX( T, V(2) )
      WRITE(*,'(A,2(1X,Z16.16))') 'SCL0', V(2), T
      END
