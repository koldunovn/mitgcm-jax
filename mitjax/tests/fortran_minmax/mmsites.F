C     MAX/MIN operand-order probe (lane MINMAX): one site per argument form.
      SUBROUTINE NOOP( x )
      REAL*8 x
      RETURN
      END
      SUBROUTINE SITES( a, b, c, ds, r )
      IMPLICIT NONE
      REAL*8 zeroRL
      PARAMETER ( zeroRL = 0.0 D 0 )
      REAL*8 a(2), b(2), c(2), ds, r(40)
      REAL*8 s, x, y, x2, y2, cA, cB
      COMMON /CMM/ cA, cB
      INTEGER i
C     S1-S2 dummy array elements
      r(1) = MAX( a(1), b(1) )
      r(2) = MIN( a(1), b(1) )
C     S3-S4 running local scalar, new value second / first
      s = a(1)
      s = MAX( s, b(1) )
      r(3) = s
      s = b(1)
      s = MAX( a(1), s )
      r(4) = s
C     S5 two local non-addressable scalars
      x = a(1)
      y = b(1)
      r(5) = MAX( x, y )
C     S6 two addressable locals
      x2 = a(1)
      y2 = b(1)
      CALL NOOP( x2 )
      CALL NOOP( y2 )
      r(6) = MAX( x2, y2 )
C     S7 COMMON block scalars
      cA = a(1)
      cB = b(1)
      r(7) = MAX( cA, cB )
C     S8-S9 dummy scalar first / second
      r(8) = MAX( ds, b(1) )
      r(9) = MAX( b(1), ds )
C     S10 expressions
      r(10) = MAX( a(1)*1.D0, b(1)*1.D0 )
C     S11 three arguments (c(1) = -1 below both)
      r(11) = MAX( a(1), b(1), c(1) )
C     S12 three arguments, the small one first
      r(12) = MAX( c(1), a(1), b(1) )
C     S13-S16 literal / PARAMETER zero
      r(13) = MAX( a(1), 0.D0 )
      r(14) = MAX( 0.D0, a(1) )
      r(15) = MIN( a(1), zeroRL )
      r(16) = MIN( zeroRL, a(1) )
C     S17-S18 local scalar with array element
      x = a(1)
      r(17) = MAX( x, b(1) )
      r(18) = MAX( b(1), x )
C     S19 loop accumulation, s = MAX(s, a(i))
      s = b(1)
      DO i = 1, 1
       s = MAX( s, a(i) )
      ENDDO
      r(19) = s
C     S20 MIN mixed: local first, dummy array second
      r(20) = MIN( x, b(1) )
C     S21 product of dummies vs array (like rDrF(K)*rHfac(K))
      r(21) = MAX( a(1)*a(2), b(1)*b(2) )
C     S22 MAX of MAX (nested)
      r(22) = MAX( MAX( a(1), c(1) ), b(1) )
C     S23 ABS argument
      r(23) = MAX( ABS(a(1)), b(1) )
      RETURN
      END
      PROGRAM MMSITES
      IMPLICIT NONE
      REAL*8 v(4), a(2), b(2), c(2), r(40), nanv
      INTEGER i, j, k, n
      READ(*,*) n
      v(1) = 0.D0
      v(2) = -v(1)
      v(3) = 1.D0
      nanv = v(1)
      IF ( n.EQ.1 ) nanv = v(1)/v(1)
      v(4) = nanv
      a(2) = 1.D0
      b(2) = 1.D0
      c(1) = -1.D0
      c(2) = -1.D0
      DO i = 1, 4
       DO j = 1, 4
        a(1) = v(i)
        b(1) = v(j)
        DO k = 1, 40
         r(k) = 7.D0
        ENDDO
        CALL SITES( a, b, c, a(1), r )
        DO k = 1, 23
         WRITE(*,'(A,I2.2,1X,Z16.16,1X,Z16.16,1X,Z16.16)')
     &        'S', k, a(1), b(1), r(k)
        ENDDO
       ENDDO
      ENDDO
      END
