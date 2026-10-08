C     Running-MAX chain probe (lane MINMAX): the statement of cg2d.F:111
C     (rhsMax = MAX(ABS(cg2d_b(i,j,bi,bj)),rhsMax), rhsMax then passed to
C     a subroutine as _GLOBAL_MAX_RL does: A), the same statement with a
C     dummy-argument running value (B) and ini_cg2d.F:110-111's two
C     statements per point (C), over 4 tiles in the DO bj, DO bi, DO j,
C     DO i order, with NaN planted at several positions. Output per case:
C     the 24 inputs and the three results as Z16.16 bits.
      SUBROUTINE GMAX( x )
      REAL*8 x
      RETURN
      END
      SUBROUTINE CHAIN( b, rA, rB, rC )
      IMPLICIT NONE
      INTEGER sNx, sNy, nSx, nSy
      PARAMETER ( sNx = 3, sNy = 2, nSx = 2, nSy = 2 )
      REAL*8 b(sNx,sNy,nSx,nSy), rA, rB, rC
      REAL*8 s(sNx,sNy,nSx,nSy)
      REAL*8 rhsMax, myNorm
      INTEGER i, j, bi, bj
C     A: cg2d.F:105-111 (rhsMax local, its address passed on below)
      rhsMax = 0. D 0
      DO bj=1,nSy
       DO bi=1,nSx
        DO j=1,sNy
         DO i=1,sNx
          rhsMax = MAX(ABS(b(i,j,bi,bj)),rhsMax)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL GMAX( rhsMax )
      rA = rhsMax
C     B: the same statement with the running value a dummy argument
      rB = 0. D 0
      DO bj=1,nSy
       DO bi=1,nSx
        DO j=1,sNy
         DO i=1,sNx
          rB = MAX(ABS(b(i,j,bi,bj)),rB)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
C     C: ini_cg2d.F:102-116 (two statements per point)
      DO bj=1,nSy
       DO bi=1,nSx
        DO j=1,sNy
         DO i=1,sNx
          s(i,j,bi,bj) = 0.5D0*b(i,j,bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      myNorm = 0. D 0
      DO bj=1,nSy
       DO bi=1,nSx
        DO j=1,sNy
         DO i=1,sNx
          myNorm = MAX(ABS(b(i,j,bi,bj)),myNorm)
          myNorm = MAX(ABS(s(i,j,bi,bj)),myNorm)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL GMAX( myNorm )
      rC = myNorm
      RETURN
      END
      PROGRAM CHAINP
      IMPLICIT NONE
      REAL*8 b(3,2,2,2), rA, rB, rC, z, nan
      INTEGER i, j, bi, bj, c
      z = 0.D0
      nan = z/z
      DO c = 0, 8
       DO bj=1,2
        DO bi=1,2
         DO j=1,2
          DO i=1,3
           b(i,j,bi,bj) = (-1)**(i+j)*(1.D0+i+3*j+7*bi+13*bj)/8.D0
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       IF ( c.EQ.1 ) b(1,1,1,1) = nan
       IF ( c.EQ.2 ) b(2,1,2,1) = nan
       IF ( c.EQ.3 ) b(3,2,2,2) = nan
       IF ( c.EQ.4 ) b(3,2,2,1) = nan
       IF ( c.EQ.5 ) b(1,2,1,1) = nan
       IF ( c.EQ.5 ) b(2,2,1,2) = nan
       IF ( c.EQ.6 ) THEN
        DO bj=1,2
         DO bi=1,2
          DO j=1,2
           DO i=1,3
            b(i,j,bi,bj) = (-1)**(i+j)*z
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDIF
       IF ( c.EQ.7 ) THEN
        DO bj=1,2
         DO bi=1,2
          DO j=1,2
           DO i=1,3
            b(i,j,bi,bj) = nan
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDIF
       IF ( c.EQ.8 ) b(1,1,1,1) = 100.D0
       IF ( c.EQ.8 ) b(3,2,1,1) = nan
       CALL CHAIN( b, rA, rB, rC )
       WRITE(*,'(A,I1,24(1X,Z16.16))') 'IN', c, b
       WRITE(*,'(A,I1,3(1X,Z16.16))') 'OUT', c, rA, rB, rC
      ENDDO
      END
