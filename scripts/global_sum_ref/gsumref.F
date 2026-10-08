C     Fortran-order reference for mitjax/eesupp/global_sum.py (plan Task 7b), compiled with the oracle's
C     compiler and flags by scripts/global_sum_ref/run.sbatch.
C
C     Transcribes the two loops a single-process MITgcm build runs for a global sum at 63cdc0b:
C       per-tile partials, the callers' loops, e.g. model/src/cg2d.F:157 and :161-175
C         errTile(bi,bj) = 0. _d 0   (here 0.D0: _d is the D exponent, CPP_EEMACROS.h)
C         DO j=1,sNy; DO i=1,sNx
C           errTile(bi,bj) = errTile(bi,bj) + cg2d_r(i,j,bi,bj)*cg2d_r(i,j,bi,bj)
C       and the tile sum of eesupp/src/global_sum_tile.F:199-204 (path without MPI)
C         sumMyPr = 0.
C         DO bj = 1,nSy; DO bi = 1,nSx
C           sumMyPr = sumMyPr + shareBufGSR8(bi,bj)
C     Tiles are stored in the order bi + (bj-1)*nSx, the order of that loop.
C
C     input.bin (big-endian stream): nCase; per case nT, sNy, sNx, a(sNx,sNy,nT), b(sNx,sNy,nT)
C     output.bin: per case partA(nT), sumA, partAB(nT), sumAB
C       A: terms a(i,j)            AB: terms a(i,j)*b(i,j) (the product formed first, as cg2d.F:173-174)
      PROGRAM GSUMREF
      IMPLICIT NONE
      INTEGER nCase, ic, nT, sNy, sNx, t, i, j
      REAL*8, ALLOCATABLE :: a(:,:,:), b(:,:,:)
      REAL*8, ALLOCATABLE :: partA(:), partAB(:)
      REAL*8 sumA, sumAB

      OPEN( 10, FILE='input.bin', ACCESS='STREAM',
     &      FORM='UNFORMATTED', STATUS='OLD' )
      OPEN( 11, FILE='output.bin', ACCESS='STREAM',
     &      FORM='UNFORMATTED', STATUS='NEW' )
      READ(10) nCase
      DO ic = 1, nCase
       READ(10) nT, sNy, sNx
       ALLOCATE( a(sNx,sNy,nT), b(sNx,sNy,nT) )
       ALLOCATE( partA(nT), partAB(nT) )
       READ(10) a
       READ(10) b
       DO t = 1, nT
        partA(t)  = 0.D0
        partAB(t) = 0.D0
        DO j = 1, sNy
         DO i = 1, sNx
          partA(t)  = partA(t)  + a(i,j,t)
          partAB(t) = partAB(t)
     &              + a(i,j,t)*b(i,j,t)
         ENDDO
        ENDDO
       ENDDO
       sumA  = 0.
       sumAB = 0.
       DO t = 1, nT
        sumA  = sumA  + partA(t)
        sumAB = sumAB + partAB(t)
       ENDDO
       WRITE(11) partA, sumA, partAB, sumAB
       DEALLOCATE( a, b, partA, partAB )
      ENDDO
      CLOSE(10)
      CLOSE(11)
      END
