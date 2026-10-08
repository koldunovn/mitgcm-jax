C     Flush-to-zero probe of the FTZ oracle variants (plan decision 13; reference/relink_ftz.sh).
C     Compiled once with the oracle build's own compiler and flags, linked twice: as the standard build links and with
C     the compiler's crtfastmath.o. Operands come from the command line, so nothing is folded at compile time.
C       ftz_probe 1.D-300 1.D-10   a normal x normal product in the subnormal range (FTZ: result 0)
C       ftz_probe 1.D-310 1.D0     a subnormal operand (DAZ: treated as 0, result 0)
      PROGRAM FTZ_PROBE
      IMPLICIT NONE
      REAL*8 x, y, z
      CHARACTER*40 arg
      CALL GET_COMMAND_ARGUMENT( 1, arg )
      READ(arg,*) x
      CALL GET_COMMAND_ARGUMENT( 2, arg )
      READ(arg,*) y
      z = x*y
      WRITE(*,'(A,3ES26.17E3)') 'FTZPROBE x, y, x*y: ', x, y, z
      END
