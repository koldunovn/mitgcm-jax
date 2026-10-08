C     JAXDUMP.h -- state of the mitjax per-substep dump shim (reference/jaxdump/jaxdump.F).
C     jd_on      :: dumps enabled (JAXDUMP_DIR set and non-empty)
C     jd_inited  :: environment has been read
C     jd_nsteps  :: number of iterations to dump; jd_steps :: their myIter values (JAXDUMP_STEPS)
C     jd_seq     :: running record counter of this process (same for all tiles of one call)
C     jd_pass    :: > 0: suffix '_p<jd_pass>' appended to stage names (stages inside an iteration loop, JAXDUMP_PASS)
C     jd_ishift  :: subtracted from myIter: 0 from the start of FORWARD_STEP, 1 after its counter update
C                   (JAXDUMP_ITERSHIFT), so every record of a step carries the step's START iteration
C     jd_dir     :: output directory
      INTEGER jd_maxsteps
      PARAMETER ( jd_maxsteps = 200 )
      COMMON /JAXDUMP_L/ jd_on, jd_inited
      LOGICAL jd_on, jd_inited
      COMMON /JAXDUMP_I/ jd_nsteps, jd_steps, jd_seq, jd_pass,
     &                   jd_ishift
      INTEGER jd_nsteps, jd_steps(jd_maxsteps), jd_seq, jd_pass
      INTEGER jd_ishift
      COMMON /JAXDUMP_C/ jd_dir
      CHARACTER*(512) jd_dir
