#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"
#ifdef ALLOW_GMREDI
# include "GMREDI_OPTIONS.h"
#endif
#ifdef ALLOW_GGL90
# include "GGL90_OPTIONS.h"
#endif
#ifdef ALLOW_KPP
# include "KPP_OPTIONS.h"
#endif
#ifdef ALLOW_PP81
# include "PP81_OPTIONS.h"
#endif
#ifdef ALLOW_MY82
# include "MY82_OPTIONS.h"
#endif
#ifdef ALLOW_COST
# include "COST_OPTIONS.h"
#endif
#ifdef ALLOW_EXF
# include "EXF_OPTIONS.h"
#endif
#ifdef ALLOW_SEAICE
# include "SEAICE_OPTIONS.h"
#endif
#ifdef ALLOW_THSICE
# include "THSICE_OPTIONS.h"
#endif
#ifdef ALLOW_SALT_PLUME
# include "SALT_PLUME_OPTIONS.h"
#endif

C=====================================================================================================================
C  jaxdump.F -- per-substep dump shim for the mitjax Fortran oracle (plan Task 4; copied from the ECCO port's
C  reference/jaxdump/jaxdump.F and adapted to MITgcm master 63cdc0b and the M1 experiments, which include exch1
C  layouts: reference/jaxdump/SUBSTEPS.md lists every change).
C
C  Enabled only when the environment variable JAXDUMP_DIR is set: with it unset every entry point returns at once,
C  and the model's state files and STDOUT are byte-identical to the uninstrumented build (scripts/tests/
C  test_jaxdump.py). Nothing is printed to STDOUT even with dumps on: the settings go to JAXDUMP_DIR/jaxdump_info.txt.
C  JAXDUMP_STEPS = myIter values separated by ',' or ':' (':' survives sbatch --export);
C  default nIter0, nIter0+1, nIter0+2. Every record of a step carries the step's START iteration (JAXDUMP_ITERSHIFT).
C
C  Output: one stream file per (iteration, tile): JAXDUMP_DIR/jd_<iter %010d>_t<tile %04d>.bin, records appended
C  in call order. Big-endian (the build uses -fconvert=big-endian). Record (version 2):
C     int32  magic = 1245990224 ('JDMP'), version = 2, iter, seq
C     char*32 stage, char*32 field, char*4 kind (C W S Z: point type; N scalar; V packed vertical grid)
C     int32  nz, sNx, sNy, OLx, OLy, tile, face, tBasex, tBasey
C     real*8 values(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, nz)   -- halos INCLUDED
C  Global key of value (i,j,k): face, iG = tBasex+i, jG = tBasey+j (halo points fall outside 1..tNx), k.
C  exch2 builds: tile, face, tBasex/y from the W2 topology (exch2_myFace, exch2_tBasex/y). exch1 builds (no
C  pkg/exch2; version 1 of the ECCO port wrote face=0, tBase=0 here): face = 0 (one global domain Nx x Ny),
C  tBasex = myXGlobalLo-1+(bi-1)*sNx, tBasey = myYGlobalLo-1+(bj-1)*sNy, tile = 1+tBasex/sNx+(tBasey/sNy)*nSx*nPx.
C  Reader: mitjax/io/dump.py.
C=====================================================================================================================

      BLOCK DATA JAXDUMP_BD
      IMPLICIT NONE
#include "JAXDUMP.h"
      DATA jd_inited /.FALSE./, jd_on /.FALSE./, jd_seq /0/, jd_pass /0/
      DATA jd_ishift /0/
      END

C---------------------------------------------------------------------------------------------------------------------
      SUBROUTINE JAXDUMP_INIT( myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "JAXDUMP.h"
      INTEGER myThid
      EXTERNAL JAXDUMP_BD
      CHARACTER*(512) sbuf
      INTEGER ln, st, i, i0, n, ios, iUnit
      INTEGER ILNBLNK
      EXTERNAL ILNBLNK

      jd_inited = .TRUE.
      jd_on = .FALSE.
      jd_nsteps = 0
      CALL GET_ENVIRONMENT_VARIABLE( 'JAXDUMP_DIR', jd_dir, ln, st )
      IF ( st .NE. 0 .OR. ln .EQ. 0 ) RETURN
      jd_on = .TRUE.
      CALL GET_ENVIRONMENT_VARIABLE( 'JAXDUMP_STEPS', sbuf, ln, st )
      IF ( st .NE. 0 .OR. ln .EQ. 0 ) THEN
        jd_nsteps = 3
        jd_steps(1) = nIter0
        jd_steps(2) = nIter0 + 1
        jd_steps(3) = nIter0 + 2
      ELSE
C       parse "a,b,c" or "a:b:c"
        i0 = 1
        DO i = 1, ln + 1
          IF ( i .GT. ln .OR. sbuf(i:i) .EQ. ','
     &                   .OR. sbuf(i:i) .EQ. ':' ) THEN
            IF ( i .GT. i0 ) THEN
              READ( sbuf(i0:i-1), *, IOSTAT=ios ) n
              IF ( ios .NE. 0 )
     &          STOP 'JAXDUMP: cannot parse JAXDUMP_STEPS'
              IF ( jd_nsteps .GE. jd_maxsteps )
     &          STOP 'JAXDUMP: too many JAXDUMP_STEPS'
              jd_nsteps = jd_nsteps + 1
              jd_steps(jd_nsteps) = n
            ENDIF
            i0 = i + 1
          ENDIF
        ENDDO
      ENDIF
C     settings into the dump directory, not STDOUT (STDOUT stays byte-identical with dumps on)
      CALL MDSFINDUNIT( iUnit, myThid )
      OPEN( iUnit, FILE=jd_dir(1:ILNBLNK(jd_dir))//'/jaxdump_info.txt',
     &      FORM='FORMATTED', STATUS='UNKNOWN', POSITION='APPEND' )
      WRITE( iUnit, '(A,I6,A)' ) 'JAXDUMP version 2: ', jd_nsteps,
     &   ' iterations'
      DO i = 1, jd_nsteps
        WRITE( iUnit, '(A,I12)' ) 'step ', jd_steps(i)
      ENDDO
      CLOSE( iUnit )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Iteration shift (see JAXDUMP.h): FORWARD_STEP sets 0 at its start and 1 right after `myIter = nIter0 + iLoop`.
C     Only sets a counter (no effect on the model, dumps on or off).
      SUBROUTINE JAXDUMP_ITERSHIFT( n )
      IMPLICIT NONE
#include "JAXDUMP.h"
      INTEGER n
      jd_ishift = n
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
      LOGICAL FUNCTION JAXDUMP_ACTIVE( myIter, myThid )
      IMPLICIT NONE
#include "JAXDUMP.h"
      INTEGER myIter, myThid
      INTEGER i
      IF ( .NOT. jd_inited ) CALL JAXDUMP_INIT( myThid )
      JAXDUMP_ACTIVE = .FALSE.
      IF ( .NOT. jd_on ) RETURN
      DO i = 1, jd_nsteps
        IF ( jd_steps(i) .EQ. myIter - jd_ishift )
     &    JAXDUMP_ACTIVE = .TRUE.
      ENDDO
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Write one tile's array (halos included) as one record.
      SUBROUTINE JAXDUMP_WRITE( stage, fname, kind, arr, nz,
     &                          bi, bj, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "JAXDUMP.h"
#ifdef ALLOW_EXCH2
# include "W2_EXCH2_SIZE.h"
# include "W2_EXCH2_TOPOLOGY.h"
#endif
      CHARACTER*(*) stage, fname, kind
      INTEGER nz, bi, bj, myIter, myThid
      _RL arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz)
      INTEGER ILNBLNK
      EXTERNAL ILNBLNK
      CHARACTER*(600) fn
      CHARACTER*32 cstage, cname
      CHARACTER*4 ckind
      INTEGER*4 hdr(4), dims(9)
      REAL*8 buf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz)
      INTEGER iUnit, tile, face, tbx, tby, i, j, k, it

#ifdef ALLOW_EXCH2
      tile = W2_myTileList(bi,bj)
      face = exch2_myFace(tile)
      tbx  = exch2_tBasex(tile)
      tby  = exch2_tBasey(tile)
#else
      face = 0
      tbx  = myXGlobalLo - 1 + (bi-1)*sNx
      tby  = myYGlobalLo - 1 + (bj-1)*sNy
      tile = 1 + tbx/sNx + (tby/sNy)*nSx*nPx
#endif
      it = myIter - jd_ishift
      WRITE(fn,'(2A,I10.10,A,I4.4,A)') jd_dir(1:ILNBLNK(jd_dir)),
     &     '/jd_', it, '_t', tile, '.bin'
      cstage = stage
      IF ( jd_pass .GT. 0 ) WRITE(cstage,'(2A,I0)')
     &   stage(1:ILNBLNK(stage)), '_p', jd_pass
      cname  = fname
      ckind  = kind
      hdr(1) = 1245990224
      hdr(2) = 2
      hdr(3) = it
      hdr(4) = jd_seq
      dims(1) = nz
      dims(2) = sNx
      dims(3) = sNy
      dims(4) = OLx
      dims(5) = OLy
      dims(6) = tile
      dims(7) = face
      dims(8) = tbx
      dims(9) = tby
      DO k = 1, nz
       DO j = 1-OLy, sNy+OLy
        DO i = 1-OLx, sNx+OLx
         buf(i,j,k) = arr(i,j,k)
        ENDDO
       ENDDO
      ENDDO
      CALL MDSFINDUNIT( iUnit, myThid )
      OPEN( iUnit, FILE=fn(1:ILNBLNK(fn)), ACCESS='STREAM',
     &      FORM='UNFORMATTED', STATUS='UNKNOWN', POSITION='APPEND' )
      WRITE( iUnit ) hdr, cstage, cname, ckind, dims, buf
      CLOSE( iUnit )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a global field (all tiles, or only tile bi0,bj0 when bi0 > 0).
      SUBROUTINE JAXDUMP_FLD( stage, fname, kind, fld, nz,
     &                        bi0, bj0, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname, kind
      INTEGER nz, bi0, bj0, myIter, myThid
      _RL fld(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz,nSx,nSy)
      INTEGER bi, bj
      DO bj = 1, nSy
       DO bi = 1, nSx
        IF ( bi0 .LE. 0 .OR. (bi .EQ. bi0 .AND. bj .EQ. bj0) )
     &    CALL JAXDUMP_WRITE( stage, fname, kind, fld(1-OLx,1-OLy,1,bi,
     &   bj),
     &                        nz, bi, bj, myIter, myThid )
       ENDDO
      ENDDO
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump the field groups named by the letters of `groups` at `stage` (all tiles if bi0 <= 0, else tile bi0,bj0):
C       d dynamics  u,v,w, etaN, etaH, dEtaHdt          t tracers   theta, salt
C       a AB histories and tendencies gU gV guNm gvNm gtNm gsNm (AB3: both levels) or guNm1 ... (AB2)
C       r r*/hFac   hFacC/W/S, recip_hFacC, rStarFacC/W/S, rStarDhCDt (NONLIN_FRSURF)
C       f forcing   surfaceForcingU/V/T/S, fu, fv, Qnet, Qsw, EmPmR, saltFlux, pLoad, phi0surf, sIceLoad
C       m mixing    rhoInSitu, IVDConvCount, hMixLayer, totPhiHyd, phiHydLow, diffKr (ALLOW_3D_DIFFKR)
C       k GGL90     GGL90TKE, GGL90viscArU/V, GGL90diffKr (ALLOW_GGL90); IDEMIX_E, IDEMIX_F_B, IDEMIX_F_S
C                   (ALLOW_GGL90_IDEMIX; pkg/ggl90/GGL90.h at 63cdc0b)     (IDEMIX new in mitjax, M3)
C       K KPP       KPPviscAz, KPPdiffKzS, KPPdiffKzT, KPPghat, KPPhbl, KPPfrac (ALLOW_KPP; pkg/kpp/KPP.h)
C       Q PP81      PPviscAr, PPdiffKr (ALLOW_PP81; pkg/pp81/PP81.h)
C       Y MY82      MYviscAr, MYdiffKr, MYhbl (ALLOW_MY82; pkg/my82/MY82.h)  (K, Q, Y new in mitjax, M3)
C       g GM/Redi   Kwx Kwy Kwz Kux Kvy Kuz Kvz GM_PsiX GM_PsiY (ALLOW_GMREDI; PsiX/Y with GM_BOLUS_ADVEC)
C       c cg2d      aW2d, aS2d, aC2d (operator), pW, pS, pC (preconditioner)
C       o SBO       SBO.h scalars (ALLOW_SBO; kind N)                       (new in mitjax)
C       q cost      fc, glofc (kind N), tile_fc, objf_hflux_tut/objf_temp_tut (per tile, kind N; ALLOW_COST)
C                                                                           (new in mitjax)
C       G geometry  grid metrics, masks, Coriolis, angles, 3-D masks and mixing parameters (JAXDUMP_GEOM)
C       V vertical  1-D vertical grid, one kind-V record per array on the first tile (JAXDUMP_VEC)
C       R r* extra  rStarFacNm1, rStarExp, rStarDh*Dt, h0Fac, etaHnm1, hFac_surf, Bo_surf (JAXDUMP_GEOM)
C       X exchange  probe of every exchange routine on index-coded and signed-zero fields (JAXDUMP_EXCH_PROBE)
C       P ptracers  pTracer_nn, gpTrNm1_nn, surfaceForcingPTr_nn for nn = 1..PTRACERS_num (ALLOW_PTRACERS;
C                   pkg/ptracers/PTRACERS_FIELDS.h at 63cdc0b)                (new in mitjax, M2)
C       I i y n     sea ice (ALLOW_SEAICE; JAXDUMP_SEAICE: static fields, state, dynamics, budget terms)
C       x EXF       time-interpolated and derived EXF fields (ALLOW_EXF; JAXDUMP_EXF)
C       h thsice    pkg/thsice state, fluxes, ocean mixed layer (ALLOW_THSICE; JAXDUMP_THSICE)
C       p plume     SaltPlumeDepth, saltPlumeFlux (ALLOW_SALT_PLUME; JAXDUMP_SPLUME)
C                   (I i y n x h p: M4, lane A session 10, re-added from the ECCO port with a re-audit on 63cdc0b)
C     Not carried over from the ECCO port: the EXF bracketing records (group e) and the bulk-formulae locals (b).
      SUBROUTINE JAXDUMP_STATE( stage, groups, bi0, bj0, myIter,
     &   myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
#include "FFIELDS.h"
#include "CG2D.h"
#ifdef ALLOW_GGL90
# include "GGL90.h"
#endif
#ifdef ALLOW_KPP
# include "KPP.h"
#endif
#ifdef ALLOW_PP81
# include "PP81.h"
#endif
#ifdef ALLOW_MY82
# include "MY82.h"
#endif
#ifdef ALLOW_GMREDI
# include "GMREDI.h"
#endif
#ifdef ALLOW_PTRACERS
# include "PTRACERS_SIZE.h"
# include "PTRACERS_FIELDS.h"
#endif
#include "JAXDUMP.h"
      CHARACTER*(*) stage, groups
      INTEGER bi0, bj0, myIter, myThid
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      INTEGER m
      CHARACTER*8 suf

      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1

      IF ( INDEX(groups,'G') .GT. 0 .OR. INDEX(groups,'V') .GT. 0
     &     .OR. INDEX(groups,'R') .GT. 0 )
     &  CALL JAXDUMP_GEOM( stage, groups, bi0, bj0, myIter, myThid )
      IF ( INDEX(groups,'X') .GT. 0 )
     &  CALL JAXDUMP_EXCH_PROBE( stage, myIter, myThid )
#ifdef ALLOW_SBO
      IF ( INDEX(groups,'o') .GT. 0 )
     &  CALL JAXDUMP_SBO( stage, myIter, myThid )
#endif
#ifdef ALLOW_COST
      IF ( INDEX(groups,'q') .GT. 0 )
     &  CALL JAXDUMP_COST( stage, myIter, myThid )
#endif
#ifdef ALLOW_SEAICE
      IF ( INDEX(groups,'I') .GT. 0 .OR. INDEX(groups,'i') .GT. 0
     &     .OR. INDEX(groups,'y') .GT. 0 .OR. INDEX(groups,'n') .GT. 0
     &     .OR. INDEX(groups,'B') .GT. 0 )
     &  CALL JAXDUMP_SEAICE( stage, groups, bi0, bj0, myIter, myThid )
#endif
#ifdef ALLOW_EXF
      IF ( INDEX(groups,'x') .GT. 0 )
     &  CALL JAXDUMP_EXF( stage, bi0, bj0, myIter, myThid )
#endif
#ifdef ALLOW_THSICE
      IF ( INDEX(groups,'h') .GT. 0 )
     &  CALL JAXDUMP_THSICE( stage, bi0, bj0, myIter, myThid )
#endif
#ifdef ALLOW_SALT_PLUME
      IF ( INDEX(groups,'p') .GT. 0 )
     &  CALL JAXDUMP_SPLUME( stage, bi0, bj0, myIter, myThid )
#endif
      IF ( INDEX(groups,'d') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'uVel','W',uVel,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'vVel','S',vVel,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'wVel','C',wVel,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'etaN','C',etaN,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'etaH','C',etaH,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dEtaHdt','C',dEtaHdt,1,bi0,bj0,myIter,
     &                  myThid)
      ENDIF
      IF ( INDEX(groups,'t') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'theta','C',theta,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'salt','C',salt,Nr,bi0,bj0,myIter,myThid)
      ENDIF
      IF ( INDEX(groups,'a') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'gU','W',gU,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'gV','S',gV,Nr,bi0,bj0,myIter,myThid)
#ifdef ALLOW_ADAMSBASHFORTH_3
       DO m = 1, 2
        WRITE(suf,'(A,I1)') '_', m
        CALL JAXDUMP_FLD(stage,'guNm'//suf(1:2),'W',guNm(1-OLx,1-OLy,1,
     &   1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'gvNm'//suf(1:2),'S',gvNm(1-OLx,1-OLy,1,
     &   1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'gtNm'//suf(1:2),'C',gtNm(1-OLx,1-OLy,1,
     &   1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'gsNm'//suf(1:2),'C',gsNm(1-OLx,1-OLy,1,
     &   1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
       ENDDO
#else
       CALL JAXDUMP_FLD(stage,'guNm1','W',guNm1,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'gvNm1','S',gvNm1,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'gtNm1','C',gtNm1,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'gsNm1','C',gsNm1,Nr,bi0,bj0,myIter,
     &   myThid)
#endif
      ENDIF
      IF ( INDEX(groups,'r') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'hFacC','C',hFacC,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'hFacW','W',hFacW,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'hFacS','S',hFacS,Nr,bi0,bj0,myIter,
     &   myThid)
       CALL JAXDUMP_FLD(stage,'recip_hFacC','C',recip_hFacC,Nr,
     &                  bi0,bj0,myIter,myThid)
#ifdef NONLIN_FRSURF
       CALL JAXDUMP_FLD(stage,'rStarFacC','C',rStarFacC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarFacW','W',rStarFacW,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarFacS','S',rStarFacS,1,bi0,bj0,
     &                  myIter,myThid)
C      rStarDhCDt (set by CALC_R_STAR, read by MONITOR): at S12_calc_rstar
C      it is the value the step's MONITOR call sees
       CALL JAXDUMP_FLD(stage,'rStarDhCDt','C',rStarDhCDt,1,bi0,bj0,
     &                  myIter,myThid)
#endif
      ENDIF
      IF ( INDEX(groups,'f') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'surfaceForcingU','W',surfaceForcingU,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'surfaceForcingV','S',surfaceForcingV,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'surfaceForcingT','C',surfaceForcingT,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'surfaceForcingS','C',surfaceForcingS,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'fu','W',fu,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'fv','S',fv,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Qnet','C',Qnet,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Qsw','C',Qsw,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'EmPmR','C',EmPmR,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'saltFlux','C',saltFlux,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pLoad','C',pLoad,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'phi0surf','C',phi0surf,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'sIceLoad','C',sIceLoad,1,bi0,bj0,
     &                  myIter,myThid)
      ENDIF
      IF ( INDEX(groups,'m') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'rhoInSitu','C',rhoInSitu,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'IVDConvCount','C',IVDConvCount,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hMixLayer','C',hMixLayer,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'totPhiHyd','C',totPhiHyd,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'phiHydLow','C',phiHydLow,1,bi0,bj0,
     &                  myIter,myThid)
#ifdef ALLOW_3D_DIFFKR
       CALL JAXDUMP_FLD(stage,'diffKr','C',diffKr,Nr,bi0,bj0,
     &                  myIter,myThid)
#endif
      ENDIF
      IF ( INDEX(groups,'c') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'aW2d','W',aW2d,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'aS2d','S',aS2d,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'aC2d','C',aC2d,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pW','W',pW,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pS','S',pS,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pC','C',pC,1,bi0,bj0,myIter,myThid)
      ENDIF
#ifdef ALLOW_GGL90
      IF ( INDEX(groups,'k') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'GGL90TKE','C',GGL90TKE,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GGL90viscArU','W',GGL90viscArU,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GGL90viscArV','S',GGL90viscArV,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GGL90diffKr','C',GGL90diffKr,Nr,
     &                  bi0,bj0,myIter,myThid)
#ifdef ALLOW_GGL90_IDEMIX
       CALL JAXDUMP_FLD(stage,'IDEMIX_E','C',IDEMIX_E,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'IDEMIX_F_B','C',IDEMIX_F_B,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'IDEMIX_F_S','C',IDEMIX_F_S,1,
     &                  bi0,bj0,myIter,myThid)
#endif
      ENDIF
#endif
#ifdef ALLOW_KPP
C     (mitjax M3) KPP fields, pkg/kpp/KPP.h at 63cdc0b
      IF ( INDEX(groups,'K') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'KPPviscAz','C',KPPviscAz,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'KPPdiffKzS','C',KPPdiffKzS,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'KPPdiffKzT','C',KPPdiffKzT,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'KPPghat','C',KPPghat,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'KPPhbl','C',KPPhbl,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'KPPfrac','C',KPPfrac,1,
     &                  bi0,bj0,myIter,myThid)
      ENDIF
#endif
#ifdef ALLOW_PP81
C     (mitjax M3) PP81 fields, pkg/pp81/PP81.h at 63cdc0b
      IF ( INDEX(groups,'Q') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'PPviscAr','C',PPviscAr,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'PPdiffKr','C',PPdiffKr,Nr,
     &                  bi0,bj0,myIter,myThid)
      ENDIF
#endif
#ifdef ALLOW_MY82
C     (mitjax M3) MY82 fields, pkg/my82/MY82.h at 63cdc0b
      IF ( INDEX(groups,'Y') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'MYviscAr','C',MYviscAr,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'MYdiffKr','C',MYdiffKr,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'MYhbl','C',MYhbl,1,
     &                  bi0,bj0,myIter,myThid)
      ENDIF
#endif
#ifdef ALLOW_GMREDI
      IF ( INDEX(groups,'g') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'Kwx','C',Kwx,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kwy','C',Kwy,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kwz','C',Kwz,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kux','W',Kux,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kvy','S',Kvy,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kuz','W',Kuz,Nr,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Kvz','S',Kvz,Nr,bi0,bj0,myIter,myThid)
#ifdef GM_BOLUS_ADVEC
       CALL JAXDUMP_FLD(stage,'GM_PsiX','W',GM_PsiX,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GM_PsiY','S',GM_PsiY,Nr,bi0,bj0,
     &                  myIter,myThid)
#endif
      ENDIF
#endif
#ifdef ALLOW_PTRACERS
C     (mitjax M2) passive tracers: every compiled tracer (PTRACERS_num,
C     not the run-time PTRACERS_numInUse); the m-th slab of each array is
C     contiguous, so it is passed as the start element
      IF ( INDEX(groups,'P') .GT. 0 ) THEN
       DO m = 1, PTRACERS_num
        WRITE(suf,'(A,I2.2)') '_', m
        CALL JAXDUMP_FLD(stage,'pTracer'//suf(1:3),'C',
     &                   pTracer(1-OLx,1-OLy,1,1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'gpTrNm1'//suf(1:3),'C',
     &                   gpTrNm1(1-OLx,1-OLy,1,1,1,m),
     &                   Nr,bi0,bj0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'surfaceForcingPTr'//suf(1:3),'C',
     &                   surfaceForcingPTr(1-OLx,1-OLy,1,1,m),
     &                   1,bi0,bj0,myIter,myThid)
       ENDDO
      ENDIF
#endif
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a routine-local array of ONE tile (e.g. sigmaX(i,j,k) inside a bi,bj loop).
      SUBROUTINE JAXDUMP_TILE( stage, fname, kind, arr, nz,
     &                         bi, bj, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname, kind
      INTEGER nz, bi, bj, myIter, myThid
      _RL arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz)
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1
      CALL JAXDUMP_WRITE( stage, fname, kind, arr, nz,
     &                    bi, bj, myIter, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a routine-local array shaped like a global field (all tiles), e.g. cg2d_b in SOLVE_FOR_PRESSURE.
      SUBROUTINE JAXDUMP_LOCAL( stage, fname, kind, arr, nz,
     &                          myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname, kind
      INTEGER nz, myIter, myThid
      _RL arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz,nSx,nSy)
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1
      CALL JAXDUMP_FLD( stage, fname, kind, arr, nz, 0, 0,
     &                  myIter, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a 2-D slice of level k of a routine-local array inside a k loop; field name <fname>_k<kkk> (nz = 1).
      SUBROUTINE JAXDUMP_TILEK( stage, fname, kind, arr, k,
     &                          bi, bj, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname, kind
      INTEGER k, bi, bj, myIter, myThid
      _RL arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      INTEGER ILNBLNK
      EXTERNAL ILNBLNK
      CHARACTER*32 kname
      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1
      WRITE(kname,'(2A,I3.3)') fname(1:ILNBLNK(fname)), '_k', k
      CALL JAXDUMP_WRITE( stage, kname, kind, arr, 1,
     &                    bi, bj, myIter, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Stage-name suffix for stages inside an iteration loop: n > 0 appends '_p<n>' to the stage name of every record
C     written until JAXDUMP_PASS(0). Only sets a counter (no effect on the model, dumps on or off).
      SUBROUTINE JAXDUMP_PASS( n )
      IMPLICIT NONE
#include "JAXDUMP.h"
      INTEGER n
      jd_pass = n
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a routine-local INTERIOR-only array arr(1:sNx,1:sNy,nz) of ONE tile (e.g. the (1:sNx,1:sNy) work arrays of
C     SEAICE_GROWTH, EXF_BULKFORMULAE); halo points of the record are written as 0.
      SUBROUTINE JAXDUMP_TILEI( stage, fname, kind, arr, nz,
     &                          bi, bj, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname, kind
      INTEGER nz, bi, bj, myIter, myThid
      _RL arr(1:sNx,1:sNy,nz)
      _RL buf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nz)
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      INTEGER i, j, k
      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1
      DO k = 1, nz
       DO j = 1-OLy, sNy+OLy
        DO i = 1-OLx, sNx+OLx
         buf(i,j,k) = 0. _d 0
        ENDDO
       ENDDO
       DO j = 1, sNy
        DO i = 1, sNx
         buf(i,j,k) = arr(i,j,k)
        ENDDO
       ENDDO
      ENDDO
      CALL JAXDUMP_WRITE( stage, fname, kind, buf, nz,
     &                    bi, bj, myIter, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a scalar (iteration count, convergence measure, ...) as a constant 2-D field on every tile (kind 'N').
      SUBROUTINE JAXDUMP_SCALAR( stage, fname, val, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname
      _RL val
      INTEGER myIter, myThid
      _RL sbuf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      INTEGER i, j, bi, bj
      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          sbuf(i,j,1,bi,bj) = val
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL JAXDUMP_FLD( stage, fname, 'N', sbuf, 1, 0, 0,
     &                  myIter, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a per-tile scalar arr(nSx,nSy) as a constant 2-D field on each tile (kind 'N').      (new in mitjax)
      SUBROUTINE JAXDUMP_TILESCAL( stage, fname, arr, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "JAXDUMP.h"
      CHARACTER*(*) stage, fname
      _RL arr(nSx,nSy)
      INTEGER myIter, myThid
      _RL sbuf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      LOGICAL JAXDUMP_ACTIVE
      EXTERNAL JAXDUMP_ACTIVE
      INTEGER i, j, bi, bj
      IF ( .NOT. JAXDUMP_ACTIVE( myIter, myThid ) ) RETURN
      jd_seq = jd_seq + 1
      DO bj = 1, nSy
       DO bi = 1, nSx
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          sbuf(i,j,1,bi,bj) = arr(bi,bj)
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL JAXDUMP_FLD( stage, fname, 'N', sbuf, 1, 0, 0,
     &                  myIter, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump a 1-D vertical array vec(n) as one record of n levels on the first tile (bi=1, bj=1), level k a constant
C     field = vec(k) (kind 'V'; mitjax/io/dump.py DumpSet.vector). Called from JAXDUMP_GEOM (already active).
C                                                                                               (new in mitjax)
      SUBROUTINE JAXDUMP_VEC( stage, fname, vec, n, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
      CHARACTER*(*) stage, fname
      INTEGER n, myIter, myThid
      _RL vec(n)
      _RL buf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,n)
      INTEGER i, j, k
      DO k = 1, n
       DO j = 1-OLy, sNy+OLy
        DO i = 1-OLx, sNx+OLx
         buf(i,j,k) = vec(k)
        ENDDO
       ENDDO
      ENDDO
      CALL JAXDUMP_WRITE( stage, fname, 'V', buf, n, 1, 1,
     &                    myIter, myThid )
      RETURN
      END

#ifdef ALLOW_SBO
C---------------------------------------------------------------------------------------------------------------------
C     SBO group o: the SBO.h scalars after SBO_CALC (pkg/sbo/SBO.h at 63cdc0b).                 (new in mitjax)
      SUBROUTINE JAXDUMP_SBO( stage, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "SBO.h"
      CHARACTER*(*) stage
      INTEGER myIter, myThid
      CALL JAXDUMP_SCALAR( stage, 'xoamc', xoamc, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'yoamc', yoamc, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'zoamc', zoamc, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'xoamp', xoamp, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'yoamp', yoamp, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'zoamp', zoamp, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'mass', mass, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'xcom', xcom, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'ycom', ycom, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'zcom', zcom, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'sboarea', sboarea, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'xoamp_fw', xoamp_fw, myIter,
     &                     myThid )
      CALL JAXDUMP_SCALAR( stage, 'yoamp_fw', yoamp_fw, myIter,
     &                     myThid )
      CALL JAXDUMP_SCALAR( stage, 'zoamp_fw', zoamp_fw, myIter,
     &                     myThid )
      CALL JAXDUMP_SCALAR( stage, 'mass_fw', mass_fw, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'xcom_fw', xcom_fw, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'ycom_fw', ycom_fw, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'zcom_fw', zcom_fw, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'mass_gc', mass_gc, myIter, myThid )
      RETURN
      END
#endif /* ALLOW_SBO */

#ifdef ALLOW_COST
C---------------------------------------------------------------------------------------------------------------------
C     Cost group q: fc, glofc (scalars), tile_fc and the tutorial terms (per tile; pkg/cost/cost.h). (new in mitjax)
      SUBROUTINE JAXDUMP_COST( stage, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "cost.h"
      CHARACTER*(*) stage
      INTEGER myIter, myThid
      CALL JAXDUMP_SCALAR( stage, 'fc', fc, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'glofc', glofc, myIter, myThid )
      CALL JAXDUMP_TILESCAL( stage, 'tile_fc', tile_fc, myIter,
     &                       myThid )
#ifdef ALLOW_COST_HFLUXM
      CALL JAXDUMP_TILESCAL( stage, 'objf_hflux_tut', objf_hflux_tut,
     &                       myIter, myThid )
      CALL JAXDUMP_TILESCAL( stage, 'objf_temp_tut', objf_temp_tut,
     &                       myIter, myThid )
#endif
      RETURN
      END
#endif /* ALLOW_COST */

C---------------------------------------------------------------------------------------------------------------------
C     Geometry (G), packed vertical grid (V) and extra r* fields (R). Called from JAXDUMP_STATE (already active).
      SUBROUTINE JAXDUMP_GEOM( stage, groups, bi0, bj0, myIter,
     &                         myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "SURFACE.h"
#ifdef ALLOW_MOM_COMMON
# include "MOM_COMMON_OPTIONS.h"
# include "MOM_VISC.h"
#endif
      CHARACTER*(*) stage, groups
      INTEGER bi0, bj0, myIter, myThid
      _RL vtmp(2*Nr+1)
      INTEGER i
C     (mitjax M1 core, Task 10 follow-up) buffer for the integer and
C     1-D-in-j grid fields of group G
      _RL gbuf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      INTEGER j, bi, bj

      IF ( INDEX(groups,'G') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'xC','C',xC,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'yC','C',yC,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'xG','Z',xG,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'yG','Z',yG,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dxC','W',dxC,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dyC','S',dyC,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dxF','C',dxF,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dyF','C',dyF,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dxG','S',dxG,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dyG','W',dyG,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dxV','Z',dxV,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'dyU','Z',dyU,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rA','C',rA,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rAw','W',rAw,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rAs','S',rAs,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rAz','Z',rAz,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dxC','W',recip_dxC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dyC','S',recip_dyC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dxF','C',recip_dxF,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dyF','C',recip_dyF,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dxG','S',recip_dxG,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dyG','W',recip_dyG,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dxV','Z',recip_dxV,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_dyU','Z',recip_dyU,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_rA','C',recip_rA,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_rAw','W',recip_rAw,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_rAs','S',recip_rAs,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_rAz','Z',recip_rAz,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'R_low','C',R_low,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rLowW','W',rLowW,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rLowS','S',rLowS,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Ro_surf','C',Ro_surf,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'rSurfW','W',rSurfW,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'rSurfS','S',rSurfS,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'recip_Rcol','C',recip_Rcol,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'maskInC','C',maskInC,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'maskInW','W',maskInW,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'maskInS','S',maskInS,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'angleCosC','C',angleCosC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'angleSinC','C',angleSinC,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'u2zonDir','W',u2zonDir,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'v2zonDir','S',v2zonDir,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'fCori','C',fCori,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'fCoriG','Z',fCoriG,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'fCoriCos','C',fCoriCos,1,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'tanPhiAtU','W',tanPhiAtU,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'tanPhiAtV','S',tanPhiAtV,1,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'maskC','C',maskC,Nr,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'maskW','W',maskW,Nr,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'maskS','S',maskS,Nr,bi0,bj0,myIter,
     &                  myThid)
       CALL JAXDUMP_FLD(stage,'recip_hFacW','W',recip_hFacW,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_hFacS','S',recip_hFacS,Nr,
     &                  bi0,bj0,myIter,myThid)
#ifdef ALLOW_3D_DIFFKR
       CALL JAXDUMP_FLD(stage,'diffKr','C',diffKr,Nr,bi0,bj0,myIter,
     &                  myThid)
#endif
#ifdef ALLOW_3D_VISCA4
       CALL JAXDUMP_FLD(stage,'viscA4Dfld','C',viscA4Dfld,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'viscA4Zfld','Z',viscA4Zfld,Nr,bi0,bj0,
     &                  myIter,myThid)
#endif
#ifdef ALLOW_3D_VISCAH
       CALL JAXDUMP_FLD(stage,'viscAhDfld','C',viscAhDfld,Nr,bi0,bj0,
     &                  myIter,myThid)
       CALL JAXDUMP_FLD(stage,'viscAhZfld','Z',viscAhZfld,Nr,bi0,bj0,
     &                  myIter,myThid)
#endif
C      mitjax M1 core (Task 10 follow-up): grid fields the grid
C      routines set that were not dumped: kSurfC/W/S, kLowC (INTEGER,
C      as exact reals), topoZ, cosFacU/V, sqCosFacU/V (1-D in j,
C      repeated along i)
       CALL JAXDUMP_FLD(stage,'topoZ','C',topoZ,1,bi0,bj0,myIter,
     &                  myThid)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           gbuf(i,j,1,bi,bj) = DFLOAT( kSurfC(i,j,bi,bj) )
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       CALL JAXDUMP_FLD(stage,'kSurfC','C',gbuf,1,bi0,bj0,
     &                  myIter,myThid)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           gbuf(i,j,1,bi,bj) = DFLOAT( kSurfW(i,j,bi,bj) )
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       CALL JAXDUMP_FLD(stage,'kSurfW','W',gbuf,1,bi0,bj0,
     &                  myIter,myThid)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           gbuf(i,j,1,bi,bj) = DFLOAT( kSurfS(i,j,bi,bj) )
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       CALL JAXDUMP_FLD(stage,'kSurfS','S',gbuf,1,bi0,bj0,
     &                  myIter,myThid)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           gbuf(i,j,1,bi,bj) = DFLOAT( kLowC(i,j,bi,bj) )
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       CALL JAXDUMP_FLD(stage,'kLowC','C',gbuf,1,bi0,bj0,
     &                  myIter,myThid)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           gbuf(i,j,1,bi,bj) = cosFacU(j,bi,bj)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       CALL JAXDUMP_FLD(stage,'cosFacU','W',gbuf,1,bi0,bj0,
     &                  myIter,myThid)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           gbuf(i,j,1,bi,bj) = cosFacV(j,bi,bj)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       CALL JAXDUMP_FLD(stage,'cosFacV','S',gbuf,1,bi0,bj0,
     &                  myIter,myThid)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           gbuf(i,j,1,bi,bj) = sqCosFacU(j,bi,bj)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       CALL JAXDUMP_FLD(stage,'sqCosFacU','W',gbuf,1,bi0,bj0,
     &                  myIter,myThid)
       DO bj = 1, nSy
        DO bi = 1, nSx
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           gbuf(i,j,1,bi,bj) = sqCosFacV(j,bi,bj)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
       CALL JAXDUMP_FLD(stage,'sqCosFacV','S',gbuf,1,bi0,bj0,
     &                  myIter,myThid)
      ENDIF

      IF ( INDEX(groups,'V') .GT. 0 ) THEN
C      one record per 1-D vertical array (JAXDUMP_VEC, kind V; mitjax: the ECCO port packed them into the rows of
C      one tile-shaped record, which needs sNx+OLx > Nr and sNy+OLy >= 15 and overran vbuf for advect_xz)
       DO i = 1, Nr
        vtmp(i) = drF(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'drF', vtmp, Nr, myIter, myThid )
       DO i = 1, Nr+1
        vtmp(i) = drC(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'drC', vtmp, Nr+1, myIter, myThid )
       DO i = 1, Nr
        vtmp(i) = rC(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'rC', vtmp, Nr, myIter, myThid )
       DO i = 1, Nr+1
        vtmp(i) = rF(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'rF', vtmp, Nr+1, myIter, myThid )
       DO i = 1, Nr
        vtmp(i) = recip_drF(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'recip_drF', vtmp, Nr, myIter, myThid )
       DO i = 1, Nr+1
        vtmp(i) = recip_drC(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'recip_drC', vtmp, Nr+1, myIter,
     &                   myThid )
       CALL JAXDUMP_VEC( stage, 'tRef', tRef, Nr, myIter, myThid )
       CALL JAXDUMP_VEC( stage, 'sRef', sRef, Nr, myIter, myThid )
       CALL JAXDUMP_VEC( stage, 'rVel2wUnit', rVel2wUnit, Nr+1, myIter,
     &                   myThid )
       CALL JAXDUMP_VEC( stage, 'wUnit2rVel', wUnit2rVel, Nr+1, myIter,
     &                   myThid )
       CALL JAXDUMP_VEC( stage, 'rhoFacC', rhoFacC, Nr, myIter, myThid )
       CALL JAXDUMP_VEC( stage, 'rhoFacF', rhoFacF, Nr+1, myIter,
     &                   myThid )
       CALL JAXDUMP_VEC( stage, 'dBdrRef', dBdrRef, Nr, myIter, myThid )
       CALL JAXDUMP_VEC( stage, 'phiRef', phiRef, 2*Nr+1, myIter,
     &                   myThid )
C      mitjax M1 core (Task 10 follow-up): deep-atmosphere factors,
C      hybrid-sigma coefficients, rkSign and gravitySign
       DO i = 1, Nr
        vtmp(i) = deepFacC(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'deepFacC', vtmp, Nr, myIter,
     &                   myThid )
       DO i = 1, Nr
        vtmp(i) = deepFac2C(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'deepFac2C', vtmp, Nr, myIter,
     &                   myThid )
       DO i = 1, Nr+1
        vtmp(i) = deepFacF(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'deepFacF', vtmp, Nr+1, myIter,
     &                   myThid )
       DO i = 1, Nr+1
        vtmp(i) = deepFac2F(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'deepFac2F', vtmp, Nr+1, myIter,
     &                   myThid )
       DO i = 1, Nr
        vtmp(i) = recip_deepFacC(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'recip_deepFacC', vtmp, Nr, myIter,
     &                   myThid )
       DO i = 1, Nr
        vtmp(i) = recip_deepFac2C(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'recip_deepFac2C', vtmp, Nr, myIter,
     &                   myThid )
       DO i = 1, Nr+1
        vtmp(i) = recip_deepFacF(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'recip_deepFacF', vtmp, Nr+1, myIter,
     &                   myThid )
       DO i = 1, Nr+1
        vtmp(i) = recip_deepFac2F(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'recip_deepFac2F', vtmp, Nr+1, myIter,
     &                   myThid )
       DO i = 1, Nr+1
        vtmp(i) = aHybSigmF(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'aHybSigmF', vtmp, Nr+1, myIter,
     &                   myThid )
       DO i = 1, Nr+1
        vtmp(i) = bHybSigmF(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'bHybSigmF', vtmp, Nr+1, myIter,
     &                   myThid )
       DO i = 1, Nr
        vtmp(i) = aHybSigmC(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'aHybSigmC', vtmp, Nr, myIter,
     &                   myThid )
       DO i = 1, Nr
        vtmp(i) = bHybSigmC(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'bHybSigmC', vtmp, Nr, myIter,
     &                   myThid )
       DO i = 1, Nr
        vtmp(i) = dAHybSigF(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'dAHybSigF', vtmp, Nr, myIter,
     &                   myThid )
       DO i = 1, Nr
        vtmp(i) = dBHybSigF(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'dBHybSigF', vtmp, Nr, myIter,
     &                   myThid )
       DO i = 1, Nr+1
        vtmp(i) = dBHybSigC(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'dBHybSigC', vtmp, Nr+1, myIter,
     &                   myThid )
       DO i = 1, Nr+1
        vtmp(i) = dAHybSigC(i)
       ENDDO
       CALL JAXDUMP_VEC( stage, 'dAHybSigC', vtmp, Nr+1, myIter,
     &                   myThid )
       vtmp(1) = rkSign
       CALL JAXDUMP_VEC( stage, 'rkSign', vtmp, 1, myIter, myThid )
       vtmp(1) = gravitySign
       CALL JAXDUMP_VEC( stage, 'gravitySign', vtmp, 1, myIter, myThid )
      ENDIF

#ifdef NONLIN_FRSURF
      IF ( INDEX(groups,'R') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'rStarFacNm1C','C',rStarFacNm1C,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarFacNm1W','W',rStarFacNm1W,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarFacNm1S','S',rStarFacNm1S,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarExpC','C',rStarExpC,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarExpW','W',rStarExpW,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarExpS','S',rStarExpS,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarDhCDt','C',rStarDhCDt,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarDhWDt','W',rStarDhWDt,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'rStarDhSDt','S',rStarDhSDt,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'pStarFacK','C',pStarFacK,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'h0FacC','C',h0FacC,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'h0FacW','W',h0FacW,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'h0FacS','S',h0FacS,Nr,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'etaHnm1','C',etaHnm1,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hFac_surfC','C',hFac_surfC,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hFac_surfW','W',hFac_surfW,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hFac_surfS','S',hFac_surfS,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Bo_surf','C',Bo_surf,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'recip_Bo','C',recip_Bo,1,
     &                  bi0,bj0,myIter,myThid)
      ENDIF
#endif
      RETURN
      END


C---------------------------------------------------------------------------------------------------------------------
C     Exchange probe (group X). Part 1, index codes: fill every point of every tile, halos included (a copy FROM a
C     halo point is visible), with an exact code, run one exchange routine, dump the result with halos:
C       code = comp*cbase + idx,  idx = ((tile-1)*(sNy+2*OLy) + (j+OLy-1))*(sNx+2*OLx) + (i+OLx-1) + 1
C     comp = 1 for u-like (and scalar) fields, 2 for v-like fields; cbase = the smallest power of 10 above
C     nTiles*(sNy+2*OLy)*(sNx+2*OLx) (dumped as the scalars xCbase, xNtiles), so any tile count and tile size decode
C     (the ECCO port's 1e6 + tile*1e4 + (j+OLy)*100 + (i+OLx) needed sNx+2*OLx < 100 and at most 99 tiles).
C     A negative value is a sign flip; a point holding its own code was not changed. Codes are never 0.
C     Part 2, signed zeros: every exchange that takes withSigns=.TRUE. once on fields of +0 (records zp*) and once
C     on fields of -0 (records zm*): the bits show whether the routine negates (+0 -> -0) or copies. Vector
C     exchanges also on the mixed pairs u=-0, v=+0 (records zu*) and u=+0, v=-0 (records zv*): exch2 forms each
C     buffer value as sa1*u + sa2*v at the source index (pkg/exch2/exch2_put_rx2.template:229-230, 317-318), so the
C     -0 of one component arrives as +0 where the other component's zero is +0; equal pairs cannot show that.
C     Arrays live in a local common (no model array is touched).
      SUBROUTINE JAXDUMP_EXCH_PROBE( stage, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#ifdef ALLOW_EXCH2
# include "W2_EXCH2_SIZE.h"
# include "W2_EXCH2_TOPOLOGY.h"
#endif
      CHARACTER*(*) stage
      INTEGER myIter, myThid
      COMMON /JAXDUMP_PROBE/ pu, pv, ps
      _RL pu(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      _RL pv(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      _RL ps(0:sNx+1,0:sNy+1,1,nSx,nSy)
      _RL cbase, ntl, zu, zv
      INTEGER m, i, j, bi, bj, iz

      CALL JAXDUMP_PROBE_BASE( cbase, ntl, myThid )
      CALL JAXDUMP_SCALAR( stage, 'xCbase', cbase, myIter, myThid )
      CALL JAXDUMP_SCALAR( stage, 'xNtiles', ntl, myIter, myThid )
      DO m = 1, 13
       CALL JAXDUMP_PROBE_FILL( pu, pv, cbase, myThid )
       IF ( m .EQ. 1 ) THEN
        CALL EXCH_XY_RL( pu, myThid )
        CALL JAXDUMP_FLD(stage,'xT','C',pu,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 2 ) THEN
        CALL EXCH_UV_XY_RL( pu, pv, .TRUE., myThid )
        CALL JAXDUMP_FLD(stage,'xUVs_u','W',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xUVs_v','S',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 3 ) THEN
        CALL EXCH_UV_XY_RL( pu, pv, .FALSE., myThid )
        CALL JAXDUMP_FLD(stage,'xUVn_u','W',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xUVn_v','S',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 4 ) THEN
        CALL EXCH_Z_3D_RL( pu, 1, myThid )
        CALL JAXDUMP_FLD(stage,'xZ','Z',pu,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 5 ) THEN
        CALL EXCH_UV_AGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xAs_u','C',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xAs_v','C',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 6 ) THEN
        CALL EXCH_UV_AGRID_3D_RL( pu, pv, .FALSE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xAn_u','C',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xAn_v','C',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 7 ) THEN
        CALL EXCH_UV_BGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xBs_u','Z',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xBs_v','Z',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 8 ) THEN
        CALL EXCH_UV_BGRID_3D_RL( pu, pv, .FALSE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xBn_u','Z',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xBn_v','Z',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 9 ) THEN
        CALL EXCH_3D_RL( pu, 1, myThid )
        CALL JAXDUMP_FLD(stage,'x3D','C',pu,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 10 ) THEN
        CALL EXCH_SM_3D_RL( pu, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xSMs','C',pu,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 11 ) THEN
        CALL EXCH_UV_DGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xDs_u','S',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xDs_v','W',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 12 ) THEN
        CALL EXCH_UV_3D_RL( pu, pv, .TRUE., 1, myThid )
        CALL JAXDUMP_FLD(stage,'xUV3s_u','W',pu,1,0,0,myIter,myThid)
        CALL JAXDUMP_FLD(stage,'xUV3s_v','S',pv,1,0,0,myIter,myThid)
       ELSEIF ( m .EQ. 13 ) THEN
C       EXCH_S3D_RL (cg2d residual / search direction): halo width 1 on
C       (0:sNx+1,0:sNy+1) arrays; dumped inside the coded full array
        DO bj = 1, nSy
         DO bi = 1, nSx
          DO j = 0, sNy+1
           DO i = 0, sNx+1
            ps(i,j,1,bi,bj) = pu(i,j,1,bi,bj)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        CALL EXCH_S3D_RL( ps, 1, myThid )
        DO bj = 1, nSy
         DO bi = 1, nSx
          DO j = 0, sNy+1
           DO i = 0, sNx+1
            pu(i,j,1,bi,bj) = ps(i,j,1,bi,bj)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        CALL JAXDUMP_FLD(stage,'xS3D','C',pu,1,0,0,myIter,myThid)
       ENDIF
      ENDDO

C     Part 2: signed zeros (-0 = SIGN(0,-1)): iz = 1: u, v +0; iz = 2: u, v -0; iz = 3: u -0, v +0;
C     iz = 4: u +0, v -0 (the mixed pairs only for the vector exchanges, m < 6)
      DO iz = 1, 4
       zu = 0. _d 0
       zv = 0. _d 0
       IF ( iz .EQ. 2 .OR. iz .EQ. 3 ) zu = SIGN( 0. _d 0, -1. _d 0 )
       IF ( iz .EQ. 2 .OR. iz .EQ. 4 ) zv = SIGN( 0. _d 0, -1. _d 0 )
       DO m = 1, 6
        IF ( iz .LE. 2 .OR. m .LT. 6 ) THEN
        DO bj = 1, nSy
         DO bi = 1, nSx
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            pu(i,j,1,bi,bj) = zu
            pv(i,j,1,bi,bj) = zv
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        ENDIF
        IF ( m .EQ. 1 ) THEN
         CALL EXCH_UV_XY_RL( pu, pv, .TRUE., myThid )
         CALL JAXDUMP_ZPAIR( stage, iz, 'UVs', 'W', 'S', pu, pv,
     &                       .TRUE., myIter, myThid )
        ELSEIF ( m .EQ. 2 ) THEN
         CALL EXCH_UV_AGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
         CALL JAXDUMP_ZPAIR( stage, iz, 'As', 'C', 'C', pu, pv,
     &                       .TRUE., myIter, myThid )
        ELSEIF ( m .EQ. 3 ) THEN
         CALL EXCH_UV_BGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
         CALL JAXDUMP_ZPAIR( stage, iz, 'Bs', 'Z', 'Z', pu, pv,
     &                       .TRUE., myIter, myThid )
        ELSEIF ( m .EQ. 4 ) THEN
         CALL EXCH_UV_DGRID_3D_RL( pu, pv, .TRUE., 1, myThid )
         CALL JAXDUMP_ZPAIR( stage, iz, 'Ds', 'S', 'W', pu, pv,
     &                       .TRUE., myIter, myThid )
        ELSEIF ( m .EQ. 5 ) THEN
         CALL EXCH_UV_3D_RL( pu, pv, .TRUE., 1, myThid )
         CALL JAXDUMP_ZPAIR( stage, iz, 'UV3s', 'W', 'S', pu, pv,
     &                       .TRUE., myIter, myThid )
        ELSEIF ( m .EQ. 6 .AND. iz .LE. 2 ) THEN
         CALL EXCH_SM_3D_RL( pu, .TRUE., 1, myThid )
         CALL JAXDUMP_ZPAIR( stage, iz, 'SMs', 'C', 'C', pu, pv,
     &                       .FALSE., myIter, myThid )
        ENDIF
       ENDDO
      ENDDO
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Dump the signed-zero probe pair: fields z<p|m><name>_u (and _v when withV), p = +0 fill, m = -0 fill.
      SUBROUTINE JAXDUMP_ZPAIR( stage, iz, name, ku, kv, pu, pv, withV,
     &                          myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
      CHARACTER*(*) stage, name, ku, kv
      INTEGER iz, myIter, myThid
      LOGICAL withV
      _RL pu(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      _RL pv(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      CHARACTER*1 pm
      INTEGER ILNBLNK
      EXTERNAL ILNBLNK
      pm = 'p'
      IF ( iz .EQ. 2 ) pm = 'm'
      IF ( iz .EQ. 3 ) pm = 'u'
      IF ( iz .EQ. 4 ) pm = 'v'
      CALL JAXDUMP_FLD( stage, 'z'//pm//name(1:ILNBLNK(name))//'_u',
     &                  ku, pu, 1, 0, 0, myIter, myThid )
      IF ( withV )
     & CALL JAXDUMP_FLD( stage, 'z'//pm//name(1:ILNBLNK(name))//'_v',
     &                   kv, pv, 1, 0, 0, myIter, myThid )
      RETURN
      END

C---------------------------------------------------------------------------------------------------------------------
C     Probe code base: cbase = smallest power of 10 > nTiles*(sNy+2*OLy)*(sNx+2*OLx); ntl = nTiles.
      SUBROUTINE JAXDUMP_PROBE_BASE( cbase, ntl, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#ifdef ALLOW_EXCH2
# include "W2_EXCH2_SIZE.h"
# include "W2_EXCH2_TOPOLOGY.h"
#endif
      _RL cbase, ntl
      INTEGER myThid
      _RL ncode
#ifdef ALLOW_EXCH2
      ntl = exch2_nTiles
#else
      ntl = nSx*nSy*nPx*nPy
#endif
      ncode = ntl*(sNy+2*OLy)*(sNx+2*OLx)
      cbase = 1. _d 0
      DO WHILE ( cbase .LE. ncode )
       cbase = cbase*10. _d 0
      ENDDO
C     2*cbase + ncode must stay an exact integer in real*8
      IF ( cbase .GT. 1. _d 15 ) STOP 'JAXDUMP: probe code overflow'
      RETURN
      END

      SUBROUTINE JAXDUMP_PROBE_FILL( pu, pv, cbase, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#ifdef ALLOW_EXCH2
# include "W2_EXCH2_SIZE.h"
# include "W2_EXCH2_TOPOLOGY.h"
#endif
      INTEGER myThid
      _RL pu(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      _RL pv(1-OLx:sNx+OLx,1-OLy:sNy+OLy,1,nSx,nSy)
      _RL cbase, idx
      INTEGER i, j, bi, bj, tile, nxp, nyp
      nxp = sNx + 2*OLx
      nyp = sNy + 2*OLy
      DO bj = 1, nSy
       DO bi = 1, nSx
#ifdef ALLOW_EXCH2
        tile = W2_myTileList(bi,bj)
#else
        tile = 1 + (myXGlobalLo-1)/sNx + (bi-1)
     &       + ((myYGlobalLo-1)/sNy + (bj-1))*nSx*nPx
#endif
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          idx = DBLE( (tile-1)*nyp + (j+OLy-1) )*DBLE( nxp )
     &        + DBLE( i+OLx-1 ) + 1. _d 0
          pu(i,j,1,bi,bj) = cbase + idx
          pv(i,j,1,bi,bj) = 2. _d 0*cbase + idx
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      RETURN
      END
C---------------------------------------------------------------------------------------------------------------------
C     (mitjax M4, lane A session 10) sea-ice groups, pkg/seaice headers at 63cdc0b (SEAICE.h, SEAICE_GRID.h);
C     re-added from the ECCO port with a re-audit (renamed: ZMAX/ZMIN -> SEAICE_zMax/zMin; TAUX/TAUY are locals of
C     SEAICE_DYNSOLVER now, dumped there as G records; new: ITD, EVP, bottom/side-drag fields). Called from
C     JAXDUMP_STATE (already active).
C       I static    HEFFM, SIMaskU/V, seaiceMaskU/V, k1/k2 metric terms
C       i state     AREA, HEFF, HSNOW, UICE, VICE, TICES (nITD), HSALT, ITD categories
C       y dynamics  ice mass, FORCEX0/Y0, PRESS0, SEAICE_zMax/zMin, viscosities, strain rates, pressure, DWATN,
C                   FORCEX/Y, uIceNm1/vIceNm1, stress divergence, EVP stresses, free drift, drag coefficients
C       n budget    d_HEFFbyNEG, d_HSNWbyNEG, saltWtrIce, frWtrIce (+ relaxation, salinity adjustment)
C       B B grid    (lane A session 11) SEAICE_BGRID_DYNAMICS only (SEAICE.h:185-211): AMASS, DAIRN, uIceB, vIceB,
C                   WINDX, WINDY, GWATX, GWATY (KGEO is INTEGER and static: not dumped)
C     Kinds: without SEAICE_CGRID, UICE/VICE sit at the South-West B-grid corner (SEAICE.h:11-18) -> kind Z; in the B
C     grid DYNSOLVER, FORCEX/Y and FORCEX0/Y0 are built from 4-point averages of UWIND/VWIND around the corner
C     (dynsolver.F) -> kind Z; AMASS, DAIRN, uIceB/vIceB, GWATX/Y likewise; WINDX/Y "over water at C points" -> C.
#ifdef ALLOW_SEAICE
      SUBROUTINE JAXDUMP_SEAICE( stage, groups, bi0, bj0, myIter,
     &                           myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "SEAICE_SIZE.h"
#include "SEAICE_PARAMS.h"
#include "SEAICE_GRID.h"
#include "SEAICE.h"
      CHARACTER*(*) stage, groups
      INTEGER bi0, bj0, myIter, myThid

      IF ( INDEX(groups,'I') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'HEFFM','C',HEFFM,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'SIMaskU','W',SIMaskU,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'SIMaskV','S',SIMaskV,1,
     &                  bi0,bj0,myIter,myThid)
#ifdef SEAICE_CGRID
       CALL JAXDUMP_FLD(stage,'seaiceMaskU','W',seaiceMaskU,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'seaiceMaskV','S',seaiceMaskV,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#if ( defined SEAICE_CGRID || defined SEAICE_BGRID_DYNAMICS )
       CALL JAXDUMP_FLD(stage,'k1AtC','C',k1AtC,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'k2AtC','C',k2AtC,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'k1AtU','W',k1AtU,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'k1AtV','S',k1AtV,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'k2AtU','W',k2AtU,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'k2AtV','S',k2AtV,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef SEAICE_CGRID
       CALL JAXDUMP_FLD(stage,'k1AtZ','Z',k1AtZ,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'k2AtZ','Z',k2AtZ,1,
     &                  bi0,bj0,myIter,myThid)
#endif
      ENDIF
      IF ( INDEX(groups,'i') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'AREA','C',AREA,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'HEFF','C',HEFF,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'HSNOW','C',HSNOW,1,
     &                  bi0,bj0,myIter,myThid)
#ifdef SEAICE_CGRID
       CALL JAXDUMP_FLD(stage,'UICE','W',UICE,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'VICE','S',VICE,1,
     &                  bi0,bj0,myIter,myThid)
#else
       CALL JAXDUMP_FLD(stage,'UICE','Z',UICE,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'VICE','Z',VICE,1,
     &                  bi0,bj0,myIter,myThid)
#endif
       CALL JAXDUMP_FLD(stage,'TICES','C',TICES,nITD,
     &                  bi0,bj0,myIter,myThid)
#ifdef SEAICE_VARIABLE_SALINITY
       CALL JAXDUMP_FLD(stage,'HSALT','C',HSALT,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef SEAICE_ITD
       CALL JAXDUMP_FLD(stage,'AREAITD','C',AREAITD,nITD,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'HEFFITD','C',HEFFITD,nITD,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'HSNOWITD','C',HSNOWITD,nITD,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'opnWtrFrac','C',opnWtrFrac,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'fw2ObyRidge','C',fw2ObyRidge,1,
     &                  bi0,bj0,myIter,myThid)
#endif
      ENDIF
      IF ( INDEX(groups,'y') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'DWATN','C',DWATN,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'uIceNm1','W',uIceNm1,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'vIceNm1','S',vIceNm1,1,
     &                  bi0,bj0,myIter,myThid)
#if ( defined SEAICE_CGRID || defined SEAICE_BGRID_DYNAMICS )
       CALL JAXDUMP_FLD(stage,'PRESS0','C',PRESS0,1,
     &                  bi0,bj0,myIter,myThid)
# ifdef SEAICE_BGRID_DYNAMICS
       CALL JAXDUMP_FLD(stage,'FORCEX0','Z',FORCEX0,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'FORCEY0','Z',FORCEY0,1,
     &                  bi0,bj0,myIter,myThid)
# else
       CALL JAXDUMP_FLD(stage,'FORCEX0','W',FORCEX0,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'FORCEY0','S',FORCEY0,1,
     &                  bi0,bj0,myIter,myThid)
# endif
       CALL JAXDUMP_FLD(stage,'SEAICE_zMax','C',SEAICE_zMax,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'SEAICE_zMin','C',SEAICE_zMin,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'ETA','C',ETA,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'etaZ','Z',etaZ,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'ZETA','C',ZETA,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'zetaZ','Z',zetaZ,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'PRESS','C',PRESS,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'tensileStrFac','C',tensileStrFac,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'e11','C',e11,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'e22','C',e22,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'e12','Z',e12,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'deltaC','C',deltaC,1,
     &                  bi0,bj0,myIter,myThid)
# ifdef SEAICE_BGRID_DYNAMICS
       CALL JAXDUMP_FLD(stage,'FORCEX','Z',FORCEX,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'FORCEY','Z',FORCEY,1,
     &                  bi0,bj0,myIter,myThid)
# else
       CALL JAXDUMP_FLD(stage,'FORCEX','W',FORCEX,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'FORCEY','S',FORCEY,1,
     &                  bi0,bj0,myIter,myThid)
# endif
#endif
#ifdef SEAICE_CGRID
       CALL JAXDUMP_FLD(stage,'seaiceMassC','C',seaiceMassC,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'seaiceMassU','W',seaiceMassU,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'seaiceMassV','S',seaiceMassV,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'stressDivergenceX','W',
     &                  stressDivergenceX,1,bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'stressDivergenceY','S',
     &                  stressDivergenceY,1,bi0,bj0,myIter,myThid)
# ifdef SEAICE_ALLOW_EVP
       CALL JAXDUMP_FLD(stage,'seaice_sigma1','C',seaice_sigma1,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'seaice_sigma2','C',seaice_sigma2,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'seaice_sigma12','Z',seaice_sigma12,1,
     &                  bi0,bj0,myIter,myThid)
# endif
# ifdef SEAICE_ALLOW_FREEDRIFT
       CALL JAXDUMP_FLD(stage,'uice_fd','W',uice_fd,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'vice_fd','S',vice_fd,1,
     &                  bi0,bj0,myIter,myThid)
# endif
# ifdef SEAICE_ALLOW_BOTTOMDRAG
       CALL JAXDUMP_FLD(stage,'CbotC','C',CbotC,1,
     &                  bi0,bj0,myIter,myThid)
# endif
# ifdef SEAICE_ALLOW_SIDEDRAG
       CALL JAXDUMP_FLD(stage,'sideDragU','W',sideDragU,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'sideDragV','S',sideDragV,1,
     &                  bi0,bj0,myIter,myThid)
# endif
#endif
      ENDIF
#ifdef SEAICE_BGRID_DYNAMICS
      IF ( INDEX(groups,'B') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'AMASS','Z',AMASS,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'DAIRN','Z',DAIRN,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'uIceB','Z',uIceB,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'vIceB','Z',vIceB,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'WINDX','C',WINDX,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'WINDY','C',WINDY,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GWATX','Z',GWATX,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'GWATY','Z',GWATY,1,
     &                  bi0,bj0,myIter,myThid)
      ENDIF
#endif
      IF ( INDEX(groups,'n') .GT. 0 ) THEN
       CALL JAXDUMP_FLD(stage,'d_HEFFbyNEG','C',d_HEFFbyNEG,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'d_HSNWbyNEG','C',d_HSNWbyNEG,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'saltWtrIce','C',saltWtrIce,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'frWtrIce','C',frWtrIce,1,
     &                  bi0,bj0,myIter,myThid)
#ifdef EXF_SEAICE_FRACTION
       CALL JAXDUMP_FLD(stage,'d_AREAbyRLX','C',d_AREAbyRLX,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'d_HEFFbyRLX','C',d_HEFFbyRLX,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef SEAICE_VARIABLE_SALINITY
       CALL JAXDUMP_FLD(stage,'saltFluxAdjust','C',saltFluxAdjust,1,
     &                  bi0,bj0,myIter,myThid)
#endif
      ENDIF
      RETURN
      END
#endif /* ALLOW_SEAICE */

#ifdef ALLOW_EXF
C---------------------------------------------------------------------------------------------------------------------
C     (mitjax M4) EXF group x, pkg/exf/EXF_FIELDS.h at 63cdc0b: the time-interpolated and derived fields (each under
C     the CPP flag that declares it). ustress/vstress: kind W/S with stressIsOnCgrid, else C (A grid).
      SUBROUTINE JAXDUMP_EXF( stage, bi0, bj0, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "EXF_PARAM.h"
#include "EXF_FIELDS.h"
      CHARACTER*(*) stage
      INTEGER bi0, bj0, myIter, myThid
      CHARACTER*1 ku, kv
      ku = 'C'
      kv = 'C'
      IF ( stressIsOnCgrid ) THEN
       ku = 'W'
       kv = 'S'
      ENDIF
       CALL JAXDUMP_FLD(stage,'ustress',ku,ustress,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'vstress',kv,vstress,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'wspeed','C',wspeed,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'uwind','C',uwind,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'vwind','C',vwind,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hflux','C',hflux,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'sflux','C',sflux,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'wStress','C',wStress,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'cw','C',cw,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'sw','C',sw,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'sh','C',sh,1,
     &                  bi0,bj0,myIter,myThid)
#ifdef ALLOW_ATM_TEMP
       CALL JAXDUMP_FLD(stage,'atemp','C',atemp,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'aqh','C',aqh,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hs','C',hs,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hl','C',hl,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'lwflux','C',lwflux,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'evap','C',evap,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'precip','C',precip,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'snowprecip','C',snowprecip,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#if defined(ALLOW_ATM_TEMP) || defined(SHORTWAVE_HEATING)
       CALL JAXDUMP_FLD(stage,'swflux','C',swflux,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef ALLOW_DOWNWARD_RADIATION
       CALL JAXDUMP_FLD(stage,'swdown','C',swdown,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'lwdown','C',lwdown,1,
     &                  bi0,bj0,myIter,myThid)
# ifdef ALLOW_ZENITHANGLE
       CALL JAXDUMP_FLD(stage,'zen_albedo','C',zen_albedo,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'zen_fsol_diurnal','C',zen_fsol_diurnal,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'zen_fsol_daily','C',zen_fsol_daily,1,
     &                  bi0,bj0,myIter,myThid)
# endif
#endif
#ifdef ATMOSPHERIC_LOADING
       CALL JAXDUMP_FLD(stage,'apressure','C',apressure,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef ALLOW_RUNOFF
       CALL JAXDUMP_FLD(stage,'runoff','C',runoff,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef ALLOW_RUNOFTEMP
       CALL JAXDUMP_FLD(stage,'runoftemp','C',runoftemp,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef ALLOW_SALTFLX
       CALL JAXDUMP_FLD(stage,'saltflx','C',saltflx,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef EXF_SEAICE_FRACTION
       CALL JAXDUMP_FLD(stage,'areamask','C',areamask,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'exf_iceFraction','C',exf_iceFraction,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef ALLOW_CLIMSST_RELAXATION
       CALL JAXDUMP_FLD(stage,'climsst','C',climsst,1,
     &                  bi0,bj0,myIter,myThid)
#endif
#ifdef ALLOW_CLIMSSS_RELAXATION
       CALL JAXDUMP_FLD(stage,'climsss','C',climsss,1,
     &                  bi0,bj0,myIter,myThid)
#endif
      RETURN
      END
#endif /* ALLOW_EXF */

#ifdef ALLOW_THSICE
C---------------------------------------------------------------------------------------------------------------------
C     (mitjax M4) pkg/thsice group h, THSICE_VARS.h at 63cdc0b: ice state, fluxes, ocean mixed layer.
      SUBROUTINE JAXDUMP_THSICE( stage, bi0, bj0, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "THSICE_PARAMS.h"
#include "THSICE_SIZE.h"
#include "THSICE_VARS.h"
      CHARACTER*(*) stage
      INTEGER bi0, bj0, myIter, myThid
       CALL JAXDUMP_FLD(stage,'iceMask','C',iceMask,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'iceHeight','C',iceHeight,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'snowHeight','C',snowHeight,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Tsrf','C',Tsrf,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Tice1','C',Tice1,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Tice2','C',Tice2,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Qice1','C',Qice1,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'Qice2','C',Qice2,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'snowAge','C',snowAge,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'sHeating','C',sHeating,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'flxCndBt','C',flxCndBt,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'siceAlb','C',siceAlb,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'icAlbNIR','C',icAlbNIR,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'icFlxSW','C',icFlxSW,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'icFlxAtm','C',icFlxAtm,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'icFrwAtm','C',icFrwAtm,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'oceFWfx','C',oceFWfx,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'oceSflx','C',oceSflx,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'oceQnet','C',oceQnet,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'hOceMxL','C',hOceMxL,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'tOceMxL','C',tOceMxL,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'sOceMxL','C',sOceMxL,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'v2ocMxL','C',v2ocMxL,1,
     &                  bi0,bj0,myIter,myThid)
      RETURN
      END
#endif /* ALLOW_THSICE */

#ifdef ALLOW_SALT_PLUME
C---------------------------------------------------------------------------------------------------------------------
C     (mitjax M4) pkg/salt_plume group p, SALT_PLUME.h at 63cdc0b.
      SUBROUTINE JAXDUMP_SPLUME( stage, bi0, bj0, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "SALT_PLUME.h"
      CHARACTER*(*) stage
      INTEGER bi0, bj0, myIter, myThid
       CALL JAXDUMP_FLD(stage,'SaltPlumeDepth','C',SaltPlumeDepth,1,
     &                  bi0,bj0,myIter,myThid)
       CALL JAXDUMP_FLD(stage,'saltPlumeFlux','C',saltPlumeFlux,1,
     &                  bi0,bj0,myIter,myThid)
      RETURN
      END
#endif /* ALLOW_SALT_PLUME */

