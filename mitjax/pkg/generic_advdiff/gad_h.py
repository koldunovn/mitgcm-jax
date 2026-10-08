"""GAD.h of pkg/generic_advdiff: its PARAMETERs as module constants (@63cdc0b pkg/generic_advdiff/GAD.h).

C !DESCRIPTION:
C Contains enumerated constants for distinguishing between different
C advection schemes and tracers.

Only the PARAMETER statements of GAD.h live here (the common blocks /GAD_PARM_C/, /GAD_PARM_I/, /GAD_PARM_L/ and the
rest of GAD.h are run-time state, set by GAD_INIT_FIXED / the namelists). Each constant cites its line and value.
"""

# GAD.h:19-21   C ENUM_UPWIND_1RST :: 1rst Order Upwind;            PARAMETER(ENUM_UPWIND_1RST=1)
ENUM_UPWIND_1RST = 1
# GAD.h:23-25   C ENUM_CENTERED_2ND :: Centered 2nd order;          PARAMETER(ENUM_CENTERED_2ND=2)
ENUM_CENTERED_2ND = 2
# GAD.h:27-29   C ENUM_UPWIND_3RD :: 3rd order upwind;              PARAMETER(ENUM_UPWIND_3RD=3)
ENUM_UPWIND_3RD = 3
# GAD.h:31-33   C ENUM_CENTERED_4TH :: Centered 4th order;          PARAMETER(ENUM_CENTERED_4TH=4)
ENUM_CENTERED_4TH = 4
# GAD.h:35-37   C ENUM_DST2 :: 2nd Order Direct Space and Time (= Lax-Wendroff); PARAMETER(ENUM_DST2=20)
ENUM_DST2 = 20
# GAD.h:39-41   C ENUM_FLUX_LIMIT :: Non-linear flux limiter;       PARAMETER(ENUM_FLUX_LIMIT=77)
ENUM_FLUX_LIMIT = 77
# GAD.h:43-45   C ENUM_DST3 :: 3rd Order Direct Space and Time;     PARAMETER(ENUM_DST3=30)
ENUM_DST3 = 30
# GAD.h:47-49   C ENUM_DST3_FLUX_LIMIT :: 3-DST flux limited;      PARAMETER(ENUM_DST3_FLUX_LIMIT=33)
ENUM_DST3_FLUX_LIMIT = 33
# GAD.h:51-53   C ENUM_OS7MP :: 7th Order One Step method with Monotonicity Preserving Limiter; PARAMETER(ENUM_OS7MP=7)
ENUM_OS7MP = 7
# GAD.h:55-57   C ENUM_SOM_PRATHER :: 2nd Order-Moment Advection Scheme, Prather, 1986; PARAMETER(ENUM_SOM_PRATHER=80)
ENUM_SOM_PRATHER = 80
# GAD.h:59-61   C ENUM_SOM_LIMITER :: 2nd Order-Moment Advection Scheme, Prather Limiter; PARAMETER(ENUM_SOM_LIMITER=81)
ENUM_SOM_LIMITER = 81
# GAD.h:63-65   C ENUM_PPM_NULL :: piecewise parabolic method with "null" limiter; PARAMETER(ENUM_PPM_NULL_LIMIT=40)
ENUM_PPM_NULL_LIMIT = 40
# GAD.h:67-69   C ENUM_PPM_MONO :: piecewise parabolic method with "mono" limiter; PARAMETER(ENUM_PPM_MONO_LIMIT=41)
ENUM_PPM_MONO_LIMIT = 41
# GAD.h:71-73   C ENUM_PPM_WENO :: piecewise parabolic method with "weno" limiter; PARAMETER(ENUM_PPM_WENO_LIMIT=42)
ENUM_PPM_WENO_LIMIT = 42
# GAD.h:75-77   C ENUM_PQM_NULL :: piecewise quartic method with "null" limiter; PARAMETER(ENUM_PQM_NULL_LIMIT=50)
ENUM_PQM_NULL_LIMIT = 50
# GAD.h:79-81   C ENUM_PQM_MONO :: piecewise quartic method with "mono" limiter; PARAMETER(ENUM_PQM_MONO_LIMIT=51)
ENUM_PQM_MONO_LIMIT = 51
# GAD.h:83-85   C ENUM_PQM_WENO :: piecewise quartic method with "weno" limiter; PARAMETER(ENUM_PQM_WENO_LIMIT=52)
ENUM_PQM_WENO_LIMIT = 52

# GAD.h:87-89   C GAD_Scheme_MaxNum :: maximum possible number for an advection scheme; PARAMETER( GAD_Scheme_MaxNum = 100 )
GAD_Scheme_MaxNum = 100
# GAD.h:91-93   C nSOM :: number of 1rst & 2nd Order-Moments: 1+1 (1D), 2+3 (2D), 3+6 (3D); PARAMETER( nSOM = 3+6 )
nSOM = 3 + 6

# GAD.h:95-97   C oneSixth :: Third/fourth order interpolation factor; _RL oneSixth; PARAMETER(oneSixth=1.D0/6.D0)
#               (a double-precision constant expression: the IEEE double quotient 1/6, as Python computes it)
oneSixth = 1.0 / 6.0

# GAD.h:99-109  loop range for computing vertical advection tendency; the live PARAMETER is :108-109
#               (the 1-OLx:sNx+OLx variant at :103-104 is commented out): iMinAdvR = 1, iMaxAdvR = sNx,
#               jMinAdvR = 1, jMaxAdvR = sNy. sNx, sNy come from the experiment's SIZE.h, so these are functions.
iMinAdvR = 1
jMinAdvR = 1


def iMaxAdvR(sNx):
    """GAD.h:108   PARAMETER ( iMinAdvR = 1 , iMaxAdvR = sNx )"""
    return sNx


def jMaxAdvR(sNy):
    """GAD.h:109   PARAMETER ( jMinAdvR = 1 , jMaxAdvR = sNy )"""
    return sNy


# GAD.h:116-118 C GAD_TEMPERATURE :: temperature;                  PARAMETER(GAD_TEMPERATURE=1)
GAD_TEMPERATURE = 1
# GAD.h:119-121 C GAD_SALINITY :: salinity;                        PARAMETER(GAD_SALINITY=2)
GAD_SALINITY = 2
# GAD.h:122-124 C GAD_TR1 :: passive tracer 1;                     PARAMETER(GAD_TR1=3)
GAD_TR1 = 3
