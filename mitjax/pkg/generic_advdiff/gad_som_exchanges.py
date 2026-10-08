"""GAD_SOM_EXCHANGES: halo exchanges of the SOM moment state som_T, som_S (GAD_SOM_VARS.h) of pkg/generic_advdiff."""

from mitjax.pkg.generic_advdiff.gad_exch_som import gad_exch_som


def gad_som_exchanges(*, cfg, ex, som_T, som_S):
    """GAD_SOM_EXCHANGES(myThid)   @63cdc0b pkg/generic_advdiff/gad_som_exchanges.F:6-47

    C     | SUBROUTINE GAD_SOM_EXCHANGES
    C     | o Apply exchanges to update overlaps
    C     |   for 2nd.Order Moment fields

    Returns (som_T, som_S): the common-block state of GAD_SOM_VARS.h (som_T, som_S: tuples of nSOM FArrays
    (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, 1:Nr), all tiles; see gad_exch_som). Without GAD_ALLOW_TS_SOM_ADV the routine is
    empty (:112) and returns its inputs. The flags tempSOM_Advection, saltSOM_Advection (GAD.h, set by
    gad_init_fixed.F:118-123) are static.
    """
    if cfg.GAD_ALLOW_TS_SOM_ADV:                                    # :112-126
#--   Apply exchanges to Temp. 2nd.O.Moments:
        if cfg.tempSOM_Advection:
            som_T = gad_exch_som(som_T, cfg.Nr, ex=ex)
#--   Apply exchanges to Salin. 2nd.O.Moments:
        if cfg.saltSOM_Advection:
            som_S = gad_exch_som(som_S, cfg.Nr, ex=ex)
    return som_T, som_S
