"""COST_ACCUMULATE_MEAN   @63cdc0b pkg/cost/cost_accumulate_mean.F:3-65"""

from mitjax.farray import loops_kji

# eesupp/inc/EEPARAMS.h: halfRL = 0.5 _d 0
halfRL = 0.5


def cost_accumulate_mean(cost, *, cfg, theta, uVel, vVel, maskC, maskW, maskS, deltaTClock, lastinterval):
    """cost_accumulate_mean( myThid )

    C     | o accumulate mean state for cost evalualtion             |

    The k, j, i iterations are independent (each point updates its own cMean* value), so the nest is vectorised.
    deltaTClock, lastinterval: traced floats (PARAMS.h, data.cost). Returns the updated CostCommon."""
    sz = cfg.size
    deltaTfrac = deltaTClock/lastinterval                               # :29
    k, j, i = loops_kji((1, sz.Nr), (1, sz.sNy), (1, sz.sNx))           # :34-36
    cost.cMeanTheta = cost.cMeanTheta.at[i, j, k].set(                  # :37-38
        cost.cMeanTheta[i, j, k] + theta[i, j, k]*deltaTfrac)
    cost.cMeanUVel = cost.cMeanUVel.at[i, j, k].set(                    # :39-40
        cost.cMeanUVel[i, j, k] + uVel[i, j, k]*deltaTfrac)
    cost.cMeanVVel = cost.cMeanVVel.at[i, j, k].set(                    # :41-42
        cost.cMeanVVel[i, j, k] + vVel[i, j, k]*deltaTfrac)
    cost.cMeanThetaUVel = cost.cMeanThetaUVel.at[i, j, k].set(          # :44-49
        cost.cMeanThetaUVel[i, j, k]
        + halfRL*(theta[i, j, k]+theta[i-1, j, k])
        * uVel[i, j, k]
        * maskW[i, j, k]*maskC[i, j, k]
        * deltaTfrac)
    cost.cMeanThetaVVel = cost.cMeanThetaVVel.at[i, j, k].set(          # :50-55
        cost.cMeanThetaVVel[i, j, k]
        + halfRL*(theta[i, j, k]+theta[i, j-1, k])
        * vVel[i, j, k]
        * maskS[i, j, k]*maskC[i, j, k]
        * deltaTfrac)
    return cost
