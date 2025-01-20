from sizes import *
from phys_const import *
from numba import jit

import numpy as np
import settings


@jit(nopython=True)
def layer_thickness(grav, tot_VMR_molmass, mu, fH, fHmin, fH2, fHe, nlayers, press, temp):
    R_GAS = 8.3144621
    AVOGADRO = 6.02e23
    amu = 1.6605402e-27
    K_BOLTZ = R_GAS / AVOGADRO

    XH2 = 2.01588
    XHe = 4.002602
    XH = 1.008

    dp = np.zeros(nlayers)
    dz = np.zeros(nlayers)

    for ilayer in range(nlayers):
        mu[ilayer] = fH2[ilayer] * XH2 + fHe[ilayer] * XHe + fH[ilayer] * XH + fHmin[
            ilayer] * XH + tot_VMR_molmass

        R_spec = K_BOLTZ / (mu[ilayer] * amu)

        if ilayer == nlayers - 1:
            p1 = np.exp((0.5) * (np.log(press[ilayer - 1] * press[ilayer])))
            p2 = (press[ilayer]**2) / p1
            dp[ilayer] = p2 - p1
            dz[ilayer] = abs((R_spec * temp[ilayer] / grav) * np.log(p1 / p2))
        else:
            p1 = np.exp((1.5) * np.log(press[ilayer]) - ((0.5) * np.log(press[ilayer + 1])))
            p2 = np.exp((0.5) * np.log(press[ilayer] * press[ilayer + 1]))

            dp[ilayer] = p2 - p1

            dz[ilayer] = abs((R_spec * temp[ilayer] / grav) * np.log(p1 / p2))

    return dz, dp

@jit(nopython=True)
def set_temp_levels(leveltemp, press, temp):

    nlayers = press.size

    logP = np.log10(press)

    for ilayer in range(nlayers):

        if ilayer == 0:
            p1 = np.exp(((1.5) * np.log(press[ilayer])) - ((0.5) * np.log(press[ilayer+1])))
            p2 = np.exp((0.5) * (np.log(press[ilayer] * press[ilayer+1])))

            leveltemp[ilayer - 1] = temp[ilayer] + (((temp[ilayer+1] - temp[ilayer]) / (logP[ilayer+1] - logP[ilayer])) * (np.log10(p1) - logP[ilayer]))

            leveltemp[ilayer] = temp[ilayer] + (((temp[ilayer+1] - temp[ilayer]) / (logP[ilayer+1] - logP[ilayer])) * (np.log10(p2) - logP[ilayer]))

        elif ilayer == nlayers-1:
            p1 = np.exp((0.5) * (np.log(press[ilayer-1] * press[ilayer])))
            p2 = (press[ilayer] ** 2) / p1

            leveltemp[ilayer] = temp[ilayer] + (((temp[ilayer] - temp[ilayer-1]) / (logP[ilayer] - logP[ilayer-1])) * (np.log10(p2) - logP[ilayer]))


        else:
            p2 = np.exp((0.5) * (np.log(press[ilayer] * press[ilayer+1])))

            leveltemp[ilayer] = temp[ilayer] + (((temp[ilayer+1] - temp[ilayer]) / (logP[ilayer+1] - logP[ilayer])) * (np.log10(p2) - logP[ilayer]))

    temper = np.roll(leveltemp, 1)
    temper[0] = leveltemp[-1]

    return temper
