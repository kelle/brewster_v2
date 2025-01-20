from sizes import *
from phys_const import *
from atmos_ops import set_temp_levels
from numba import jit
from gfluxi import gfluxi, bbplk

import numpy as np
import settings


@jit(nopython=True)
def run_RT(clphotspec, othphotspec, cf, clphot, othphot, do_cf, npatches, inwavenum, cloudy, press, temp, nwave, opd_ext, opd_lines, opd_CIA, opd_rayl, opd_scat, opd_hmbff, gg, dp, pcover):

    nlayers = press.size
    nlevel = nlayers + 1

    spectrum = np.zeros(0)
    DTAUC = np.zeros(nlayers)
    SSALB = np.zeros(nlayers)
    COSBAR = np.zeros(nlayers)
    temper = np.zeros(nlevel)
    tau1 = 0.0
    tau2 = 0.0
    p1 = 0.0
    taup_cl = 0.0
    taup_oth = 0.0
    tau = 0.0

    gflup = np.zeros(nlevel)
    fdi = np.zeros(nlevel)

    ALBEDO = 0.000000001

    temper = set_temp_levels(temper, press, temp)

    upflux = np.zeros(nwave)
    spectrum = np.zeros(nwave)
    tau_cloud = np.zeros((nlayers, nwave))
    tau_others = np.zeros((nlayers, nwave))

    for ipatch in range(npatches):
        for ilayer in range(nlayers):
            tau_cloud[ilayer, :] = opd_ext[ilayer, :]

            tau_others[ilayer, :] = opd_lines[ilayer, :] + opd_CIA[ilayer, :] + opd_rayl[ilayer, :] + opd_hmbff[ilayer, :]

            opd_ext[ilayer, :] = opd_ext[ilayer, :] + opd_lines[ilayer, :] + opd_CIA[ilayer, :] + opd_rayl[ilayer, :] + opd_hmbff[ilayer, :]

            opd_scat[ilayer, :] = opd_scat[ilayer, :] + opd_rayl[ilayer, :]

        for iwave in range(nwave):
            SSALB = np.zeros(nlayers)
            COSBAR = np.zeros(nlayers)

            for ilayer in range(nlayers):
                if cloudy != 0:
                    SSALB[ilayer] = opd_scat[ilayer, iwave] / opd_ext[ilayer, iwave]
                    COSBAR[ilayer] = gg[ilayer, iwave]
                else:
                    SSALB[ilayer] = opd_scat[ilayer, iwave] / opd_ext[ilayer, iwave]
                    COSBAR[ilayer] = 0.0

            # Set reference tau for cloud taup_cl and others taup_oth
            taup_cl = 1.0
            taup_oth = 1.0

            cldone = 0
            othdone = 0

            for ilayer in range(nlayers):

                # put optical depth into the right variable for radtran
                DTAUC[ilayer] = opd_ext[ilayer, iwave]

                # now sort out the diagnostics for the photospheres
                if clphot and not cldone:

                    # this bit calculates the pressure level where tau_cloud
                    # reaches some value set above. Activate in python code with
                    # gnostics
                    if sum(tau_cloud[:ilayer, iwave]) > taup_cl:
                        cldone = 1
                        tau2 = sum(tau_cloud[:ilayer, iwave])
                        tau1 = tau2 - tau_cloud[ilayer - 1, iwave]

                        if ilayer == nlayers:
                            p1 = np.exp(
                                (0.5) * np.log(press[ilayer - 1] * press[ilayer]))
                        else:
                            p1 = np.exp(((1.5) * np.log(press[ilayer])) -
                                        ((0.5) * np.log(press[ilayer + 1])))

                        clphotspec[ipatch, iwave] = p1 + (
                                (taup_cl - tau1) * dp[ilayer] / tau_cloud[ilayer, iwave])

                if (othphot and (not othdone)):
                    # this bit calculates the pressure level where tau_other
                    # (i.e. not clouds) reaches some value set above.
                    # Activate in python code with gnostics
                    if (sum(tau_others[:ilayer, iwave]) > taup_oth):
                        othdone = 1
                        tau2 = sum(tau_others[:ilayer, iwave])
                        tau1 = tau2 - tau_others[ilayer, iwave]

                        if (ilayer == nlayers):
                            p1 = np.exp(
                                0.5 * (np.log(press[ilayer - 1] * press[ilayer])))
                        else:
                            p1 = np.exp((1.5 * np.log(press[ilayer])) - (
                                    0.5 * np.log(press[ilayer + 1])))

                        othphotspec[ipatch, iwave] = p1 + (
                                (taup_oth - tau1) * dp[ilayer] / tau_others[ilayer, iwave])

            gflup[0] = gfluxi(temper, DTAUC, SSALB, COSBAR, inwavenum[iwave], ALBEDO, gflup, fdi)  # Call subroutine
            upflux[iwave] = gflup[0]
        spectrum = spectrum + (upflux * pcover)

    if do_cf:
        for ipatch in range(npatches):
            for iwave in range(nwave):
                tau = 0
                for ilayer in range(nlayers):
                    tau += opd_ext[ilayer, iwave]
                    cf[ipatch, :nwave, ilayer] = temp[ilayer] * opd_ext[ilayer, iwave] / np.exp(tau)

    return spectrum, clphotspec, othphotspec, cf
