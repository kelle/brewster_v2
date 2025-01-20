from sizes import *
#from common_arrays import *
#from define_types import *
from phys_const import *

import numpy as np

def cloudatlas(column, sizdist):
    qscat = np.zeros((nmiewave, nrad, nclouds))
    qext = np.zeros((nmiewave, nrad, nclouds))
    cos_qscat = np.zeros((nmiewave, nrad, nclouds))
    radius_in = np.zeros((nrad, nclouds))
    radius = np.zeros((nrad, nclouds))
    dr = np.zeros((nrad, nclouds))
    rup = np.zeros((nrad, nclouds))
    scat_cloud = np.zeros((nmiewave, nrad, nclouds))
    ext_cloud = np.zeros((nmiewave, nrad, nclouds))
    cqs_cloud = np.zeros((nmiewave, nrad, nclouds))
    opd_ext = np.zeros((nmiewave, nrad))
    opd_scat = np.zeros((nmiewave, nrad))
    cos_qs = np.zeros((nmiewave, nrad))

    # Define variables
    norm = 0.0
    rr = 0.0
    r2 = 0.0
    rg = 0.0
    rsig = 0.0
    pw = 0.0
    pir2ndz = 0.0
    arg1 = 0.0
    arg2 = 0.0
    bot = 0.0
    f1 = 0.0
    f2 = 0.0
    intfact = 0.0
    lintfact = 0.0
    vrat = 0.0
    rmin = 0.0
    sum_ndz_2 = 0.0
    sum_ndz_1 = 0.0
    a = 0.0
    b = 0.0
    ndz = 0.0
    drr = 0.0
    arg3 = 0.0
    argscat = 0.0
    argext = 0.0
    argcosqs = 0.0
    logcon = 0.0
    qpir2 = 0.0
    frac = 0.0
    cld1arr = np.zeros(nclouds)

    # Define integers
    icloud = 0
    imiewave = 0
    irad = 0
    ilayer = 0
    oldw1 = 0
    oldw2 = 0
    idum1 = 0
    idum2 = 0
    iwave = 0
    sizdist = 0
    loc1 = 0
    loc1a = 0
    loc1b = 0
    loc1m = 0
    loc = np.zeros(1, dtype=int)
    miefile = ""

    for icloud in range(nclouds):

        if column[0].cloud[icloud].name.strip() == 'soot':
            vrat = 1.3
            rmin = 1e-8
        else:
            vrat = 2.2
            rmin = 1e-7

        pw = 1. / 3.
        f1 = (2 * vrat / (1 + vrat)) ** pw
        f2 = (2 / (1 + vrat)) ** pw * vrat ** (pw - 1)

        for irad in range(nrad):
            radius[irad][icloud] = rmin * vrat ** (float(irad) / 3.)
            rup[irad][icloud] = f1 * radius[irad][icloud]
            dr[irad][icloud] = f2 * radius[irad][icloud]

        if sizdist > 0:
            miefile = "../Clouds/" + column[0].cloud[icloud].name.strip() + ".mieff"
        elif sizdist < 0:
            miefile = "../Clouds/" + column[0].cloud[icloud].name.strip() + ".dhs"

        with open(miefile, 'r') as f:
            idum1, idum2 = map(int, f.readline().split())
            if idum1 != nmiewave or idum2 != nrad:
                print("Problem with mie coefficients file contents wrong waves or radii", miefile)
                exit(1)

        for irad in range(nrad):
            radius_in = int(f.readline().strip())
            if abs(radius[irad][icloud] - radius_in) > 0.01 * radius[irad][icloud]:
                print("Radius grid mismatch in mie file: ", miefile)
                exit(1)
            for imiewave in range(nmiewave):
                miewavelen, qscat, qext, cos_qscat = map(float, f.readline().strip().split())

    miewaven = 1.0 / miewavelen
    loc = np.argmin(np.abs(miewavelen - 1e-4))
    loc1 = loc[0]

    scat_cloud = 0.0
    ext_cloud = 0.0
    cqs_cloud = 0.0
    opd_ext = 0.0
    opd_scat = 0.0
    cos_qs = 0.0
    sum_ndz_1 = 0.0
    sum_ndz_2 = 0.0

    cld1arr = [0.0 for i in range(nlayers)]
    for ilayer in range(nlayers):
        cld1arr[ilayer] = column[ilayer].cloud[0].dtau1

    loc = np.argmax(cld1arr)
    idum1 = loc[0]

    for ilayer in range(nlayers):
        for icloud in range(nclouds):
            if column[ilayer].cloud[icloud].dtau1 > 1e-6:
                if abs(sizdist) == 2:
                    rsig = 1. + (column[ilayer].cloud[icloud].rsig * 4)
                    rg = column[ilayer].cloud[icloud].rg * 1e-4

                    r2 = rg ** 2 * np.exp(2 * np.log(rsig) ** 2)

                    norm = 0.
                    for irad in range(nrad):
                        rr = radius[irad][icloud]
                        arg1 = dr[irad][icloud] / (np.sqrt(2. * np.pi) * rr * np.log(rsig))
                        arg2 = -(np.log(rr / rg)) ** 2 / (2 * np.log(rsig) ** 2)
                        qpir2 = np.pi * rr ** 2 * qext[loc1][irad][icloud]
                        norm += (qpir2 * arg1 * np.exp(arg2))

                    ndz = 1. / norm

                    for imiewave in range(nmiewave):
                        for irad in range(nrad):
                            rr = radius[irad][icloud]
                            arg1 = dr[irad][icloud] / (np.sqrt(2. * np.pi) * rr * np.log(rsig))
                            arg2 = -(np.log(rr / rg)) ** 2 / (2 * np.log(rsig) ** 2)
                            pir2ndz = ndz * np.pi * rr ** 2 * arg1 * np.exp(arg2)

                            scat_cloud[ilayer][imiewave][icloud] += qscat[imiewave][irad][icloud] * pir2ndz
                            ext_cloud[ilayer][imiewave][icloud] += qext[imiewave][irad][icloud] * pir2ndz
                            cqs_cloud[ilayer][imiewave][icloud] += cos_qscat[imiewave][irad][icloud] * pir2ndz

                        opd_scat[ilayer][imiewave] += scat_cloud[ilayer][imiewave][icloud]
                        opd_ext[ilayer][imiewave] += ext_cloud[ilayer][imiewave][icloud]
                        cos_qs[ilayer][imiewave] += cqs_cloud[ilayer][imiewave][icloud]

                else:
                    # Hansen distribution

                    # radii supplied in um, convert to cm
                    a = column[ilayer].cloud[icloud].rg * 1e-4
                    # b is not a length, it is dimensionless
                    b = column[ilayer].cloud[icloud].rsig

                    # first need to get ndz from the optical depth dtau at 1um
                    bot = 0
                    for irad in range(nrad):
                        rr = radius[irad][icloud]
                        drr = dr[irad][icloud]
                        arg1 = (-rr / (a * b)) + math.log(drr)
                        arg2 = ((1. - 3. * b) / b) * math.log(rr)
                        argext = math.log(qext[loc1][irad][icloud] * math.pi * rr ** 2)
                        bot += math.exp(arg1 + arg2 + argext)

                    logcon = np.log(column[ilayer].cloud[icloud].dtau1 / bot)
                    arg3 = ((((2. * b) - 1.) / b) * np.log(a * b))
                    arg2 = np.log(np.math.gamma((1. - (2. * b)) / b))
                    ndz = np.exp(logcon + arg2 - arg3)
                    arg1 = ((((2. * b) - 1.) / b) * np.log(a * b)) + np.log(ndz)
                    logcon = (arg1 - arg2)

                    for imiewave in range(nmiewave):
                        for irad in range(nrad):
                            rr = radius[irad, icloud]
                            drr = dr[irad, icloud]
                            arg1 = (-rr / (a * b)) + np.log(drr)
                            arg2 = ((1. - 3. * b) / b) * np.log(rr)
                            argscat = np.log(qscat[imiewave, irad, icloud] * np.pi * rr ** 2)
                            argext = np.log(qext[imiewave, irad, icloud] * np.pi * rr ** 2)
                            argcosqs = cos_qscat[imiewave, irad, icloud] * np.pi * rr ** 2
                            scat_cloud[ilayer, imiewave, icloud] += np.exp(logcon + arg1 + arg2 + argscat)
                            ext_cloud[ilayer, imiewave, icloud] += np.exp(logcon + arg1 + arg2 + argext)
                            cqs_cloud[ilayer, imiewave, icloud] += np.exp(logcon + arg1 + arg2) * argcosqs

                        opd_scat[ilayer, imiewave] += scat_cloud[ilayer, imiewave, icloud]
                        opd_ext[ilayer, imiewave] += ext_cloud[ilayer, imiewave, icloud]
                        cos_qs[ilayer, imiewave] += cqs_cloud[ilayer, imiewave, icloud]

            for iwave in range(nwave):
                wdiff = np.abs(miewaven - wavenum[iwave])
                oldw1 = np.argmin(wdiff)

                if miewaven[oldw1] < wavenum[iwave]:
                    oldw2 = oldw1 + 1
                else:
                    oldw2 = oldw1
                    oldw1 = oldw2 - 1

                lintfact = (wavenum[iwave] - miewaven[oldw1]) / (miewaven[oldw2] - miewaven[oldw1])

                column[ilayer].opd_ext[iwave] = ((opd_ext[ilayer, oldw2] - opd_ext[ilayer, oldw1]) * lintfact) + \
                                                opd_ext[ilayer, oldw1]

                column[ilayer].opd_scat[iwave] = ((opd_scat[ilayer, oldw2] - opd_scat[ilayer, oldw1]) * lintfact) + \
                                                 opd_scat[ilayer, oldw1]

                column[ilayer].gg[iwave] = (((cos_qs[ilayer, oldw2] - cos_qs[ilayer, oldw1]) * lintfact) + cos_qs[
                    ilayer, oldw1]) / column[ilayer].opd_scat[iwave]

                if column[ilayer].opd_scat[iwave] < 1e-50:
                    column[ilayer].opd_scat[iwave] = 0.
                    column[ilayer].gg[iwave] = 0.

                if column[ilayer].opd_ext[iwave] < 1e-50:
                    column[ilayer].opd_ext[iwave] = 0.

    return