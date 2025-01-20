from atmos_ops import *
from sizes import *
from numba import jit

import numpy as np
import settings


@jit(nopython=True)
def line_mixer(ilayer, opd_lines, linelist, linetemps, dz, ngas, nwave, ndens, nlinetemps, temp, VMR):

    logkap1 = np.zeros((ngas, nwave))
    logkap2 = np.zeros((ngas, nwave))
    logintkappa = np.zeros((ngas, nwave))
    tdiff = np.zeros(nlinetemps)

    for i in range(nlinetemps):
        tdiff[i] = np.abs(linetemps[i] - temp[ilayer])

    Tlay1 = np.argmin(tdiff)

    if linetemps[Tlay1] < temp[ilayer]:
        Tlay2 = Tlay1 + 1
    else:
        Tlay2 = Tlay1 - 1

    if temp[ilayer] < linetemps[0]:
        Tlay1 = 0
        Tlay2 = 1
    elif temp[ilayer] > linetemps[nlinetemps - 1]:
        Tlay1 = nlinetemps - 2
        Tlay2 = nlinetemps - 1

    if Tlay1 > Tlay2:
        torder = 1
        intfact = (np.log10(temp[ilayer]) - np.log10(linetemps[Tlay2])) / (
                    np.log10(linetemps[Tlay1]) - np.log10(linetemps[Tlay2]))
    else:
        torder = 2
        intfact = (np.log10(temp[ilayer]) - np.log10(linetemps[Tlay1])) / (
                    np.log10(linetemps[Tlay2]) - np.log10(linetemps[Tlay1]))

    if temp[ilayer] > linetemps[nlinetemps - 1]:
        intfact = (np.log10(temp[ilayer]) - np.log10(linetemps[Tlay2])) / (
                    np.log10(linetemps[Tlay2]) - np.log10(linetemps[Tlay1]))

    logkap1 = linelist[:, ilayer, Tlay1, :]
    logkap2 = linelist[:, ilayer, Tlay2, :]

    if torder == 1:
        logintkappa = ((logkap1 - logkap2) * intfact) + logkap2
    else: #torder == 2:
        logintkappa = ((logkap2 - logkap1) * intfact) + logkap1
    #else:
    #    print("something wrong with interpolate order")

    if temp[ilayer] > linetemps[nlinetemps - 1]:
        logintkappa = ((logkap2 - logkap1) * intfact) + logkap2

    for igas in range(ngas):
        opd_lines[ilayer, :] = opd_lines[ilayer, :]+(VMR[igas, ilayer] * ndens[ilayer] * dz[ilayer] * 0.0001) * (10 ** logintkappa[igas, :])

    return opd_lines[ilayer, :]


@jit(nopython=True)
def get_ray(ch4index, wavelen, nwave, ndens, nlayers, opd_rayl, VMR, fH2, fHe, dz):

    XN0 = 2.687e19
    cfray = 32.0 * np.pi ** 3.0 * 1.0e21 / (3.0 * XN0)

    dpol = np.array([1.022, 1.0, 1.0])
    gnu = np.array([[1.355e-4, 3.469e-5, 4.318e-4], [1.235e-6, 8.139e-8, 3.408e-6]])

    tec = np.zeros(nwave)
    wa = wavelen

    if ch4index != 0:
        ng = 3
        gasss = np.zeros(ng)
    else:
        ng = 2

    for ilayer in range(nlayers):
        gasss = [fH2[ilayer], fHe[ilayer], VMR[ch4index, ilayer]]
        cold = ndens[ilayer] * dz[ilayer] * 1.0e-4
        taur = np.zeros(nwave)

        for nn in range(ng):
            tec = cfray * (dpol[nn] / wa ** 4) * (gnu[(0, nn)] + gnu[(1, nn)] / wa ** 2) ** 2
            taur = taur+ cold * gasss[nn] * tec * 1.0e-5 / XN0

        opd_rayl[ilayer, :] = taur

    return opd_rayl


@jit(nopython=True)
def get_cia(cia, ciatemp, grav, ch4index, opd_cia, temp, press, VMR, fH2, fHe, fH, dz):

    nlayers = press.size

    ph2h2 = cia[0,:,:]
    ph2he = cia[1,:,:]
    ph2h = cia[2,:,:]
    ph2ch4 = cia[3,:,:]

    for ilayer in range(nlayers):
        tdiff = np.abs(ciatemp - temp[ilayer])
        tcia1 = np.argmin(tdiff)

        if ciatemp[tcia1] < temp[ilayer]:
            tcia2 = tcia1 + 1
        else:
            tcia2 = tcia1
            tcia1 = tcia2 - 1

        if temp[ilayer] < ciatemp[0]:
            tcia1 = 0
            tcia2 = 1
        elif temp[ilayer] > ciatemp[nciatemps - 1]:
            tcia1 = nciatemps - 2
            tcia2 = nciatemps - 1

        if tcia1 == 0:
            intfact = (np.log10(temp[ilayer]) - np.log10(ciatemp[0])) / (
                        np.log10(ciatemp[1]) - np.log10(ciatemp[0]))

            ciaH2H2 = 10 ** (((ph2h2[1, :] - ph2h2[0, :]) * intfact) + ph2h2[0, :])
            ciaH2He = 10 ** (((ph2he[1, :] - ph2he[0, :]) * intfact) + ph2he[0, :])
            ciaH2H = 10 ** (((ph2h[1, :] - ph2h[0, :]) * intfact) + ph2h[0, :])
            ciaH2CH4 = 10 ** (((ph2ch4[1, :] - ph2ch4[0, :]) * intfact) + ph2ch4[0, :])

        else:
            intfact = (np.log10(temp[ilayer]) - np.log10(ciatemp[tcia1])) / (
                        np.log10(ciatemp[tcia2]) - np.log10(ciatemp[tcia1]))

            ciaH2H2 = 10.0 ** (((ph2h2[tcia2, :] - ph2h2[tcia1, :]) * intfact) + ph2h2[tcia1, :])
            ciaH2He = 10.0 ** (((ph2he[tcia2, :] - ph2he[tcia1, :]) * intfact) + ph2he[tcia1, :])
            ciaH2H = 10.0 ** (((ph2h[tcia2, :] - ph2h[tcia1, :]) * intfact) + ph2h[tcia1, :])
            ciaH2CH4 = 10.0 ** (((ph2ch4[tcia2, :] - ph2ch4[tcia1, :]) * intfact) + ph2ch4[tcia1, :])

        n_amg = (press[ilayer] / 1.01325) * (273.15 / temp[ilayer])

        if ch4index != 0:
            opd_cia[ilayer, :] = (n_amg ** 2 * fH2[ilayer] * dz[ilayer] * 100) * (
                    (fH2[ilayer] * ciaH2H2) +
                    (fHe[ilayer] * ciaH2He) +
                    (fH[ilayer] * ciaH2H) +
                    (VMR[ch4index, ilayer] * ciaH2CH4))
        else:
           opd_cia[ilayer, :] = (n_amg ** 2 * fH2[ilayer] * dz[ilayer] * 100) * (
                    (fH2[ilayer] * ciaH2H2) +
                    (fH[ilayer] * ciaH2H) +
                    (fHe[ilayer] * ciaH2He))

    return opd_cia