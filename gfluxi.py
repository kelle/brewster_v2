import numpy as np
import settings

from dsolver import dsolver
from numba import jit


@jit(nopython=True)
def bbplk(waven, T):
    C = 299792458.0
    kb = 1.38064852e-23
    h = 6.62607004e-34

    wavelen = 1.0e-6 * (1.0e4 / waven)

    bb_value = 1.0e-6 * ((2.0 * h * C ** 2) / wavelen ** 5) / (np.exp(h * C / (wavelen * kb * T)) - 1.0)

    return bb_value


@jit(nopython=True)
def gfluxi(TEMP, TAU, W0, COSBAR, wavenum, RSF, fup, fdown):

    # mean ubar
    UBARI = 0.5

    # parameters
    nlayer = 64
    nlevel = 65
    ngaussi = 8

    # arrays
    B0 = np.zeros(nlayer)
    B1 = np.zeros(nlayer)

    ALPHA = np.zeros(nlayer)
    LAMDA = np.zeros(nlayer)
    XK1 = np.zeros(nlayer)
    XK2 = np.zeros(nlayer)
    GAMA = np.zeros(nlayer)
    CP = np.zeros(nlayer)
    CM = np.zeros(nlayer)
    CPM1 = np.zeros(nlayer)
    CMM1 = np.zeros(nlayer)
    E1 = np.zeros(nlayer)
    E2 = np.zeros(nlayer)
    E3 = np.zeros(nlayer)
    E4 = np.zeros(nlayer)
    g = np.zeros(nlayer)
    xj = np.zeros(nlayer)
    h = np.zeros(nlayer)
    xk = np.zeros(nlayer)
    alpha1 = np.zeros(nlayer)
    alpha2 = np.zeros(nlayer)
    sigma1 = np.zeros(nlayer)
    sigma2 = np.zeros(nlayer)

    fpt = np.zeros(nlevel)
    fmt = np.zeros(nlevel)
    em = np.zeros(nlevel)
    em2 = np.zeros(nlevel)
    em3 = np.zeros(nlevel)
    epp = np.zeros(nlevel)

    x = [0.0446339553, 0.1443662570,
         0.2868247571, 0.4548133152, 0.6280678354,
         0.7856915206, 0.9086763921, 0.9822200849]

    w = [0.0032951914, 0.0178429027,
         0.0454393195, 0.0791995995, 0.1060473594,
         0.1125057995, 0.0911190236, 0.0445508044]

    iflag = 0

    fup = np.zeros(nlevel)
    fdown = np.zeros(nlevel)

    for j in range(nlayer):
        ALPHA[j] = np.sqrt((1.0 - W0[j]) / (1.0 - W0[j] * COSBAR[j]))
        LAMDA[j] = ALPHA[j] * (1.0 - W0[j] * COSBAR[j]) / UBARI
        GAMA[j] = (1.0 - ALPHA[j]) / (1.0 + ALPHA[j])
        term = 0.5 / (1.0 - W0[j] * COSBAR[j])

        B1[j] = (bbplk(wavenum, TEMP[j + 1]) - bbplk(wavenum, TEMP[j])) / TAU[j]
        B0[j] = bbplk(wavenum, TEMP[j])

        if (TAU[j] < 1e-6):
            B1[j] = 0.0
            B0[j] = 0.5 * (bbplk(wavenum, TEMP[j]) + bbplk(wavenum, TEMP[j + 1]))

        CP[j] = B0[j] + B1[j] * TAU[j] + B1[j] * term
        CM[j] = B0[j] + B1[j] * TAU[j] - B1[j] * term
        CPM1[j] = B0[j] + B1[j] * term
        CMM1[j] = B0[j] - B1[j] * term

    for j in range(nlayer):
        EP = np.exp(35.0)
        if LAMDA[j] * TAU[j] < 35.0:
            EP = np.exp(LAMDA[j] * TAU[j])
        EMM = 1.0 / EP
        E1[j] = EP + GAMA[j] * EMM
        E2[j] = EP - GAMA[j] * EMM
        E3[j] = GAMA[j] * EP + EMM
        E4[j] = GAMA[j] * EP - EMM

    TAUTOP = TAU[0]
    BTOP = (1.0 - np.exp(-TAUTOP / UBARI)) * bbplk(wavenum, TEMP[0])
    BSURF = bbplk(wavenum, TEMP[nlevel - 1])
    BOTTOM = BSURF + B1[nlayer - 1] * UBARI

    XK1, XK2 = dsolver(nlayer, GAMA, CP, CM, CPM1, CMM1, E1, E2, E3, E4, BTOP, BOTTOM, RSF, XK1, XK2)

    for ng in range(ngaussi):
        ugauss = x[ng]
        for j in range(nlayer):
            if W0[j] >= 0.01:
                alphax = ((1 - W0[j]) / (1 - W0[j] * COSBAR[j])) ** 0.5

                g[j] = 2 * np.pi * W0[j] * XK1[j] * (1 + COSBAR[j] * alphax) / (1 + alphax)
                h[j] = 2 * np.pi * W0[j] * XK2[j] * (1 - COSBAR[j] * alphax) / (1 + alphax)
                xj[j] = 2 * np.pi * W0[j] * XK1[j] * (1 - COSBAR[j] * alphax) / (1 + alphax)
                xk[j] = 2 * np.pi * W0[j] * XK2[j] * (1 + COSBAR[j] * alphax) / (1 + alphax)

                alpha1[j] = 2 * np.pi * (B0[j] + B1[j] * (UBARI * W0[j] * COSBAR[j] / (1 - W0[j] * COSBAR[j])))
                alpha2[j] = 2 * np.pi * B1[j]
                sigma1[j] = 2 * np.pi * (B0[j] - B1[j] * (UBARI * W0[j] * COSBAR[j] / (1 - W0[j] * COSBAR[j])))
                sigma2[j] = alpha2[j]
            else:
                g[j] = 0.0
                h[j] = 0.0
                xj[j] = 0.0
                xk[j] = 0.0
                alpha1[j] = 2 * np.pi * B0[j]
                alpha2[j] = 2 * np.pi * B1[j]
                sigma1[j] = alpha1[j]
                sigma2[j] = alpha2[j]

        fpt[nlevel - 1] = 2.0 * np.pi * (BSURF + B1[nlayer - 1] * ugauss)
        fmt[0] = 2.0 * np.pi * (1.0 - np.exp(-TAUTOP / ugauss)) * bbplk(wavenum, TEMP[0])

        for j in range(nlayer):
            em[j] = np.exp(-LAMDA[j] * TAU[j])
            em2[j] = np.exp(-TAU[j] / ugauss)
            em3[j] = em[j] * em2[j]
            epp[j] = np.exp(35.0)
            obj = LAMDA[j] * TAU[j]
            if obj < 35.0:
                epp[j] = np.exp(obj)

            fmt[j + 1] = fmt[j] * em2[j] + xj[j] / (LAMDA[j] * ugauss + 1.0) * (epp[j] - em2[j]) + xk[j] / (
                    LAMDA[j] * ugauss - 1.0) * (em2[j] - em[j]) + sigma1[j] * (1.0 - em2[j]) + sigma2[j] * (
                                 ugauss * em2[j] + TAU[j] - ugauss)

        for j in range(nlayer - 1, -1, -1):
            fpt[j] = fpt[j + 1] * em2[j] + (g[j] / (LAMDA[j] * ugauss - 1.0)) * (epp[j] * em2[j] - 1.0) + (
                    h[j] / (LAMDA[j] * ugauss + 1.0)) * (1.0 - em3[j]) + alpha1[j] * (1.0 - em2[j]) + alpha2[
                         j] * (ugauss - (TAU[j] + ugauss) * em2[j])

        fup[0] = fup[0] + w[ng] * fpt[0]

    return fup[0]