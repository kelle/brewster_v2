from numba import jit

import numpy as np


@jit(nopython=True)
def dtridgl(L, AF, BF, CF, DF):
    NMAX = 301

    AS, DS, XK = np.zeros(L), np.zeros(L), np.zeros(L)

    AS[-1] = AF[-1] / BF[-1]
    DS[-1] = DF[-1] / BF[-1]

    for i in range(L-2, -1, -1):
        X = 1.0 / (BF[i] - CF[i] * AS[i+1])
        AS[i] = AF[i] * X
        DS[i] = (DF[i] - CF[i] * DS[i+1]) * X

    XK[0] = DS[0]
    for i in range(1, L):
        XK[i] = DS[i] - AS[i] * XK[i-1]

    return XK

@jit(nopython=True)
def dsolver(NL, GAMA, CP, CM, CPM1, CMM1, E1, E2, E3, E4, BTOP, BSURF, RSF, XK1, XK2):

    L = 2 * NL

    # Declare variables
    AF = np.zeros(L)
    BF = np.zeros(L)
    CF = np.zeros(L)
    DF = np.zeros(L)
    XK = np.zeros(L)

    # Solve for AF, BF, CF, and DF
    AF[0] = 0.0
    BF[0] = GAMA[0] + 1.0
    CF[0] = GAMA[0] - 1.0
    DF[0] = BTOP - CMM1[0]

    AF[1::2][:-1] = (E1[:-1] + E3[:-1]) * (GAMA[1:] - 1.0)
    BF[1::2][:-1] = (E2[:-1] + E4[:-1]) * (GAMA[1:] - 1.0)
    CF[1::2][:-1] = 2.0 * (1.0 - GAMA[1:] ** 2)
    DF[1::2][:-1] = (GAMA[1:] - 1.0) * (CPM1[1:] - CP[:-1]) + (1.0 - GAMA[1:]) * (CM[:-1] - CMM1[1:])


    AF[::2][1:] = 2.0 * (1.0 - GAMA[:-1] ** 2)
    BF[::2][1:] = (E1[:-1] - E3[:-1]) * (GAMA[1:]+1.0)
    CF[::2][1:] = (E1[:-1] + E3[:-1]) * (GAMA[1:] - 1.0)
    DF[::2][1:] = E3[:-1] * (CPM1[1:] - CP[:-1]) + E1[:-1] * (CM[:-1] - CMM1[1:])

    AF[-1] = E1[-1] - RSF * E3[-1]
    BF[-1] = E2[-1] - RSF * E4[-1]
    CF[-1] = 0.0
    DF[-1] = BSURF - CP[-1] + RSF * CM[-1]

    # Call the DTRIDGL function to solve the equations
    XK = dtridgl(L, AF, BF, CF, DF)

    # Unmix the coefficients
    for i in range(0, NL, 2):
        XK1[i] = XK[2 * i] + XK[2 * i + 1]
        XK2[i] = XK[2 * i] - XK[2 * i + 1]

    return XK1, XK2
