import numpy as np

def prism_non_uniform(obspec, modspec, resel):
    nobs = obspec.shape[1]
    nmod = modspec.shape[1]

    delta1 = np.zeros(nobs)
    delta2 = np.zeros(nobs)
    delta = np.zeros(nobs)
    gauss = np.zeros(nmod)
    Fratio_int = np.zeros(nobs)

    Fmod = modspec[1,:]
    wlmod = modspec[0,:]
    wlobs = obspec[0,:]

    delta1 = np.abs(wlobs - np.roll(wlobs, 1))
    delta2 = np.abs(wlobs - np.roll(wlobs, -1))
    delta = 0.5 * (delta1 + delta2)

    delta[0] = delta[1]
    delta[-1] = delta[-2]

    for i in range(nobs):
        sigma = delta[i] * resel / 2.355
        gauss = np.exp(-(wlmod - wlobs[i])**2 / (2*sigma**2))
        gauss /= np.sum(gauss)
        Fratio_int[i] = np.sum(gauss * Fmod)

    return Fratio_int

def conv_uniform_FWHM(obspec, modspec, fwhm):
    nobs = obspec.shape[1]
    nmod = modspec.shape[1]

    gauss = np.zeros(nmod)
    Fratio_int = np.zeros(nobs)

    wlobs = obspec[0,:]
    Fmod = modspec[1,:]
    wlmod = modspec[0,:]


    for i in range(nobs):
        sigma = fwhm / 2.355
        gauss = np.exp(-(wlmod - wlobs[i])**2 / (2 * sigma**2))
        gauss = gauss / np.sum(gauss)
        Fratio_int[i] = np.sum(gauss * Fmod)

    return Fratio_int

def conv_uniform_R(obspec, modspec, R):
    nobs = obspec.shape[1]
    nmod = modspec.shape[1]

    gauss = np.zeros(nmod)
    Fratio_int = np.zeros(nobs)

    wlobs = obspec[0,:]
    wlmod = modspec[0,:]
    Fmod = modspec[1,:]

    for i in range(nobs):
        # sigma is FWHM / 2.355
        sigma = (wlobs[i] / R) / 2.355
        gauss = np.exp(-(wlmod-wlobs[i])**2/(2*sigma**2))
        gauss = gauss / np.sum(gauss)
        Fratio_int[i] = np.sum(gauss*Fmod)

    return Fratio_int

def conv_non_uniform_R(obspec, modspec, R):
    nobs = obspec.shape[1]
    nmod = modspec.shape[1]

    gauss = np.zeros(nmod)

    wlobs = obspec[0,:]
    Fmod = modspec[1,:]
    wlmod = modspec[0,:]

    Fratio_int = np.zeros(nobs)

    for i in range(nobs):
        sigma = (wlobs[i] / R[i]) / 2.355
        gauss = np.exp(-(wlmod - wlobs[i])**2 / (2*sigma**2))
        gauss /= np.sum(gauss)
        Fratio_int[i] = np.sum(gauss * Fmod)

    return Fratio_int
