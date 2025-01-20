from sizes import *
from phys_const import *
from atmos_ops import set_temp_levels, layer_thickness
from gas_opacity import line_mixer, get_cia, get_ray
from clouds import cloudatlas
from setup_RT import run_RT
from numba import jit

import numpy as np
import settings

def forward(temp, logg, R2D2, gasnum, logVMR, pcover, do_clouds, cloudnum,
    cloudrad, cloudsig, cloudprof, inlinetemps, press, inwavenum,
    linelist, cia, ciatemps, clphot, othphot,
    do_cf, bfing, bff):


    nlayers = press.size
    ngas = gasnum.size
    nwave = inwavenum.size
    nlinetemps = inlinetemps.size
    nclouds = cloudnum.size
    npress = press.size
    npatches = do_clouds.size

    cloudname = np.empty((npatches, nclouds), dtype='U15')
    cl_phot_press = np.zeros((npatches, maxwave))
    oth_phot_press = np.zeros((npatches, maxwave))
    out_spec = np.empty((2, nwave))
    clphotspec = np.empty((npatches, maxwave))
    othphotspec = np.empty((npatches, maxwave))
    cf = np.empty((npress, nwave, nclouds))
    cfunc = np.zeros((npatches, maxwave, maxlayers))

    ndens = np.zeros((nlayers))

    opd_scat = np.zeros((nlayers, nwave))
    gg = np.zeros((nlayers, nwave))
    opd_CIA = np.zeros((nlayers, nwave))
    opd_ext = np.zeros((nlayers, nwave))
    opd_lines = np.zeros((nlayers, nwave))
    opd_rayl = np.zeros((nlayers, nwave))
    opd_hmbff = np.zeros((nlayers, nwave))

    fe = np.zeros((nlayers))
    fH = np.zeros((nlayers))
    fHmin = np.zeros((nlayers))
    fH2 = np.zeros((nlayers))
    fHe = np.zeros((nlayers))
    mu = np.zeros((nlayers))
    tol_VMR = np.zeros((ngas))
    tot_molmass = np.zeros((ngas))


    wavelen = 1e4 / inwavenum
    grav = 10. ** (logg) / 100.

    with open("gaslist.dat", "r") as file:
        maxgas = int(file.readline())
        gaslist = []
        masslist = []
        for igas in range(maxgas):
            line = file.readline().split()
            idum1 = int(line[0])
            gas = line[1]
            mass = float(line[2])
            gaslist.append(gas)
            masslist.append(mass)

    gasname = []
    molmass = []

    for igas in range(ngas):
        gasname.append(gaslist[gasnum[igas] - 1].strip())
        molmass.append(masslist[gasnum[igas] - 1])

    with open("cloudlist.dat", "r") as file:
        maxcloud = int(file.readline())
        cloudlist = []
        for icloud in range(maxcloud):
            line = file.readline().split()
            idum2 = int(line[0])
            cloud = line[1]
            cloudlist.append(cloud)

    for ipatch in range(npatches):
        for icloud in range(nclouds):
            if cloudnum[ipatch, icloud] > 50:
                cloudname[ipatch, icloud] = "mixto"
            else:
                cloudname[ipatch, icloud] = cloudlist[cloudnum[ipatch, icloud] - 1].strip()


    ch4index = 0

    VMRname = np.empty((ngas, nlayers), dtype=list)
    VMR = np.zeros((ngas, nlayers))
    molmass_layered = np.zeros((ngas, nlayers))

    for igas in range(ngas):
        for ilayer in range(nlayers):
            VMRname[igas, ilayer] = gasname[igas].strip()
            VMR[igas, ilayer] = 10. ** (logVMR[igas, ilayer])
            molmass_layered[igas, ilayer] = molmass[igas]

    if (VMRname[igas, 0] == "ch4"):
        ch4index = igas


    if (bfing):
        fe[0] = 10. ** bff[0, :]
        fH[0] = 10. ** bff[1, :]
        fHmin[0] = 10. ** bff[2, :]
    else:
        fe[0] = 0.
        fH[0] = 0.
        fHmin[0] = 0.

    tol_VMR = VMR[:, 0]
    tot_molmass = molmass_layered[:, 0]

    for ilayer in range(nlayers):
        allelse = (np.sum(tol_VMR) +
                   fe[ilayer] +
                   fH[ilayer] +
                   fHmin[ilayer])

        fboth = 1.0 - allelse

        fratio = 0.84

        fH2[ilayer] = fratio * fboth
        fHe[ilayer] = (1.0 - fratio) * fboth

    tot_VMR_molmass = np.sum(tol_VMR * tot_molmass)

    dz, dp = layer_thickness(grav, tot_VMR_molmass, mu, fH, fHmin, fH2, fHe, nlayers, press, temp)

    for ilayer in range(nlayers):
        ndens[ilayer] = 1.0e+5 * press[ilayer] / (K_BOLTZ * temp[ilayer])


    for ilayer in range(nlayers):
        opd_lines[ilayer, :] = line_mixer(ilayer, opd_lines, linelist, inlinetemps, dz, ngas, nwave, ndens,
                                          nlinetemps, temp, VMR)

    # Get the bff opacities
    # if bfing:
    #    get_hmbff()

    opd_rayl = get_ray(ch4index, wavelen, nwave, ndens, nlayers, opd_rayl, VMR, fH2, fHe, dz)

    opd_CIA = get_cia(cia, ciatemps, grav, ch4index, opd_CIA, temp, press, VMR, fH2, fHe, fH, dz)

    specflux,clphotspec,othphotspec,cf = run_RT(clphotspec, othphotspec, cf, clphot, othphot, do_cf, npatches, inwavenum, do_clouds, press, temp, nwave, opd_ext, opd_lines, opd_CIA, opd_rayl, opd_scat, opd_hmbff, gg, dp, pcover)

    out_spec = np.zeros((2, nwave))
    out_spec[0,:] = wavelen
    out_spec[1,:] = specflux * R2D2

    if clphot:
        for ipatch in range(npatches):
            cl_phot_press[ipatch, :nwave] = clphotspec[ipatch, :nwave]

    # Process othphot data
    if othphot:
        for ipatch in range(npatches):
            oth_phot_press[ipatch, :nwave] = othphotspec[ipatch, :nwave]

    # Process cf data
    if do_cf:
        for ipatch in range(npatches):
            for ilayer in range(nlayers):
                cfunc[ipatch, :nwave, ilayer] = cf[ipatch, :nwave, ilayer]

    return out_spec, clphotspec, oth_phot_press, cf
