#!/usr/bin/env python

"""This is Brewster: the golden retriever of smelly atmospheres"""
from __future__ import print_function
from __future__ import division

from builtins import str
from builtins import range
import time
import numpy as np
import scipy as sp
import pymultinest as mn
import nestkit as nestkit
import read_cia
import TPmod
import settings
import os
import gc
import sys
import pickle
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from numba import jit
import cloudnest
from sizes import *
from phys_const import *
import mpi4py
import read_cia

__author__ = "Ben Burningham"
__copyright__ = "Copyright 2015 - Ben Burningham"
__credits__ = ["Ben Burningham"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Ben Burningham"
__email__ = "burninghamster@gmail.com"
__status__ = "Development"




runname = "0359_JIT_Test"

obspec = np.loadtxt("0359.txt",dtype='d',unpack=True)

w1 = 0.9
w2 = 12.0

fwhm = 999

dist = 13.57 # Is a retrieved parameter with Gaussian priors
dist_err = 0.37

npatches = 1
nclouds = 1

do_clouds = np.zeros([npatches],dtype='i')

do_clouds[:] = 0

cloudnum = np.zeros([npatches,nclouds],dtype='i')
cloudtype =np.zeros([npatches,nclouds],dtype='i')

cloudtype[:,0] = 1

chemeq = 0

do_bff = 0

proftype = 1
pfile = "t400g562nc_m+0.5.dat"

knots = 5

logcoarsePress = np.linspace(-4.0, 2.4, knots)
logfinePress = np.arange(-4.0, 2.4, 0.1)

# forward model wants pressure in bar
#logcoarsePress = np.arange(-4.0, 3.0, 0.5)
coarsePress = pow(10,logcoarsePress)
press = pow(10,logfinePress)

xpath = "/Users/harshil/PycharmProjects/Brewster/Brewster_local/Linelists/"
xlist = 'gaslistR10K.dat'

gaslist = ['h2o','ch4','co','co2','nh3','h2s','Na','K', 'ph3']

ngas = len(gaslist)

malk = 1

make_arg_pickle = 2

outdir = "/Users/harshil/PycharmProjects/Brewster/Brewster_local/Output/"

finalout = runname+".pk1"

# use the fudge factor?
do_fudge = 1

prof = np.full(5, 100.)
if (proftype == 9):
    modP, modT = np.loadtxt(pfile, skiprows=1, usecols=(1, 2), unpack=True)
    tfit = InterpolatedUnivariateSpline(np.log10(modP), modT, k=1)
    prof = tfit(logcoarsePress)

# Now we'll get the opacity files into an array
inlinetemps, inwavenum, linelist, gasnum, nwave = nestkit.get_opacities(gaslist, w1, w2, press, xpath, xlist, malk)

# Get the cia bits
tmpcia, ciatemps = read_cia.read_cia("CIA_DS_aug_2015.dat", inwavenum, nwave)
cia = np.asarray((np.empty((4,ciatemps.size,nwave), dtype='float32')))
cia[:,:,:] = tmpcia[:,:,:nwave]
ciatemps = np.asarray(ciatemps)

# grab BFF and Chemical grids
bff_raw, ceTgrid, metscale, coscale, gases_myP = nestkit.sort_bff_and_CE(chemeq, "chem_eq_tables_P3K.pic", press,
                                                                         gaslist)

#init_all_a()

settings.init()
settings.runargs = gases_myP, chemeq, dist, dist_err, cloudtype, do_clouds, gasnum, gaslist, cloudnum, inlinetemps, coarsePress, press, inwavenum, linelist, cia, ciatemps, fwhm, obspec, proftype, do_fudge, prof, do_bff, bff_raw, ceTgrid, metscale, coscale

nlayers = press.size
nlinetemps = inlinetemps.size
npress = press.size

settings.init_number()
settings.number = nlayers, ngas, npatches, nwave, nlinetemps, nclouds, npress

# Write the arguments to a pickle if needed
if make_arg_pickle > 0:
    pickle.dump(settings.runargs, open(outdir + runname + "_runargs.pic", "wb"))
    if make_arg_pickle == 1:
        sys.exit()

# put it all together in the sampler..

n_params = nestkit.countdims(settings.runargs)

# print(n_params)

result = mn.solve(LogLikelihood=nestkit.lnlike, Prior=nestkit.priormap, n_dims=n_params, n_live_points=500,
                  outputfiles_basename=outdir + runname, verbose=True)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for col in zip(result['samples'].transpose()):
    print('%.3f +- %.3f' % (col.mean(), col.std()))

