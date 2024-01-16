import argparse, os, sys, time

sys.path.insert(0,'/cosma/home/dp270/dc-zhan11/hydra-fisher/')

from mpiutils import *
from sky_covariance import *
from fisher_information import FisherInformation



fs = np.load('sorted_freqs.npy')/1e6

ell = np.load('/cosma8/data/dp270/dc-zhan11/response_sh_gaussian_lmax90_nside64_processed/response_sh_ellm_0000.npy')[:,0]

direc = '/cosma8/data/dp270/dc-zhan11/response_sh_gaussian_lmax90_nside64_processed'

pattern = 'XtXresponse_sh_*.npy'

n_betas = 3

Gal_FF = Universal_SED(fs, GalacticFreeFree(), n_betas)
Gal_Sync = Universal_SED(fs, GalacticSynchrotron(), n_betas)
Ext_FF = Universal_SED(fs, ExtragalacticFreeFree(), n_betas)
Ext_point = Universal_SED(fs, ExtragalacticPointSource(), n_betas)
Background = Universal_SED(fs, ExtragalacticBackground(), n_betas)

foregrounds = [Gal_FF, 
               Gal_Sync, 
               Ext_FF,
               Ext_point,
               Background]

n_fields = len(foregrounds)



Finfo = FisherInformation(foregrounds, fs, ell, direc, pattern)

Fisher_matrix = Finfo.parallel_Fisher_calculation()

savedir = '/cosma8/data/dp270/dc-zhan11/fisher_matrix/'

barrier()

if rank == 0:
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.save(savedir + 'Fisher_matrix.npy', Fisher_matrix)
    np.save(savedir + 'Fisher_parameter_inds.npy', np.array(Finfo.all_params_list))
    print('Fisher matrix saved.')

