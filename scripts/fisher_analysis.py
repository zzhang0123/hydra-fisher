import argparse, os, sys, time

sys.path.insert(0,'/cosma/home/dp270/dc-zhan11/hydra-fisher/')

from mpiutils import *
from sky_covariance import *
from fisher_information import FisherInformation



description = "Computing the Fisher matrix ..."
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--frequency", type=str, action="store", 
                    required=True, dest="frequency",
                    help="Path to frefuency file.")

parser.add_argument("--ell", type=str, action="store", 
                    required=True, dest="ell",
                    help="Path to ell file.")

parser.add_argument("--response", type=str, action="store", 
                    required=True, dest="response",
                    help="Path to XtX operator.")

parser.add_argument("--outdir", type=str, action="store", 
                    required=True, dest="outdir",
                    help="Path to output directory.")

parser.add_argument("--beam", type=str, action="store",
                    required=False, dest="beam",
                    help="The name of the beam.")


args = parser.parse_args()



#fs = np.load('/cosma/home/dp270/dc-zhan11/hydra-fisher/sorted_freqs.npy')/1e6
fs = np.load(argparse.frequency)/1e6

#ell = np.load('/snap8/scratch/dp270/dc-zhan11/response_sh_gaussian_lmax90_nside64_processed/response_sh_ellm_0000.npy')[:,0]
ell = np.load(argparse.ell)[:,0]

#direc = '/snap8/scratch/dp270/dc-zhan11/response_sh_gaussian_lmax90_nside64_processed/'
#direc = '/snap8/scratch/dp270/dc-zhan11/response_sh_vivaldi_lmax90_nside64_processed/'
direc = argparse.response

#savedir = '/cosma8/data/dp270/dc-zhan11/fisher_matrix/'
savedir = argparse.outdir

beam_kind = argparse.beam


pattern = 'XtXresponse_sh_*.npy'

n_betas = 1

Gal_FF = Universal_SED(fs, GalacticFreeFree(), n_betas)
Gal_Sync = Universal_SED(fs, GalacticSynchrotron(), n_betas)
Ext_FF = Universal_SED(fs, ExtragalacticFreeFree(), n_betas)
Ext_point = Universal_SED(fs, ExtragalacticPointSource(), n_betas)
Background = Universal_SED(fs, ExtragalacticBackground1(), n_betas)

foregrounds = [Gal_Sync, 
               Gal_FF,
               Ext_point, 
               Ext_FF,
               Background]

n_fields = len(foregrounds)


Finfo = FisherInformation(foregrounds, fs, ell, direc, npy=True, pattern=pattern)

Fisher_matrix = Finfo.parallel_Fisher_calculation()

barrier()

if rank == 0:
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.save(savedir + 'Fisher_matrix_'+beam_kind+'.npy', Fisher_matrix)
    np.save(savedir + 'Fisher_parameter'+beam_kind+'.npy', np.array(Finfo.all_params_list))
    print('Fisher matrix saved.')

