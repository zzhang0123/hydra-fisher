import argparse, os, sys, time

sys.path.insert(0,'/cosma/home/dp270/dc-zhan11/hydra-fisher/')

from mpiutils import *

from processing import DataProcessing

from fisher_utils import get_sorted_filenames, radian_per_hour

import numpy as np


description = "Precompute ..."
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--template", type=str, action="store", 
                    required=True, dest="template",
                    help="Path to template UVData file.")

parser.add_argument("--datadir", type=str, action="store", 
                    required=True, dest="datadir",
                    help="Path to X operator directory.")

parser.add_argument("--noise", type=str, action="store", 
                    required=True, dest="noisedir",
                    help="Path to noise.")

parser.add_argument("--outdir", type=str, action="store", 
                    required=True, dest="outdir",
                    help="Path to output directory.")


args = parser.parse_args()

# Configure mpi
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nworkers = comm.Get_size()

# Set-up variables

#save_directory = "/snap8/scratch/dp270/dc-zhan11/response_sh_vivaldi_lmax90_nside64_processed/"
save_directory = args.outdir

#noise = np.load("/cosma8/data/dp270/dc-zhan11/auto_response_sh_gaussian_lmax90_nside64/auto_correlation_0000.npy")[:,:,0]
noise = np.load(args.noisedir)[:,:,0]

directory = args.datadir
#"/cosma8/data/dp270/dc-bull2/response_sh_vivaldi_lmax90_nside64_band1/"
temp_file = args.template
#"/cosma/home/dp270/dc-bull2/H4C_sum_all-bands_frf_etc_115_135MHz.shortbls.uvh5"

vis_file_list = get_sorted_filenames(directory, 'response_sh_*.npy')
ellm_file_list = get_sorted_filenames(directory, 'response_sh_ellm_*.npy')

vis_file_local_list = partition_list_mpi(vis_file_list)
ellm_file_local_list = partition_list_mpi(ellm_file_list)

assert len(vis_file_local_list)==len(ellm_file_local_list), "The number of local files is not the same."

VisResponse = DataProcessing(directory, template=temp_file)



noise = np.sqrt(noise.real**2 / (40*40 * 166000)) # 40nights, the integration time is 40s, and the frequency bandwidth is 166000 Hz.

assert noise.dtype == np.float64, "The noise array is not float64."

barrier()

for i in range(len(vis_file_local_list)):
    vis_file_local = vis_file_local_list[i]
    ellm_file_local = ellm_file_local_list[i]
    VisResponse(vis_file_local, 
                ellm_file_local, 
                save_directory,
                start_time = radian_per_hour * 4,     
                end_time = radian_per_hour * 6.25,  
                step_time = radian_per_hour * (40 / 3600),    
                reference_time = 0.10183045,  
                noise_scale = noise,
                save_XtX=True)

