import argparse, os, sys, time

sys.path.insert(0,'/cosma/home/dp270/dc-zhan11/hydra-fisher/')

from mpiutils import *

from processing import DataProcessing

from fisher_utils import get_sorted_filenames, radian_per_hour

import numpy as np

directory = "/cosma8/data/dp270/dc-bull2/response_sh_gaussian_lmax90_nside64/"

vis_file_list = get_sorted_filenames(directory, 'response_sh_*.npy')
ellm_file_list = get_sorted_filenames(directory, 'response_sh_ellm_*.npy')

vis_file_local_list = partition_list_mpi(vis_file_list)
ellm_file_local_list = partition_list_mpi(ellm_file_list)

assert len(vis_file_local_list)==len(ellm_file_local_list), "The number of local files is not the same."

VisResponse = DataProcessing(directory)

save_directory = "/cosma8/data/dp270/dc-zhan11/response_sh_gaussian_lmax90_nside64_processed/"

noise = np.load("/cosma8/data/dp270/dc-zhan11/auto_response_sh_polybeam_lmax90_nside64/auto_correlation_0000.npy")[:,:,0]

noise = np.sqrt(noise.real**2 / (40 * 166000))

assert noise.dtype == np.float64, "The noise array is not float64."

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

