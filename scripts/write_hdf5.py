import numpy as np
import os, sys


sys.path.insert(0,'/cosma/home/dp270/dc-zhan11/hydra-fisher/')


from mpiutils import *
from fisher_utils import get_sorted_filenames
import h5py


directory = "/cosma8/data/dp270/dc-zhan11/response_sh_gaussian_lmax90_nside64_processed/"
pattern = "XtXresponse_sh_*.npy"

operator_path_list = get_sorted_filenames(directory, pattern, get_path=True)
nfreq = len(operator_path_list)
nmodes = 8280
ind_list = np.arange(nfreq)

local_files = partition_list_mpi(operator_path_list)
local_keys = partition_list_mpi(ind_list)

file_path = os.path.join(directory, 'XtXresponse_sh.hdf5')

with h5py.File(file_path, 'w', driver='mpio', comm=world) as file:
    dsets = [file.create_dataset(str(j), shape=(nmodes, nmodes), dtype='float') for j in range(nfreq)]
    barrier()
    for k in range(len(local_files)):
        dsets[local_keys[k]][...] = np.load(local_files[k])  

barrier()

print('XtXresponse_sh.hdf5 saved.')