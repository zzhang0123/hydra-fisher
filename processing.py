import numpy as np
import fisher_utils as fu
import os

# Data preprocessing

class DataProcessing():
    def __init__(self, directory, cross_only=True, one_way_baseline=True, minimum_ell=1):
        """
        Input:
        directory: directory where the data files are stored
        cross_only: if True, only cross-correlations are considered
        one_way_baseline: if True, only one direction per baseline is considered
        minimum_ell: minimum ell to consider in the analysis
        """
        self.directory = directory
        self.cross_only = cross_only
        self.one_way_baseline = one_way_baseline
        self.minimum_ell = minimum_ell


    def __call__(self, 
                 vis_filename,      # vis_filename: name of the visibility response file
                 ellm_filename,     # ellm_filename: name of the ell and m file
                 save_directory,    # save_directory: directory where the processed data files are saved
                 generate_time_series_data=True, # if True, extrapolate time series visibility response data from the input data.
                 start_time=0,      # start_time: start time of the time series data
                 end_time=2*np.pi,  # end_time: end time of the time series data
                 step_time=0.01,    # step_time: time step of the time series data
                 reference_time=0,  # reference_time: reference time for the time series data, i.e. the time of the input data.
                 save_XtX=True):    # if True, save the transposed response matrix multiply the response matrix to the save_directory; if False, just save the response matrix.
        """
        Output:
        saved files: vis_filename and ellm_filename in the save_directory; if save_XtX=True, XtXvis_filename in the save_directory
        If save_XtX=True, the output vis_filename is the transposed response matrix multiply the response matrix.
            vis.shape=(NFREQS, NLMS*2 - Nmodes_to_mask,  NLMS*2 - Nmodes_to_mask), data type=float
        If save_XtX=False, the output vis_filename is the response matrix.
            vis.shape=(NFREQS, NTIMES*NBASELINES, NLMS*2 - Nmodes_to_mask, 2), data type=float
        ellm.shape=(NLMS*2 - Nmodes_to_mask, 2), data type=int
        """
        if fu.are_directories_equivalent(save_directory, self.directory):
            print("The input and output directories are the same. Exiting the function.")
            return

        vis = np.load(os.path.join(self.directory, vis_filename)) # vis.shape=(NFREQS, NTIMES, NANTS, NANTS, NLMS*2), data type=complex
        ellm = np.load(os.path.join(self.directory, ellm_filename))
        self.ell = ellm[:,0]
        self.m = ellm[:,1]
        self.NLMS = ellm.shape[0]
        if generate_time_series_data:
            assert vis.shape[1]==1, "The time-axis dimension of the input array must be 1. \
                                    If you do not want to generate time series data, set generate_time_series_data=False."
            time_sequence = np.arange(start_time, end_time, step_time) - reference_time
            vis = self.generate_time_series_data(vis, time_sequence) # vis.shape=(NFREQS, NTIMES, NBASELINES, NLMS*2), data type=complex
        vis = self.vis_response_mask(vis, 
                                     cross_only=self.cross_only, 
                                     one_way_baseline=self.one_way_baseline, 
                                     minimum_ell=self.minimum_ell) # vis.shape=(NFREQS, NTIMES*NBASELINES, NLMS*2 - Nmodes_to_mask, 2), data type=float
        if save_XtX:
            vis = np.einsum('fnar, fnbr -> fab', vis, vis, optimize=True) # vis.shape=(NFREQS, NTIMES*NBASELINES, NLMS*2 - Nmodes_to_mask, 2), data type=float
            fu.save_array_to_directory(vis, save_directory, 'XtX'+vis_filename)
        else:
            fu.save_array_to_directory(vis, save_directory, vis_filename)
        ellm = np.stack((self.ell, self.m), axis=-1) # ellm.shape=(NLMS*2 - Nmodes_to_mask, 2), data type=int
        fu.save_array_to_directory(ellm, save_directory, ellm_filename)
        print("Data processing completed.")
        return
        
    # Generate time series visibility response data
    def generate_time_series_data(self, vis_array, times):
        """
        Input:
        vis_array.shape=(NFREQS, NTIMES=1, NANTS, NANTS, NLMS*2), data type=complex
        times.shape=(N_times,), data type=float (in radians)

        Output:
        data.shape=(NFREQS, N_times, NANTS, NANTS, NLMS*2), data type=complex
        """
        shape = vis_array.shape
        assert shape[1]==1, "The time-axis dimension of the input array must be 1."
        data = vis_array[:,0,:,:,:].reshape(shape[0], shape[2], shape[3], self.NLMS, 2) # data.shape=(NFREQS, NANTS, NANTS, NLMS, 2)
        rot_array = create_rotation_matrices(self.m, times) # rot_array.shape=(NLMS, N_times, 2, 2)
        data = np.einsum('mtab, fijmb -> ftijma', rot_array, data, optimize=True) # data_rot.shape=(NFREQS, NANTS, NANTS, NLMS, N_times, 2)
        data = data.reshape(shape[0], times.size, shape[2], shape[3], self.NLMS*2)
        return data    

    @fu.complex_to_real_array_decorator
    def vis_response_mask(self, response_matr, cross_only=True, one_way_baseline=True, minimum_ell = 1):
        """
        Input: response_matr.shape=(NFREQS, NTIMES, NANTS, NANTS, NLMS*2), data type=complex
            The first four dimensions index independent visibility measurements, the last dimension indexes the real and imaginary parts of all the spherical harmonics modes of the temperature map. 
            This array describes the linear response of the interferometer to the sky spherical harmonics modes.

        Output: result.shape=(NFREQS, NTIMES*NBASELINES, NLMS*2 - Nmodes_to_mask, 2), data type=float
            The output is a masked version of the input array.
            We have used the decorater "complex_to_real_array_decorator" to convert the output complex array to a real array.
            That's why the last dimension of the output array is 2, which indexes the real and imaginary parts of complex array.

        Function:
        Mask the visbility response array to get only the responses 
        1) to cross-correlations (if cross_only=True),
        2) to only one direction per baseline (if one_way_baseline=True),
        3) to the selected spherical harmonics modes (ell >= minimum_ell).
        Since the imaginary parts of all m=0 modes of a temperature map are zero, this function also removes the responses to those modes.
        """

        shape = response_matr.shape
        assert shape[2] == shape[3], "Data must have the same dimension in the two antenna axes"
        assert shape[-1] == 2*self.NLMS, "Data must have the same dimension as twice the number of SH modes (response matrics to real and imaginary parts for each sky spherical mode)"
        
        # Create a mask for the baselines
        if one_way_baseline:
            # Create an upper triangular mask for the two antenna axes
            if cross_only:
                mask = np.triu(np.ones((shape[2], shape[3]), dtype=bool), 1)
            else:
                mask = np.triu(np.ones((shape[2], shape[3]), dtype=bool), 0)
        else:
            if cross_only:
                mask = np.ones((shape[2], shape[3]), dtype=bool)
                mask[np.diag_indices(shape[2])] = False
            else:
                mask = np.ones((shape[2], shape[3]), dtype=bool)
        # Get the indices of the unmasked baselines
        self.ants_pair_indices = np.argwhere(mask)
        self.n_baselines = self.ants_pair_indices.shape[0]


        # Apply masks to the data  

        # Apply the baseline mask to the data, and reshape it to have less dimensions
        result = response_matr[:, :, mask, :].reshape(shape[0], shape[1]*self.n_baselines, -1) # shape=(NFREQS, NTIMES*NBASELINES, NLMS*2)
              
        # Mask the SH modes to avoid the ones corresponding to the imaginary part of the m=0 modes.
        useful_modes = np.concatenate( (np.arange(self.NLMS), self.NLMS + np.where(self.m>0)[0]) )
        ell = np.append(self.ell, self.ell[np.where(self.m>0)])
        m = np.append(self.m, self.m[np.where(self.m>0)])
        result = result[..., useful_modes]

        # Mask the SH modes whose ell is greater than the minimum_ell
        result = result[..., np.where(ell>=minimum_ell)[0]] # shape=(NFREQS, NTIMES*NBASELINES, Nsources), where Nsources = NLMS*2 - Nmodes_to_mask

        # Update the ell and m arrays
        self.ell = ell[np.where(ell>=minimum_ell)]
        self.m = m[np.where(ell>=minimum_ell)]

        return result


def create_rotation_matrices(m, t):
    N_m = len(m)
    N_t = len(t)
    result = np.zeros((N_m, N_t, 2, 2))

    for i, m_val in enumerate(m):
        for j, t_val in enumerate(t):
            theta = m_val * t_val
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            result[i, j] = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]

    return result