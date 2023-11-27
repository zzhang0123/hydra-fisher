###################################################################################################
# Data space covariance class
    

class DataSpaceCovariance():
    def __init__(self, ell, m, vis, freqs, cross_only=True, one_way_baseline=False, noise_scale=1e-3, minimum_ell=10):
         # vis.shape=(NFREQS, NTIMES, NANTS, NANTS, NSRCS)
        self.n_SH_modes = ell.size
        self.noise_scale = noise_scale
        self.ell = ell
        self.m = m
        self.freqs = freqs
        self.vis = self.vis_response_mask(vis, cross_only=cross_only, one_way_baseline=one_way_baseline, minimum_ell=minimum_ell)
        # self.vis.shape=(NFREQS, NTIMES*NBASELINES, NLMS,2)
        shape = self.vis.shape
        self.n_real_dof = shape[0] * shape[1] * shape[3]

    @fu.myTiming
    def __call__(self, APS_function):
        cov = 0.5 * APS_function(self.ell, self.freqs, self.freqs)
        result = np.einsum('aslm, lab, btln -> asmbtn', self.vis, cov, self.vis)
        return result.reshape(self.n_real_dof, self.n_real_dof)


    @fu.complex_to_real_array_decorator
    def vis_response_mask(self, response_matr, cross_only=True, one_way_baseline=False, minimum_ell = 10):

        shape = response_matr.shape
        assert shape[2] == shape[3], "Data must have the same dimension in the two antenna axes"
        assert shape[-1] == 2*self.n_SH_modes, "Data must have the same dimension as twice the number of SH modes"
        
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
        result = response_matr[:, :, mask, :].reshape(shape[0], shape[1]*self.n_baselines, -1)
              
        # Mask the SH modes to avoid the ones corresponding to the imaginary part of the monopole.
        useful_modes = np.concatenate( (np.arange(self.n_SH_modes), self.n_SH_modes + np.where(self.m>0)[0]) )
        ell = np.append(self.ell, self.ell[np.where(self.m>0)])
        result = result[..., useful_modes]
        # Mask the SH modes whose ell is greater than the minimum_ell
        result = result[..., np.where(ell>=minimum_ell)[0]]

        # Update the ell and m arrays
        self.ell = ell[np.where(ell>=minimum_ell)]
        self.m = np.append(self.m, self.m[np.where(self.m>0)])[np.where(ell>=minimum_ell)]

        return result
    
    def noise_covariance(self):
        return self.noise_scale**2 * np.eye(self.n_real_dof)
