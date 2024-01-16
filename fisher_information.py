import numpy as np
import fisher_utils as fu
from mpiutils import *


###################################################################################################
# Fisher matrix class

class FisherInformation():
    def __init__(self, APS_obj_list, freqs, ell, directory, pattern):
        """"
        APS_obj_list: a list of (universal SED) APS classes
        freqs: a list of frequencies in MHz
        directory: the directory where the XtX matrices are stored
        pattern: the pattern of the XtX matrices, e.g., 'XtXresponse_sh_*.npy'
        """
        self.fields = APS_obj_list
        self.n_freqs = len(freqs)
        self.n_fields = len(APS_obj_list)
        self.ell = ell
        self.sorted_filenames = fu.get_sorted_filenames(directory, pattern, get_path=True)

        self.SED_all = np.array([APS_obj.SED for APS_obj in APS_obj_list])
        C_ell_all = np.array([0.5 * APS_obj.angular_covariance(ell) for APS_obj in APS_obj_list]).flatten() # 0.5 factor accounts for the real/imag parts of the covariance

        self.M = self.operator(self.SED_all, self.SED_all)
        self.C_aux = np.linalg.inv(np.diag(1/C_ell_all) + self.M) 
        self.C_ell_all = np.diag(C_ell_all)

        self.all_params_list = self.all_params_list()
        self.n_all_params = len(self.all_params_list)

    def generate_partial_derivative_SED_all(self, i, j):
        """
        i: the index of the field
        j: the index of the beta-parameters
        """
        aux = np.zeros((self.n_fields, self.n_freqs))
        aux[i, :] = self.fields[i].SED_derivatives[j,:]
        
        return aux
    
    def generate_partial_derivative_C_ell_all(self, i, j):
        """
        i: the index of the field
        j: the index of the angular structure parameters, 
            j=0 -> A, j=1 -> alpha.
        """
        aux = np.zeros((self.n_fields, self.n_freqs))
        if j == 0:
            aux[i, :] = 0.5 * self.fields[i].angular_covar_A_derivative(self.ell)
        elif j == 1:
            aux[i, :] = 0.5 * self.fields[i].angular_covar_alpha_derivative(self.ell)
        else:
            raise ValueError("The index of the angular structure parameter is wrong.")
        
        return np.diag(aux.flatten())

    def F_alpha_beta_aa(self, field1_index, parameter1_index, field2_index, parameter2_index):
        """
        Both parameters are angular structure parameters.
        """
        aux = self.calculation_module()  # add parentheses to avoid memory error
        aux_1 = aux @ self.generate_partial_derivative_C_ell_all(field1_index, parameter1_index)
        aux_2 = aux @ self.generate_partial_derivative_C_ell_all(field2_index, parameter2_index)
        return 0.5 * np.trace(aux_1 @ aux_2)


    def F_alpha_beta_af(self, field1_index, parameter1_index, field2_index, parameter2_index):
        """
        The first parameter is an angular structure parameter,
        while the other parameter is a SED (or frequency/spectral) parameter.
        """
        aux_0 = self.calculation_module()
        C_partial = self.generate_partial_derivative_C_ell_all(field1_index, parameter1_index)
        aux_1_l = self.calculation_module(order=1, left_type=True, field1_ind=field2_index, param1_ind=parameter2_index)
        aux_1_r = self.calculation_module(order=1, left_type=False, field2_ind=field2_index, param2_ind=parameter2_index)
        
        result = (aux_0 @ C_partial) @ (aux_1_r @ self.C_ell_all) 
        result += (aux_1_l @ C_partial) @ (aux_0 @ self.C_ell_all)

        return 0.5 * np.trace(result)

    def F_alpha_beta_ff(self, field1_index, parameter1_index, field2_index, parameter2_index):
        """
        Both parameters are SED (frequency/spectral) parameters.
        """
        aux_0 = self.calculation_module() @ self.C_ell_all
        aux_f1_1_l = self.calculation_module(order=1, left_type=True, field1_ind=field1_index, param1_ind=parameter1_index) @ self.C_ell_all
        aux_f1_1_r = self.calculation_module(order=1, left_type=False, field2_ind=field1_index, param2_ind=parameter1_index) @ self.C_ell_all
        aux_f2_1_l = self.calculation_module(order=1, left_type=True, field1_ind=field2_index, param1_ind=parameter2_index) @ self.C_ell_all
        aux_f2_1_r = self.calculation_module(order=1, left_type=False, field2_ind=field2_index, param2_ind=parameter2_index) @ self.C_ell_all
        aux_f1_f2 = self.calculation_module(order=2, field1_ind=field1_index, param1_ind=parameter1_index, field2_ind=field2_index, param2_ind=parameter2_index) @ self.C_ell_all
        aux_f2_f1 = self.calculation_module(order=2, field1_ind=field2_index, param1_ind=parameter2_index, field2_ind=field1_index, param2_ind=parameter1_index) @ self.C_ell_all

        result = aux_f1_1_r @ aux_f2_1_r + aux_f2_1_l @ aux_f1_1_l + aux_0 @ aux_f1_f2 + aux_f2_f1 @ aux_0

        return 0.5 * np.trace(result)

    def F_alpha_beta(self, field1, param1, field2, param2):
        if param1<=1 and param2<=1:
            return self.F_alpha_beta_aa(field1, param1, field2, param2)
        elif param1>1 and param2>1:
            return self.F_alpha_beta_ff(field1, param1 - 2, field2, param2 - 2)
        elif param1<=1 and param2>1:
            return self.F_alpha_beta_af(field1, param1, field2, param2 - 2)
        elif param1>1 and param2<=1:
            return self.F_alpha_beta_af(field2, param2, field1, param1 - 2)
    
    def parallel_Fisher_calculation(self):
        """
        Calculate the Fisher matrix in parallel.
        """

        def Fisher_matrix_row(inds_pair):
            field_1_ind, param_1_ind = inds_pair
            row_vector = np.zeros(self.n_all_params)
            for i in range(self.n_all_params):
                field_2_ind, param_2_ind = self.all_params_list[i]
                row_vector[i] = self.F_alpha_beta(field_1_ind, param_1_ind, field_2_ind, param_2_ind)
            return row_vector

        Fisher_matrix = parallel_map(Fisher_matrix_row, self.all_params_list)

        return np.array(Fisher_matrix)
    
    def all_params_list(self):
        params = []
        for field_ind in range(self.n_fields):
            n_params = 2 + self.fields[field_ind].n_beta
            for param_ind in range(n_params):
                params.append([field_ind, param_ind])
        return params

    def operator(self, f1, f2):
        """
        f1 and f2 are arrays of shape (n_fields, n_freqs).
        """
        result = 0
        for i in range(self.n_freqs):
            XtX = np.load(self.sorted_filenames[i])[0]
            result += self.generate_block_matrix(XtX, f1[:, i], f2[:, i])
        return result

    def calculation_module(self, order=0, left_type=True, field1_ind=0, param1_ind=0, field2_ind=0, param2_ind=0):
        if order == 0:
            result = self.M - self.M @ (self.C_aux @ self.M)
        elif order == 2:
            sed_partial_1 = self.generate_partial_derivative_SED_all(field1_ind, param1_ind)
            sed_partial_2 = self.generate_partial_derivative_SED_all(field2_ind, param2_ind)
            M1 = self.operator(sed_partial_1, sed_partial_2)
            M2 = self.operator(sed_partial_1, self.SED_all)
            M3 = self.operator(self.SED_all, sed_partial_2)
            result = M1 - M2 @ (self.C_aux @ M3)
        elif order == 1:
            if left_type:
                sed_partial = self.generate_partial_derivative_SED_all(field1_ind, param1_ind)
                M1 = self.operator(sed_partial, self.SED_all)
                result = M1 - M1 @ (self.C_aux @ self.M)
            else:
                sed_partial = self.generate_partial_derivative_SED_all(field2_ind, param2_ind)
                M1 = self.operator(self.SED_all, sed_partial)
                result = M1 - self.M @ (self.C_aux @ M1)
        else:
            raise ValueError("The derivative order of the calculation module is wrong.")

        return result
    
    def generate_block_matrix(self, X, f1, f2):
        """
        X is a matrix of shape (N, N).
        f1 and f2 are arrays of length d.

        The resulting matrix is of shape (dN, dN).
        """
        assert len(f1) == len(f2)
        a = len(f1)

        N = X.shape[0]

        # Compute the outer product of f_i and f_j
        outer_product = np.outer(f1, f2)

        # Expand dimensions to match X
        outer_product = outer_product[:, np.newaxis, :, np.newaxis]

        # Multiply by X and reshape
        result_matrix = outer_product * X[np.newaxis, :, np.newaxis, :]

        # Reshape and tile to create the final aN by aN matrix
        result_matrix = result_matrix.reshape(a * N, a * N)

        return result_matrix
    


class BaseFisherMatrix():
    def __init__(self, data_covariance_class_instance, *args):
        self.data_covariance_obj = data_covariance_class_instance
        self.ags_classes = args
        # self.total_APS_function = self.sum_functions_inputs(args)
        self.n_params = 4 * len(args)
        Sigma = self.data_covariance_obj(self.sum_aps_inputs) + self.data_covariance_obj.noise_covariance()
        self.Sigma_inv = np.linalg.inv(Sigma)
    
    def sum_functions_inputs(self, *args):
        def sum_func(*inputs):
            return sum(APSclass.angular_powerspectrum(*inputs) for APSclass in args)
        return sum_func
    
    def sum_aps_inputs(self, *inputs):
        result = 0
        for APSclass in self.ags_classes:
            result += APSclass.angular_powerspectrum(*inputs)
        return result
    
    def derivative_func_list(self):
        func_list = []
        for obj in self.ags_classes:
            for func in [obj.aps_A_derivative, 
                         obj.aps_alpha_derivative, 
                         obj.aps_beta_derivative, 
                         obj.aps_zeta_derivative]:
                func_list.append(func)
        return func_list
    

    @fu.myTiming
    def Fisher_matrix(self):
        partial_aps_list = self.derivative_func_list()
        assert len(partial_aps_list) == self.n_params, "The number of partial derivatives is not equal to the number of parameters"

        Fisher = np.zeros((self.n_params, self.n_params))

        Sigma_derivatives_list = [self.Sigma_inv@self.data_covariance_obj(partial_aps) for partial_aps in partial_aps_list]

        del self.data_covariance_obj
        for i in range(self.n_params):
            Fisher[i,i] = 0.5 * np.einsum('ab, ba ', Sigma_derivatives_list[i], Sigma_derivatives_list[i])
            for j in range(i+1, self.n_params):
                # Fisher[j, i] = Fisher[i,j] = self.Fisher_covar_part(self.Sigma_inv, Sigma_derivatives_list[i], Sigma_derivatives_list[j])
                Fisher[j, i] = Fisher[i,j] = 0.5 * np.einsum('ab, ba ', Sigma_derivatives_list[i], Sigma_derivatives_list[j])
        return Fisher
        

