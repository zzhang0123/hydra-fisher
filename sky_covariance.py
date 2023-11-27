import numpy as np
import fisher_util as fu


###################################################################################################
# This file contains classes for 
# 1) the angular power spectrum (APS) of the sky signal.
# 2) the data covariance in the visibility space; the covariance is composed of two parts, 
#    the instrumental noise covariance and the sky signal covariance. The later is projected from 
#    the sky space to the visibility space by the visibility response operators.
# 3) the fisher analysis. The fisher analysis is performed in the visibility space. 
#    It can deal with arbitrary number of the APS models, each of which has its own 4 parameters.


###################################################################################################
# Problem 1: C_ell units: K^2, but visibility response operator is defined as [visibility]/Jy.
#            (Need to convert from Jy to K.) # Jacob
# Problem 2: Large data arrays. (Might need to speed up for realistic size.) Using Pytorch? Or Jax?





###################################################################################################
# Angular power spectrum classes


class BaseAPS(object):
    nu_pivot = 130.0
    l_pivot = 1000.0

    def angular_covariance(self, ell):
        return self.A*(ell / self.l_pivot)**(self.alpha)
    
    def frequency_covariance(self, nu1, nu2):
        return (nu1*nu2/self.nu_pivot**2)**(self.beta) * np.exp( -0.5 * (np.log(nu1/nu2) / self.zeta)**2)
    
    def angular_powerspectrum(self, ell, nu1, nu2):
        frequency_part = fu.calculate_function_values(nu1, nu2, self.frequency_covariance)
        angular_part = self.angular_covariance(ell)
        return angular_part[:, np.newaxis, np.newaxis] * frequency_part
    
    def angular_covar_alpha_derivative(self, ell):
        return self.A*(ell / self.l_pivot)**(self.alpha) * np.log(ell / self.l_pivot)
    
    def angular_covar_A_derivative(self, ell):
        return (ell / self.l_pivot)**(self.alpha)
    
    def freq_covar_beta_derivative(self, nu1, nu2):
        return np.log(nu1*nu2/self.nu_pivot**2) * self.frequency_covariance(nu1, nu2)
    
    def freq_covar_zeta_derivative(self, nu1, nu2):
        return ((np.log(nu1/nu2))**2/self.zeta**3) * self.frequency_covariance(nu1, nu2)
    
    def aps_A_derivative(self, ell, nu1, nu2):
        frequency_part = fu.calculate_function_values(nu1, nu2, self.frequency_covariance)
        angular_part = self.angular_covar_A_derivative(ell)
        return angular_part[:, np.newaxis, np.newaxis] * frequency_part
    
    def aps_alpha_derivative(self, ell, nu1, nu2):
        frequency_part = fu.calculate_function_values(nu1, nu2, self.frequency_covariance)
        angular_part = self.angular_covar_alpha_derivative(ell)
        return angular_part[:, np.newaxis, np.newaxis] * frequency_part
    
    def aps_beta_derivative(self, ell, nu1, nu2):
        frequency_part = fu.calculate_function_values(nu1, nu2, self.freq_covar_beta_derivative)
        angular_part = self.angular_covariance(ell)
        return angular_part[:, np.newaxis, np.newaxis] * frequency_part
    
    def aps_zeta_derivative(self, ell, nu1, nu2):
        frequency_part = fu.calculate_function_values(nu1, nu2, self.freq_covar_zeta_derivative)
        angular_part = self.angular_covariance(ell)
        return angular_part[:, np.newaxis, np.newaxis] * frequency_part


class GalacticSynchrotron(BaseAPS):
    """
    Reference: Mario G. Santos (2005)
    Units: K^2
    """
    A = 7.00e-4
    alpha = -2.40
    beta = -2.80
    zeta = 4.0

class ExtragalacticPointSource(BaseAPS):
    """
    Reference: Mario G. Santos (2015)
    Units: K^2
    """
    A = 5.70e-5
    alpha = -1.1
    beta = -2.07
    zeta = 1.0

class ExtragalacticFreeFree(BaseAPS):
    """
    Reference: Mario G. Santos (2015)
    Units: K^2
    """
    A = 1.40e-8
    alpha = -1.0
    beta = -2.10
    zeta = 35.0

class GalacticFreeFree(BaseAPS):
    """
    Reference: Mario G. Santos (2015)
    Units: K^2
    """
    A = 8.80e-8
    alpha = -3.0
    beta = -2.15
    zeta = 35.0

class CustomizeAPS(BaseAPS):
    def __init__(self, A, alpha, beta, zeta):
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta




###################################################################################################
# Fisher matrix class

class FisherMatrix():
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
        






