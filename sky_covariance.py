import numpy as np
import fisher_utils as fu
from scipy.optimize import minimize


###################################################################################################
# This file contains classes for 
# 1) the angular power spectrum (APS) of the sky signal.
# 2) the data covariance in the visibility space; the covariance is composed of two parts, 
#    the instrumental noise covariance and the sky signal covariance. The later is projected from 
#    the sky space to the visibility space by the visibility response operators.
# 3) the fisher analysis. The fisher analysis is performed in the visibility space. 
#    It can deal with arbitrary number of the APS models, each of which has its own 4 parameters.



###################################################################################################

# Basic class: spatical part of the angular power spectrum



class AngularStructure(object):
    l_pivot = 1000.0
    ell_c = 10

    def angular_covariance(self, ell):
        result = np.zeros_like(np.array(ell))
        for i in range(len(ell)):
            if ell[i] < self.ell_c:
                result[i] = (np.e ** self.A)*(self.ell_c / self.l_pivot)**(self.alpha)
            else:
                result[i] = (np.e ** self.A)*(ell[i] / self.l_pivot)**(self.alpha)
        return result
    
    def angular_covar_alpha_derivative(self, ell):
        result = np.zeros_like(np.array(ell))
        for i in range(len(ell)):
            if ell[i] < self.ell_c:
                result[i] = (np.e ** self.A)*(self.ell_c / self.l_pivot)**(self.alpha) * np.log(self.ell_c / self.l_pivot)
            else:           
                result[i] = (np.e ** self.A)*(ell[i] / self.l_pivot)**(self.alpha) * np.log(ell[i] / self.l_pivot)
        return result
        
    def angular_covar_A_derivative(self, ell):
        result = np.zeros_like(np.array(ell))
        for i in range(len(ell)):
            if ell[i] < self.ell_c:
                result[i] = (np.e ** self.A)*(self.ell_c / self.l_pivot)**(self.alpha)
            else:
                result[i] = (np.e ** self.A)*(ell[i] / self.l_pivot)**(self.alpha)
        return result
        

"""
class AngularStructure(object):
    l_pivot = 1000.0

    def angular_covariance(self, ell):
        return self.A*(ell / self.l_pivot)**(self.alpha)
    
    def angular_covar_alpha_derivative(self, ell):
        return self.A*(ell / self.l_pivot)**(self.alpha) * np.log(ell / self.l_pivot)

    def angular_covar_A_derivative(self, ell):
        return (ell / self.l_pivot)**(self.alpha)
"""

# Angular power spectrum classes: C_l^alpha, beta, zeta 

class BaseAPS(AngularStructure):
    nu_pivot = 130.0
    
    def frequency_covariance(self, nu1, nu2):
        return (nu1*nu2/self.nu_pivot**2)**(self.beta) * np.exp( -0.5 * (np.log(nu1/nu2) / self.zeta)**2)
    
    def log_frequency_covariance(self, nu1, nu2):
        """
        x = log(nu/nu_pivot)
        """
        x1 = np.log(nu1/self.nu_pivot)
        x2 = np.log(nu2/self.nu_pivot)
        return self.beta * (x1+x2) - (x1-x2)**2 / (2 * self.zeta**2)
    
    def log_freq_cov_matrix(self, freqs):
        return fu.calculate_function_values(freqs, freqs, self.log_frequency_covariance)
    
    def angular_powerspectrum(self, ell, nu1, nu2):
        frequency_part = fu.calculate_function_values(nu1, nu2, self.frequency_covariance)
        angular_part = self.angular_covariance(ell)
        return angular_part[:, np.newaxis, np.newaxis] * frequency_part
    
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
    #A = 7.00e-4
    A = -7.264
    alpha = -2.40
    beta = -2.80
    zeta = 4.0

class ExtragalacticPointSource(BaseAPS):
    """
    Reference: Mario G. Santos (2005)
    Units: K^2
    """
    #A = 5.70e-5 
    A = -9.772
    alpha = -1.1
    beta = -2.07
    zeta = 1.0

class ExtragalacticFreeFree(BaseAPS):
    """
    Reference: Mario G. Santos (2005)
    Units: K^2
    """
    #A = 1.40e-8
    A = -18.084
    alpha = -1.0
    beta = -2.10
    zeta = 35.0

class GalacticFreeFree(BaseAPS):
    """
    Reference: Mario G. Santos (2005)
    Units: K^2
    """
    #A = 8.80e-8
    A = -16.246
    alpha = -3.0
    beta = -2.15
    zeta = 35.0

class ExtragalacticBackground1(BaseAPS):
    """
    Reference: 
    Units: K^2
    """
    #A = 3.00e-4
    A = -8.112
    alpha = -2.40
    beta = -2.66
    zeta = 4.0

class ExtragalacticBackground2(BaseAPS):
    """
    Reference: 
    Units: K^2
    """
    #A = 3.00e-4
    A = -8.112
    alpha = 0
    beta = -2.66
    zeta = 30.0


class CustomizeAPS(BaseAPS):
    def __init__(self, A, alpha, beta, zeta):
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta


###################################################################################################
# Another augular power spectrum class (with universal spectral structure)
        
class Universal_SED(AngularStructure):

    def __init__(self, freqs,  aps_class_obj, order=3):
        self.A = aps_class_obj.A
        self.alpha = aps_class_obj.alpha
        self.n_beta = order
        self.nu_pivot = aps_class_obj.nu_pivot
        self.fit_polynomial(freqs, aps_class_obj, order)
        self.generate_SED_derivatives(freqs)

    def log_freq(self, nu, nu_pivot=130.0):
        return np.log(nu/nu_pivot)

    def polynomial_freq_covariance(self, params, x):
        """"
        x = log(nu/nu_pivot)
        SED model function: 
            f(x) = exp( x * beta(x) ) 
                 = exp( x * beta0 + x**2 * beta1 + x**3 * beta2 + ...)   -- Tylor expansion
        Logrithm of frequency-frequency covariance), the poloynomial function of x = log(nu/nu_pivot):
            d(x_i, x_j) = sum_k params[k] * (x_i**(k+1) + x_j**(k+1))
        """
        var1_grid, var2_grid = np.meshgrid(x, x, indexing='ij')
        aux = [params[k] * (var1_grid**(k+1) + var2_grid**(k+1)) for k in range(len(params))]
        return np.array(aux).sum(axis=0)

    def fit_polynomial(self, freqs, aps_obj, order):

        def loss_function(params, x, ref_data):
            predicted = self.polynomial_freq_covariance(params, x)
            return np.sum((predicted - ref_data)**2)

        x = self.log_freq(freqs, aps_obj.nu_pivot)
        y = aps_obj.log_freq_cov_matrix(freqs)
    
        initial_params = np.ones(order)  # Initial guess for coefficients
        initial_params[0] = aps_obj.beta
        result = minimize(loss_function, initial_params, args=(x, y), method='BFGS')

        if result.success:
            fitted_params = result.x
            self.betas = fitted_params # shape: (n_beta,)
            return 
        else:
            raise ValueError("Fitting failed.")
    
    def SED_function(self, freqs):
        xs = self.log_freq(freqs, self.nu_pivot)
        params = self.betas
        aux = [params[k] * xs**(k+1)  for k in range(len(params))]
        log_result = np.array(aux).sum(axis=0)
        self.SED = np.exp(log_result)
        return self.SED  # shape: (n_freqs,)
    
    def generate_SED_derivatives(self, freqs):
        aux = self.SED_function(freqs)
        xs = self.log_freq(freqs, self.nu_pivot)
        result = np.zeros((len(self.betas), len(freqs)))

        for k in range(len(self.betas)):
            result[k] = aux * xs**(k+1)

        self.SED_derivatives = result  # shape: (n_beta, n_freqs)
        return 
    






