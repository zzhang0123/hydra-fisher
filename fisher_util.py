import time
import functools
import numpy as np

def calculate_function_values(var1_values, var2_values, your_function):
    """
    Calculate function values for all combinations of two input variable arrays using NumPy.

    Parameters:
    var1_values: NumPy array of the first input variable values
    var2_values: NumPy array of the second input variable values
    your_function: The function to calculate, which takes two NumPy arrays as input.

    Returns:
    A NumPy array containing the function values for all combinations.
    """
    # Create a grid of all combinations of var1 and var2
    var1_grid, var2_grid = np.meshgrid(var1_values, var2_values, indexing='ij')

    # Use your_function to calculate the function values for all combinations
    result = your_function(var1_grid, var2_grid)

    return result

def myTiming(func):
    """
    A decorator that prints the time a function takes
    to execute.
    """
    @functools.wraps(func)
    def decorated_function(*args, **kwargs):
        print("Entering" + str(func))
        begin = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print("Elaspsed time", end - begin)
        return(result)
    return(decorated_function)


# Define the decorator
def complex_to_real_array_decorator(func):
    """
    A decorator that converts a function that returns a complex array into a function that returns a real array.
    """
    def wrapper(*args, **kwargs):
        # Call the original function to get the complex array
        complex_result = func(*args, **kwargs)
        
        # Split the complex array into real and imaginary parts
        real_part = np.real(complex_result)
        imag_part = np.imag(complex_result)
        
        # Stack the real and imaginary parts along a new axis
        real_imag_array = np.stack((real_part, imag_part), axis=-1)
        
        return real_imag_array
    
    return wrapper

class ClassDecoratorChangeRefences:
    """
    A class decorator that changes the reference ell and frequency of the decorated class.
    """
    def __init__(self, l_pivot=1000, nu_pivot=130):
        self.l_pivot = l_pivot
        self.nu_pivot = nu_pivot

    def __call__(self, cls):
        # Add or modify behavior of the original class here
        def wrapped(*args, **kwargs):
            instance = cls(*args, **kwargs)
            instance.l_pivot = self.l_pivot
            instance.nu_pivot = self.nu_pivot
            return instance
        return wrapped