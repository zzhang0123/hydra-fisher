import time
import functools
import numpy as np
import glob
import re
import os
from mpiutils import rank0


radian_per_hour = 2*np.pi/24

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

def myTiming_rank0(func):
    @functools.wraps(func)
    def decorated_function(*args, **kwargs):
        if rank0:
            print("Entering" + str(func))
            begin = time.perf_counter()
            # print(begin)
        result = func(*args, **kwargs)
        if rank0:
            end = time.perf_counter()
            # print(end)
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


def are_directories_equivalent(dir1, dir2):
    # Convert both paths to absolute paths and normalize them
    normalized_dir1 = os.path.normpath(os.path.abspath(dir1))
    normalized_dir2 = os.path.normpath(os.path.abspath(dir2))

    # Additionally, normalize the case for case-insensitive file systems
    return os.path.normcase(normalized_dir1) == os.path.normcase(normalized_dir2)
 

def save_array_to_directory(array, directory, filename):
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        os.makedirs(directory)

    # Full path for the file
    file_path = os.path.join(directory, filename)

    # Save the array
    np.save(file_path, array)
    return


def get_sorted_filenames(directory, pattern, get_path=False):
    # Construct the joined pattern to match files
    # pattern = 'response_sh_*.npy' or 'response_sh_ellm_*.npy' or ...
    join_pattern = os.path.join(directory, pattern)

    # Use glob to find files matching the joined pattern
    files = glob.glob(join_pattern)

    # Filter out files that don't have a four-digit number following "response_sh_"
    # Assuming pattern is something like 'response_sh_*.npy' or 'response_sh_ellm_*.npy'
    # We extract the part before '*' and use it in the regex
    base_pattern = pattern.split('*')[0]  # Extract the base part of the pattern
    regex_pattern = rf"{re.escape(base_pattern)}(\d{{4}})\.npy$"
    regex = re.compile(regex_pattern)
    filtered_files = [f for f in files if regex.search(os.path.basename(f))]

    # Sort the files
    sorted_files = sorted(filtered_files, key=lambda x: int(regex.search(os.path.basename(x)).group(1)))
    
    # Extract only the filenames
    sorted_filenames = [os.path.basename(f) for f in sorted_files]

    if get_path:
        return sorted_files
    else:
        return sorted_filenames


def extract_freqs(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains 'freqs:'
            if 'freqs:' in line:
                # Extract the frequency value using regex
                match = re.search(r'freqs:\s*(\d+\.\d+)', line)
                if match:
                    return float(match.group(1))
    return None