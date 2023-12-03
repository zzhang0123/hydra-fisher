import numpy as np


import fisher_utils as fu

import torch

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


torch.from_numpy(np.array([1,2,3,4,5,6,7,8,9,10])).view(2,5)

