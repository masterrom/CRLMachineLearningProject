import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def stableSoftMax(x):
    x = np.exp(x - np.max(x))
    return x/x.sum(axis=0)

z = [2345, 3456, 6543]
x = [1,2,3]

