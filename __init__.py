# Necessary Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.fft import fftn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
