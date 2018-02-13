import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as tcuda
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

from tqdm import *
if isnotebook():
    mtqdm=tqdm_notebook
    mtrange=tnrange
    write=print
else:
    mtqdm=tqdm
    mtrange=trange
    write=tqdm.write
