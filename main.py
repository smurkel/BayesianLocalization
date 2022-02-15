from util import *

import modelparams as prm
import markov
import numpy as np
import bayesianLocalization as bl
from srModel import *

data = Load("testStack.tif")
model = Model(*data.shape)

## Add one particle
MAP = bl.estimate_map(data[:, :, 13])
print(MAP)
MAP = bl.refine_map(data[:, :, 13], MAP)
print(MAP)










