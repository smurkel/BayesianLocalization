from util import *

import settings as prm
import markov
import numpy as np
import bayesianLocalization_stack as bl
from model import *
import matplotlib.pyplot as plt


# Make noisy data background
data = Load("testStack.tif")
print(data.shape)
data += simulator.noise(data.shape, mu = 0.0, sigma = 2.0)
print(np.shape(data))
# Make a model
model = Model(data)
particle = model.add_particle(method = "missing density")

print(particle)

bl.maximum_a_posteriori_estimate(model, particle)




