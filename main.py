from util import *

import settings as prm
import markov
import numpy as np
import bayesianLocalization_stack as bl
from model import *
import matplotlib.pyplot as plt


# Make noisy data background
data = Load("testStack2.tiff")
data += simulator.noise(data.shape, mu = 0.0, sigma = 2.0)
# Make a model
model = Model(data)

particleList = list()

for i in range(20):
    particleList.append(model.add_particle(method = "missing density"))
    model.set_active_particle(particleList[-1])
    model.optimize_active_particle()
    model.score_active_particle()
    print(particleList[-1])

model.set_active_particle(None)
stack = model.generate_stack()
Save(stack, "resultStack.tif")
