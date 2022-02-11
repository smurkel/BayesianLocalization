from util import *

import simulator
import markov
import numpy as np
import bayesianLocalization as bl
####################
###### params ######
####################
nParticles = 1
nFrames = 30
imgSize = 5
noiseMu = 100.0
noiseSigma = 10.0
p_on = 0.05
p_off = 0.2
p_bleach = 0.1
p_false_emission = 0.0
####################
####################
bl.IMG_SIZE = imgSize
data = simulator.Particle(0.0, 0.0, 2.0, 1000.0).getImage(imgSize) + simulator.noise(imgSize, 0.0, 0.0)
MAP, _ = bl.estimate_map(data)
MAP = bl.refine_map(data, MAP)
print("Maximum a-posteriori estimate for a:")
print(MAP)
print("Laplace approximation")
print(bl.laplace_approximation_value(data, MAP))
