# Generate a test stack
import numpy as np
from model import *
from util import *

data = np.zeros((21, 21, 20))
n_particles = 20
model = Model(data)
for p in range(n_particles):
    particle = model.add_particle(method = "random")



stack = model.generate_stack()
for particle in model.particles:
    print(particle)

Save(stack, "testStack3")
