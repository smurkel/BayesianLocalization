
from itertools import count
import settings
import numpy as np

import simulator


class Particle:
    idgen = count(0)

    def __init__(self, a=None, b=None):
        self.id = next(Particle.idgen)
        """
        A particle can be initialized with or without values for the model parameters:
            a = (x, y, size, intensity)
            b = (b0, b1, ..., bN)
        """
        self.a = a
        self.b = b

        self.roi = None
        self.img = None

    def __eq__(self, other):
        if isinstance(other, Particle):
            return self.id == other.id
        return False

    def update_roi(self, img_width, img_height):
        xmin = max([0, self.a[0] - settings.CROP_SIZE])
        ymin = max([0, self.a[1] - settings.CROP_SIZE])
        xmax = min([img_width, self.a[0] + settings.CROP_SIZE])
        ymax = min([img_height, self.a[0] + settings.CROP_SIZE])
        self.roi = (xmin, ymin, xmax, ymax)

    def update_img(self):
        self.img = np.zeros((settings.CROP_SIZE * 2 + 1, settings.CROP_SIZE * 2 + 1))
        ix = 0
        for x in range(self.roi[0], self.roi[2]):
            iy = 0
            for y in range(self.roi[0], self.roi[2]):
                self.img[ix,iy] = simulator.particle_pixel_value(self.a, (x, y))
                iy+=1
            ix+=1

class Model:

    def __init__(self, shape):
        """
        :param shape: tuple, (stack_width, stack_height, stack_depth)
        """
        self.width = shape[0]
        self.height = shape[1]
        self.depth = shape[2]

        self.particles = list()

    def add_particle(self, method = "random"):
        """
        :param method: the method determines in which way a new particle is generated. Options are
            "random": a new particle is added at a uniform random x, y position and with size and brightness according to the prior pdf set in the settings file
        """
        new_particle = None
        if method == "random":
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            size = settings.PRIOR_PDF_SIGMA.rvs(1)
            intensity = settings.PRIOR_PDF_INTENSITY.rvs(1)
            new_particle = Particle((x, y, size, intensity))
            self.particles.append(new_particle)

        return new_particle

    def optimize_particle(self, particle):
        # 1. MAP estimation

        # 2. MAP refinement
        # 3. Find best state sequence

    def score_particle(self, particle):