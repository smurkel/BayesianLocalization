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

    def __str__(self):
        return "\nParticle (id = {}) with parameters:\na = ({},{},{},{})\nb = ".format(self.id, *self.a) + str(self.b)

    def set_a(self, a):
        self.a = a
        self.update_roi()
        self.update_img()

    def update_roi(self):
        xmin = int(self.a[0] - settings.CROP_SIZE)
        ymin = int(self.a[1] - settings.CROP_SIZE)
        xmax = int(self.a[0] + settings.CROP_SIZE)
        ymax = int(self.a[1] + settings.CROP_SIZE)
        if xmin < 0:
            xmin = 0
            xmax = 2 * settings.CROP_SIZE
        elif xmax >= settings.MODEL_WIDTH:
            xmin = settings.MODEL_WIDTH - 2 * settings.CROP_SIZE
            xmax = settings.MODEL_WIDTH
        if ymin < 0:
            ymin = 0
            ymax = 2 * settings.CROP_SIZE
        elif ymax >= settings.MODEL_HEIGHT:
            ymin = settings.MODEL_HEIGHT - 2 * settings.CROP_SIZE
            ymax = settings.MODEL_HEIGHT

        self.roi = (xmin, ymin, xmax, ymax)
        return self.roi

    def update_img(self):
        self.img = np.zeros((self.roi[2]-self.roi[0], self.roi[3]-self.roi[1]))
        ix = 0
        for x in range(self.roi[0], self.roi[2]):
            iy = 0
            for y in range(self.roi[1], self.roi[3]):
                self.img[ix, iy] = simulator.particle_pixel_value(self.a, (x, y))
                iy += 1
            ix += 1
        return self.img
