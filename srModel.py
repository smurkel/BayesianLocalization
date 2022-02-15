import settings as prm
import bayesianLocalization as bl
import simulator as sim
import numpy as np
from itertools import count

class Particle:
    idgen = count(0)

    def __init__(self, x, y, s, i):
        self.x = x
        self.y = y
        self.s = s
        self.i = i
        self.b = list()
        self.id = next(Particle.idgen)

    def genImage(self, imageRadius):
        return sim.fp_image(imageRadius, self.a)

    def __eq__(self, other):
        if isinstance(other, Particle):
            return self.id == other.id
        return False

class Model:
    def __init__(self, data):
        self.data = data
        self.width, self.height, self.frames = data.shape
        self.particles = list()

    def genStack(self):
        stack = np.zeros((self.width, self.height, self.frames))
        particle_roi = list()
        particle_imgs = list()
        print("Img size = {} x {}".format(2 * self.psf_roi_rad, 2 * self.psf_roi_rad))
        for p in self.particles:
            print("Generating img for particle number {}".format(p.id))
            x_min = max([0, int(p.x - self.psf_roi_rad)])
            x_max = min([self.width, int(p.x + self.psf_roi_rad)])
            y_min = max([0, int(p.y - self.psf_roi_rad)])
            y_max = min([self.height, int(p.y + self.psf_roi_rad)])
            p_img = np.zeros((x_max-x_min, y_max-y_min))
            particle_roi.append((x_min,x_max,y_min,y_max))
            i = 0
            for x in range(x_min, x_max):
                j = 0
                for y in range(y_min, y_max):
                    p_img[i,j] = sim.fp_pixel_value((p.x - x_min - 1, p.y - y_min - 1, p.s, p.i), i, j)
                    j+=1
                i+=1
            particle_imgs.append(p_img)
        for f in range(self.frames):
            print("Rendering frame nr. {}".format(f))
            for i in range(len(self.particles)):
                if self.particles[i].b[f]:
                    roi = particle_roi[i]
                    stack[roi[0]:roi[1],roi[2]:roi[3], f] += particle_imgs[i]
        return stack

    def addParticle(self, particle = None):
        if isinstance(particle, Particle):
            self.particles.append(particle)
        else:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            sigma = prm.PRIOR_PDF_SIGMA.rvs(1)
            intensity = prm.PRIOR_PDF_INTENSITY.rvs(1)
            newParticle = Particle(x,y,sigma,intensity)
            newParticle.b = prm.HMM.sample(length = self.frames)
            self.particles.append(newParticle)

    def __str__(self):
        return "Bayesian localization model:\n\t{} particles\n\t{} x {} img size\n\t{} frames".format(len(self.particles), self.width, self.height, self.frames)



