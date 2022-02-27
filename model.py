from particle import *
import settings
import matplotlib.pyplot as plt
import bayesianLocalization_stack as bl

from util import sum_z
class Model:

    def __init__(self, data):
        """
        :param shape: tuple, (stack_width, stack_height, stack_depth)
        """
        self.shape = data.shape
        self.data = data
        self.data_zsum = sum_z(self.data)

        self.width = self.shape[0]
        self.height = self.shape[1]
        self.depth = self.shape[2]
        self.stack = np.zeros(self.shape)
        self.stack_zsum = np.zeros(self.shape)
        self.particles = list()
        self.active_particle = None


        settings.MODEL_WIDTH = self.width
        settings.MODEL_HEIGHT = self.height
        settings.MODEL_DEPTH = self.depth

    def generate_stack(self):
        """
        The stack is an image stack of the same shape as the input data, representing what the current particle
        model looks like. All of the model's particles are shown, except for one: the active_particle. The model stack
        is thus the 'background' of the model; i.e., it shows the particles with fixed parameters, on top of which the
        currently active particle has yet to be added.
        """
        self.stack = np.zeros((self.width, self.height, self.depth))
        for p in self.particles:
            p.update_roi()
            p.update_img()
        for f in range(self.depth):
            for p in self.particles:
                if p == self.active_particle:
                    continue
                if p.b[f]:
                    self.stack[p.roi[0]:p.roi[2], p.roi[1]:p.roi[3], f] += p.img
        self.stack_zsum = sum_z(self.stack)

        return self.stack

    def set_active_particle(self, particle):
        self.active_particle = particle

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
            new_particle = Particle([x, y, size, intensity], settings.HMM.sample(length = self.depth))
            self.particles.append(new_particle)
        if method == "missing density":
            self.set_active_particle(None)
            self.generate_stack()
            residual_density = self.data_zsum - self.stack_zsum
            residual_peak = np.unravel_index(np.argmax(residual_density), (self.width, self.height))
            x = residual_peak[0]
            y = residual_peak[1]
            size = settings.PRIOR_PDF_SIGMA.rvs(1)
            intensity = settings.PRIOR_PDF_INTENSITY.rvs(1)
            new_particle = Particle([x, y, size, intensity], settings.HMM.sample(length = self.depth))
            self.particles.append(new_particle)
        return new_particle

    def remove_particle(self, particle):
        if particle in self.particles:
            self.particles.remove(particle)

    def optimize_active_particle(self):
        bl.map_estimate(self, self.active_particle)
        bl.map_refine(self, self.active_particle)


    def score_active_particle(self):
        P_F, P_N = bl.calculate_particle_probabilities(self, self.active_particle)
        self.active_particle.p_f = P_F
        self.active_particle.p_n = P_N
        return P_F, P_N