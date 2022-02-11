import numpy as np
from scipy.stats import norm
import markov

class Particle:
    def __init__(self, xpos = 0.0, ypos = 0.0, sigma = 2.0, intensity = 1000.0):
        self.xpos = xpos
        self.ypos = ypos
        self.sigma = sigma
        self.intensity = intensity

    def getImage(self, img_size):
        size = img_size * 2 + 1
        img = np.zeros((size, size))
        psf = norm()
        norm_x = norm(loc=self.xpos, scale=self.sigma)
        norm_y = norm(loc=self.ypos, scale=self.sigma)
        for i in range(-img_size, img_size + 1):
            for j in range(-img_size, img_size + 1):
                x = self.xpos - i
                y = self.ypos - j
                img[i + img_size, j + img_size] = self.intensity * (norm_x.cdf(i + 0.5) - norm_x.cdf(i - 0.5)) * (
                            norm_y.cdf(j + 0.5) - norm_y.cdf(j - 0.5))

        return img.astype(np.int16)

def noise(img_size, mu, sigma):
    size = img_size * 2 + 1
    return np.random.normal(loc = mu, scale = sigma, size = (size, size)).astype(np.int16)


def timelapse(img_size, n_frames, particle_list, markov_model, noise_mu, noise_sigma):
    size = img_size * 2 + 1
    data = np.zeros((size, size, n_frames))
    for n in range(n_frames):
        data[:, :, n] = np.random.normal(loc = noise_mu, scale = noise_sigma, size = (size, size))
    for p in particle_list:
        states = markov.GetObservationSequence(markov_model, length = n_frames)
        print(states)
        p_on = p.getImage(img_size)
        for n in range(n_frames):
            if states[n]:
                data[:, :, n] += p_on
    return data.astype(np.int16)



