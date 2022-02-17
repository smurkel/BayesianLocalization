import numpy as np
from scipy.stats import norm


def particle_pixel_value(params, pixel_index):
    norm_x = norm(loc=params[0], scale=params[2])
    norm_y = norm(loc=params[1], scale=params[2])
    return (norm_x.cdf(pixel_index[0] + 0.5) - norm_x.cdf(pixel_index[0] - 0.5)) *\
           (norm_y.cdf(pixel_index[1] + 0.5) - norm_y.cdf(pixel_index[1] - 0.5)) *\
           params[3]

### OLD ###
def particle_image(imageRadius, params):
    size = imageRadius * 2 + 1
    img = np.zeros((size, size))
    norm_x = norm(loc=params[0], scale=params[2])
    norm_y = norm(loc=params[1], scale=params[2])
    for i in range(-imageRadius, imageRadius + 1):
        for j in range(-imageRadius, imageRadius + 1):
            x = params[0] - i
            y = params[1] - j
            img[i + imageRadius, j + imageRadius] = params[3] * (norm_x.cdf(i + 0.5) - norm_x.cdf(i - 0.5)) * (
                    norm_y.cdf(j + 0.5) - norm_y.cdf(j - 0.5))
    return img

def fp_pixel_value(params, pixel_i, pixel_j):
    norm_x = norm(loc=params[0], scale=params[2])
    norm_y = norm(loc=params[1], scale=params[2])
    return params[3] * (norm_x.cdf(pixel_i + 0.5) - norm_x.cdf(pixel_i - 0.5)) * (
                    norm_y.cdf(pixel_j + 0.5) - norm_y.cdf(pixel_j - 0.5))

def noise(img_size, mu, sigma):
    return np.random.normal(loc = mu, scale = sigma, size = img_size).astype(np.int16)


def timelapse(img_size, n_frames, particle_list, markov_model, noise_mu, noise_sigma):
    size = img_size * 2 + 1
    data = np.zeros((size, size, n_frames))
    for n in range(n_frames):
        data[:, :, n] = np.random.normal(loc = noise_mu, scale = noise_sigma, size = (size, size))
    for p in particle_list:
        states = markov.GetObservationSequence(markov_model, length = n_frames)
        p_on = p.getImage(img_size)
        for n in range(n_frames):
            if states[n]:
                data[:, :, n] += p_on
    return data.astype(np.int16)





