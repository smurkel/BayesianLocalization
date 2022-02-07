import numpy as np
from scipy.stats import norm


def particle(img_size, xpos, ypos, sigma, intensity):
    size = img_size * 2 + 1
    img = np.zeros((size,size))
    psf = norm()
    norm_x = norm(loc = xpos, scale = sigma)
    norm_y = norm(loc = ypos, scale = sigma)
    for i in range(-img_size, img_size + 1):
        for j in range(-img_size, img_size + 1):
            x = xpos - i
            y = ypos - j
            img[i + img_size, j + img_size] = intensity * (norm_x.cdf(i + 0.5) - norm_x.cdf(i - 0.5)) * (norm_y.cdf(j + 0.5) - norm_y.cdf(j - 0.5))
    return img.astype(np.int16)

def noise(img_size, mu, sigma):
    size = img_size * 2 + 1
    return np.random.normal(loc = mu, scale = sigma, size = (size, size)).astype(np.int16)



