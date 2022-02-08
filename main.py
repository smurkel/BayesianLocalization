from util import *

import simulator
import matplotlib.pyplot as plt
from scipy.stats import lognorm, uniform, norm

np.seterr(divide = 'ignore')

"""SET IMAGE AND MICROSCOPE PARAMETERS"""
PSF_INTENSITY_MU = 1000
PSF_INTENSITY_LOGNORM_S = 0.7
PSF_RADIUS_MU = 2.0 # mean of the gaussian psf's standard deviation in units of pixels.
PSF_RADIUS_LOGNORM_S = 0.1

NOISE_MEAN = 0
NOISE_SIGMA = 2.0

IMG_SIZE = 4
"""DEFINE MCMC BEHAVIOUR"""
MCMC_ITERATIONS = 100
MCMC_BURN = 0
MCMC_XY_SIGMA = 0.01
MCMC_RADIUS_SIGMA = 0.01
MCMC_INTENSITY_SIGMA = 5.0
"""DEFINE THE NOISE MODEL - IN PRACTICE, THE NOISE MODEL WILL BE DERIVED FROM IMAGES; IT IS A GAUSSIAN PDF WITH MEAN AND STD TAKEN FROM BACKGROUND PIXEL INTENSITY."""
PDF_NOISE = norm(loc = NOISE_MEAN, scale = NOISE_SIGMA)
"""DEFINE PRIOR PROBABILITIES FOR X, Y, SIGMA, and INTENSITY."""
PDF_X = uniform(loc = -IMG_SIZE, scale = 2 * IMG_SIZE)
PDF_Y = uniform(loc = -IMG_SIZE, scale = 2 * IMG_SIZE)
PDF_SIGMA = lognorm(s = PSF_RADIUS_LOGNORM_S, scale = PSF_RADIUS_MU)
PDF_INTENSITY = lognorm(s = PSF_INTENSITY_LOGNORM_S, scale = PSF_INTENSITY_MU)
#################

def prior_x(val):
    # x prior probability is: flat probability, but it is somewhere in the image.
    return PDF_X.pdf(val)

def prior_y(val):
    # y prior probability is: flat probability, but it is somewhere in the image.
    return PDF_Y.pdf(val)

def prior_sigma(val):
    # sigma prior probability is: log normal distribution
    return PDF_SIGMA.pdf(val)

def prior_intensity(val):
    # intensity prior probability is lognormal.
    return PDF_INTENSITY.pdf(val)

def mcmc_posterior_pdf(data):
    """This can be improved by starting the random walk not at a random sample from the prior distributions, but instead near the mode of the likelihood. Also: the data should be cropped such that the mode of the likelihood
    corresponds to the mean of the x and y prio dsitributions; i.e. crop with probable particle position in center.
    Useful lecture: https://www.youtube.com/watch?v=pHsuIaPbNbY&ab_channel=MLSSIceland2014"""
    parameter_samples = list()
    x = (PDF_X.rvs(1), PDF_Y.rvs(1), PDF_SIGMA.rvs(1), PDF_INTENSITY.rvs(1))
    x = (0.0, 0.0, 2.0, 1000.0)
    p_x = likelihood(data, x) + prior_probability(*x)

    max_log = p_x
    for i in range(MCMC_ITERATIONS):
        if i % 10 == 0:
            print("Iteration {}".format(i))
        if p_x > max_log:
            max_log = p_x
        # Step
        dx0 = np.random.normal(loc=0, scale=MCMC_XY_SIGMA)
        dx1 = np.random.normal(loc=0, scale=MCMC_XY_SIGMA)
        dx2 = np.random.normal(loc=0, scale=MCMC_RADIUS_SIGMA)
        dx3 = np.random.normal(loc=0, scale=MCMC_INTENSITY_SIGMA)

        # Perform step
        y = (x[0]+dx0, x[1]+dx1, x[2]+dx2, x[3]+dx3)

        # Calculate new score
        p_prior = prior_probability(*y)
        p_data = likelihood(data, y)
        p_y = p_data + p_prior
        R = p_y - p_x

        if R > 0:
            parameter_samples.append([y[0], y[1], y[2], y[3], p_y])
            x = y
            p_x = p_y
        else:
            if np.random.uniform(0, 1, 1) < np.exp(R):
               parameter_samples.append([y[0], y[1], y[2], y[3], p_y])
               x = y
               p_x = p_y
            else:
                parameter_samples.append([x[0], x[1], x[2], x[3], p_x])

    return parameter_samples[MCMC_BURN:]

def prior_probability(x, y, sigma, intensity):
    return np.log(prior_x(x)) + np.log(prior_y(y)) + np.log(prior_sigma(sigma)) + np.log(prior_intensity(intensity))

def likelihood(data, parameters):
    # Data is an image with a gaussian spot and noise.
    # Parameters are of a model that describes a Gaussian spot.
    # Here:
    # 1 - 'simulate' the data according to parameters.
    model = simulator.particle(IMG_SIZE, *parameters)
    # 2 - subtract model from data. what remains is noise.
    noise = data - model
    # 3 - evaluate the probability of observing this noise given the noise model
    log_probability = eval_noise(noise)
    return log_probability


def eval_noise(noise):
    N = noise.shape[0]
    log_probability = 0.0
    for i in range(N):
        for j in range(N):
                log_probability += np.log(PDF_NOISE.pdf(noise[i,j]))
    return log_probability

ground_truth = (0.0, 0.0 ,2.0, 1000.0)
data = simulator.particle(IMG_SIZE, *ground_truth) + simulator.noise(IMG_SIZE, 0, 1.0)

prms = mcmc_posterior_pdf(data)
prms = np.asarray(prms)
n_bins = 25
x = prms[:, 0]
y = prms[:, 1]
s = prms[:, 2]
i = prms[:, 3]
p = prms[:, 4]


plt.subplot(5,1,1)
plt.hist(x, bins = n_bins)
plt.title("X")
plt.subplot(5,1,2)
plt.hist(y, bins = n_bins)
plt.title("Y")
plt.subplot(5,1,3)
plt.hist(s, bins = n_bins)
plt.title("S")
plt.subplot(5,1,4)
plt.hist(i, bins = n_bins)
plt.title("I")
plt.subplot(5,1,5)
plt.hist(p, bins = n_bins)
plt.title("P")

plt.figure(2)
n = np.shape(x)[0]
c = np.zeros((n, 4))
c[:, 0] = np.linspace(0, 1, n)
c[:, 1] = np.linspace(1, 0, n)
c[:, 3] = np.zeros(n) + 0.5
plt.imshow(data)

plt.plot(y + IMG_SIZE, x + IMG_SIZE, linewidth = 1, color = (0.0, 0.0, 0.0))
for j in range(n):
    plt.plot(y[j] + IMG_SIZE, x[j] + IMG_SIZE, markerfacecolor = c[j], marker = 'o', markeredgecolor = (1.0, 1.0, 1.0), markersize = 5)

plt.figure(3)
plt.subplot(1,3,1)
plt.imshow(data)
plt.subplot(1,3,2)
final_model = simulator.particle(IMG_SIZE, x[-1], y[-1], s[-1], i[-1])
plt.imshow(final_model)
plt.title("Final model")
plt.subplot(1,3,3)
plt.imshow(simulator.particle(IMG_SIZE,x[0],y[0],s[0],i[0]))
plt.title("First non burn model")

plt.figure(4)
plt.subplot(1,4,1)
plt.plot(x)
plt.subplot(1,4,2)
plt.plot(y)
plt.subplot(1,4,3)
plt.plot(s)
plt.subplot(1,4,4)
plt.plot(i)

plt.figure(5)
plt.plot(p)
plt.title("Estimated model accuracy vs. iteration nr.")
plt.show()