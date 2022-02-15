from util import *

import simulator
import modelparams as prm
np.seterr(divide = 'ignore')

"""SET IMAGE AND MICROSCOPE PARAMETERS"""


def prior_x(val):
    # x prior probability is: flat probability, but it is somewhere in the image.
    return prm.PDF_X.pdf(val)

def prior_y(val):
    # y prior probability is: flat probability, but it is somewhere in the image.
    return prm.PDF_Y.pdf(val)

def prior_sigma(val):
    # sigma prior probability is: log normal distribution
    return prm.PDF_SIGMA.pdf(val)

def prior_intensity(val):
    # intensity prior probability is lognormal.
    return prm.PDF_INTENSITY.pdf(val)

def estimate_map(data, initial_suggestion = None):
    """This can be improved by starting the random walk not at a random sample from the prior distributions, but instead near the mode of the likelihood. Also: the data should be cropped such that the mode of the likelihood
    corresponds to the mean of the x and y prio dsitributions; i.e. crop with probable particle position in center.
    Useful lecture: https://www.youtube.com/watch?v=pHsuIaPbNbY&ab_channel=MLSSIceland2014"""
    parameter_samples = list()
    if initial_suggestion:
        x = initial_suggestion
    else:
        x = (prm.PDF_X.rvs(1), prm.PDF_Y.rvs(1), prm.PDF_SIGMA.rvs(1), prm.PDF_INTENSITY.rvs(1))
    p_x = likelihood(data, x) + prior_probability(*x)
    MAP = x
    max_log = p_x
    for i in range(prm.MAP_ESTIMATE_ITERATIONS):
        print("ESTIMATING MAP iteration {}/{}".format(i, prm.MAP_ESTIMATE_ITERATIONS))
        if p_x > max_log:
            MAP = x
            max_log = p_x
        # Step
        dx0 = np.random.normal(loc=0, scale=prm.MAP_PROPOSAL_XY_SIGMA)
        dx1 = np.random.normal(loc=0, scale=prm.MAP_PROPOSAL_XY_SIGMA)
        dx2 = np.random.normal(loc=0, scale=prm.MAP_PROPOSAL_RADIUS_SIGMA)
        dx3 = np.random.normal(loc=0, scale=prm.MAP_PROPOSAL_INTENSITY_SIGMA)

        # Perform step
        y = (x[0]+dx0, x[1]+dx1, x[2]+dx2, x[3]+dx3)

        # Calculate new score
        p_y = likelihood(data, y) + prior_probability(*y)
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

    return MAP

def refine_map(data, map):
    x = map
    Px = likelihood(data, x)  # prior_probability not checked, because only xpos and ypos change which are uniformly distributed.
    for i in range(prm.MAP_REFINE_ITERATIONS):
        dpos = np.random.normal(loc=0, scale=prm.MAP_REFINE_XY_SIGMA)
        if (i % 2) == 0:
            y = (x[0] + dpos, x[1], x[2], x[3])
        else:
            y = (x[0], x[1] + dpos, x[2], x[3])
        Py = likelihood(data, y) # prior_probability not checked, because only xpos and ypos change which are uniformly distributed.
        if Py > Px:
            x = y
            Px = Py
    return x


def prior_probability(x, y, sigma, intensity):
    return np.log(prior_x(x)) + np.log(prior_y(y)) + np.log(prior_sigma(sigma)) + np.log(prior_intensity(intensity))

def likelihood(data, parameters):
    # Data is an image with a gaussian spot and noise.
    # Parameters are of a model that describes a Gaussian spot.
    # Here:
    # 1 - 'simulate' the data according to parameters.
    model = simulator.particle_image(prm.IMG_SIZE, parameters)
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
                log_probability += np.log(prm.PDF_NOISE.pdf(noise[i,j]))
    return log_probability

def eval_noise_stack(noise):
    w, h, f = noise.shape
    log_p = 0
    for i in range(f):
        log_p += eval_noise(noise[:, :, i])
    return log_p

def hessian(data, params):
    def f(x):
        return likelihood(data, (x[0],x[1],x[2],x[3])) + prior_probability(x[0],x[1],x[2],x[3])
    Hessian = np.zeros((4,4))
    # Second derivatives first.
    x = np.zeros(4)
    x[0] = params[0]
    x[1] = params[1]
    x[2] = params[2]
    x[3] = params[3]

    for i in range(4):
        dxi = np.zeros(4)
        dxi[i] = prm.NUMERICAL_DERIVATIVE_DELTA[i]
        for j in range(4):
            dxj = np.zeros(4)
            dxj[j] = prm.NUMERICAL_DERIVATIVE_DELTA[j]
            temp = (f(x + dxi + dxj) - f(x + dxj - dxi) - f(x - dxj + dxi) + f(x + dxj - dxi) ) / (2 * prm.NUMERICAL_DERIVATIVE_DELTA[i] * prm.NUMERICAL_DERIVATIVE_DELTA[j])
            Hessian[i, j] = temp
    return Hessian

def laplace_approximation_value(data, MAP):
    MAP_logprob = likelihood(data, MAP) + prior_probability(*MAP)
    H = hessian(data, MAP)
    detH = -np.linalg.det(H)
    log_value = MAP_logprob + np.log(2 * np.pi ** (4/2)) - np.log(np.sqrt((-np.linalg.det(H))))
    return float(log_value)

def markov_approximation_value(data, MAP, markovModel):
    width, height, num_frames = data.shape
    log_likelihood_off_state = list()
    log_p_samples = list()
    for f in range(num_frames):
        log_likelihood_off_state.append(eval_noise(data[:, :, f]))
    for s in range(prm.MARKOV_SAMPLES):
        state_sequence = markovModel.sample(length = num_frames)
        log_p = markovModel.log_probability(state_sequence)
        for f in range(num_frames):
            if state_sequence[f]: # State is ON
                log_p += likelihood(data[:, :, f], MAP)
            else:
                log_p += log_likelihood_off_state[f]
        log_p_samples.append(log_p)
    avg_log_p = log_avg_probability_from_logprobabilities(log_p_samples)
    state_space_volume = num_frames * np.log(2)
    return np.log(state_space_volume) + avg_log_p - np.log(prm.MARKOV_SAMPLES)

def log_avg_probability_from_logprobabilities(logp):
    maxlog = logp[0]
    for i in logp:
        if i > maxlog:
            maxlog = i
    offset_log = list()
    for i in logp:
        offset_log.append(maxlog - i)
    val = 0
    for i in offset_log:
        val += np.exp(-i)
    log_avg_p = np.log(val) + maxlog
    return log_avg_p

def estimate_state(data, MAP):
    # For every frame, estimate whether a particle at the MAP was ON or OFF, based on the relative probability of P(D|F)/P(D|N)
    width, height, n_frames = data.shape
    state = list()
    for f in range(n_frames):
        pf = likelihood(data[:, :, f], MAP)
        pn = eval_noise(data)
        state.append(pf > pn)
    return state