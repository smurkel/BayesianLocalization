
import settings
import numpy as np
import matplotlib.pyplot as plt
from util import *
np.seterr(divide='ignore')
from copy import deepcopy

def logprior_a(particle):
    logp_a0 = settings.PRIOR_PDF_X
    logp_a1 = settings.PRIOR_PDF_X
    logp_a2 = settings.PRIOR_PDF_SIGMA.logpdf(particle.a[2])
    logp_a3 = settings.PRIOR_PDF_INTENSITY.logpdf(particle.a[3])
    return logp_a0 + logp_a1 + logp_a2 + logp_a3

def frame_loglikelihood(data_crop_frame, model_crop_frame, particle, particle_state_frame):
    residual_noise = data_crop_frame - model_crop_frame
    if particle_state_frame:
        residual_noise -= particle.img
    return noise_frame_loglikelihood(residual_noise)

def loglikelihood(data_crop, model_crop, particle):
    nx, ny, nf = data_crop.shape
    loglikelihood = 0.0
    for f in range(nf):
        loglikelihood += frame_loglikelihood(data_crop[:, :,f], model_crop[:, :, f], particle, particle.b[f])
    return loglikelihood

def noise_frame_loglikelihood(noise_frame):
    nx, ny = noise_frame.shape
    loglikelihood_frame = 0.0
    for i in range(nx):
        for j in range(ny):
            loglikelihood_frame += settings.PDF_NOISE.logpdf(noise_frame[i, j])
    return loglikelihood_frame

def noise_loglikelihood(noise):
    nx, ny, nf = noise.shape
    loglikelihood = 0.0
    for frame in range(nf):
        loglikelihood += noise_frame_loglikelihood(noise[:, :, frame])
    return loglikelihood

def map_estimate(model, particle):
    print("ESTIMATING MAP")
    def gen_residual_noise_image():
        roi = particle.update_roi()
        data = model.data_zsum[roi[0]:roi[2],roi[1]:roi[3]]
        model_background = model.stack_zsum[roi[0]:roi[2],roi[1]:roi[3]]
        particle_img = particle.update_img() * model.depth # multiplied by model.depth to pretend it is ON in every frame.
        return data - model_background - particle_img

    residual_noise = gen_residual_noise_image()
    current_params = particle.a
    current_score = noise_frame_loglikelihood(residual_noise)

    it = 0
    while it < settings.MAP_ESTIMATE_ITERATIONS:
        p0 = np.random.normal(loc=current_params[0], scale=settings.MAP_PROPOSAL_XY_SIGMA)
        p1 = np.random.normal(loc=current_params[1], scale=settings.MAP_PROPOSAL_XY_SIGMA)
        p2 = np.random.normal(loc=current_params[2], scale=settings.MAP_PROPOSAL_RADIUS_SIGMA)
        p3 = current_params[3]
        proposed_params = [p0, p1, p2, p3]

        particle.a = proposed_params
        residual_noise = gen_residual_noise_image()
        proposal_score = noise_frame_loglikelihood(residual_noise)

        if proposal_score > current_score:
            current_score = proposal_score
            current_params = proposed_params
        else:
            particle.a = current_params

        it+=1

def map_refine(model, particle):
    print("REFINING MAP")
    # FIND SOME INITIAL MARKOV CHAIN BASED ON THE CURRENT PARAMS
    b = list()
    roi = particle.update_roi()
    particle_img = particle.update_img()

    for f in range(model.depth):
        data = model.data[roi[0]:roi[2], roi[1]:roi[3], f]
        model_background = model.stack[roi[0]:roi[2], roi[1]:roi[3], f]
        residual_particle_img = data - model_background
        off_state_residual = noise_frame_loglikelihood(residual_particle_img)
        on_state_residual = noise_frame_loglikelihood(residual_particle_img - particle_img)
        b.append(on_state_residual > off_state_residual)

    particle.b = b

    # FOR A SUBSET OF THE ON-STATE FRAMES, REFINE THE ESTIMATE FOR a BY RANDOM WALK
    # a) grab a subset of frames
    on_state_frames = np.where(particle.b)[0]
    selected_frames = np.random.choice(on_state_frames,
                                       size=min([len(on_state_frames), settings.MAP_REFINE_FRAMES_USED]), replace=False)

    # b) compute current score with that subset

    def score_particle():
        score = 0
        roi = particle.update_roi()
        particle_img = particle.update_img()
        for f in selected_frames:
            data = model.data[roi[0]:roi[2], roi[1]:roi[3], f]
            model_background = model.stack[roi[0]:roi[2], roi[1]:roi[3], f]
            residual = data - model_background - particle_img
            score += noise_frame_loglikelihood(residual)
        score += logprior_a(particle)
        return score

    current_params = particle.a
    current_score = score_particle()
    # c) do the refinement
    it = 0
    PROPOSAL_SIGMA = [settings.MAP_REFINE_XY_SIGMA, settings.MAP_REFINE_XY_SIGMA, settings.MAP_REFINE_RADIUS_SIGMA,
                      settings.MAP_REFINE_INTENSITY_SIGMA]
    while it < settings.MAP_REFINE_ITERATIONS:
        for i in range(4):
            proposed_params = deepcopy(current_params)
            proposed_params[i] += np.random.normal(loc=0, scale=PROPOSAL_SIGMA[i])
            particle.a = proposed_params
            proposal_score = score_particle()
            if proposal_score > current_score:
                current_score = proposal_score
                current_params = proposed_params
            else:
                particle.a = current_params
        it += 1


def maximum_a_posteriori_estimate(model, particle):
    print("START")
    print(particle)
    def gen_residual_noise_image():
        roi = particle.update_roi()
        data = model.data_zsum[roi[0]:roi[2],roi[1]:roi[3]]
        model_background = model.stack_zsum[roi[0]:roi[2],roi[1]:roi[3]]
        particle_img = particle.update_img() * model.depth # multiplied by model.depth to pretend it is ON in every frame.
        return data - model_background - particle_img

    ## STAGE 1 - estimating x and y with intensity fixed
    # Calculate probabilities
    residual_noise = gen_residual_noise_image()
    current_params = particle.a
    current_score = noise_frame_loglikelihood(residual_noise)
    # Note that in a typical MCMC algorithm, the 'score' would be the loglikelihood plus the prior probability of the
    # the parameters. Since the brightness parameter in our model is per-frame, i.e., it corresponds to the
    # observed nr. of photons in a single ON-state frame, and we start out by looking at the summed intensity
    # of the entire dataset/model stack, we initially leave out the prior term for efficiency. Note that the prior
    # probablitity distribution of x and y (a[0] and a[1]) is a constant - it would make no difference in the random
    # walk anyway.

    it = 0
    while it < settings.MAP_ESTIMATE_ITERATIONS:
        p0 = np.random.normal(loc=current_params[0], scale=settings.MAP_PROPOSAL_XY_SIGMA)
        p1 = np.random.normal(loc=current_params[1], scale=settings.MAP_PROPOSAL_XY_SIGMA)
        p2 = np.random.normal(loc=current_params[2], scale=settings.MAP_PROPOSAL_RADIUS_SIGMA)
        p3 = current_params[3]
        proposed_params = [p0, p1, p2, p3]

        particle.a = proposed_params
        residual_noise = gen_residual_noise_image()
        proposal_score = noise_frame_loglikelihood(residual_noise)

        if proposal_score > current_score:
            current_score = proposal_score
            current_params = proposed_params
        else:
            particle.a = current_params

        it+=1

    print("INITIAL ESTIMATE")
    print(particle)
    # FIND SOME INITIAL MARKOV CHAIN BASED ON THE CURRENT PARAMS

    b = list()
    roi = particle.update_roi()
    particle_img = particle.update_img()

    for f in range(model.depth):
        data = model.data[roi[0]:roi[2], roi[1]:roi[3], f]
        model_background = model.stack[roi[0]:roi[2], roi[1]:roi[3], f]
        residual_particle_img = data - model_background
        off_state_residual = noise_frame_loglikelihood(residual_particle_img)
        on_state_residual = noise_frame_loglikelihood(residual_particle_img - particle_img)
        b.append(on_state_residual > off_state_residual)

    particle.b = b
    print("STATE SEQUENCE")
    print(particle.b)

    # FOR A SUBSET OF THE ON-STATE FRAMES, REFINE THE ESTIMATE FOR a BY RANDOM WALK
    # a) grab a subset of frames
    on_state_frames = np.where(particle.b)[0]
    print("SELECTED FRAMES")
    print(on_state_frames)
    selected_frames = np.random.choice(on_state_frames, size=min([len(on_state_frames), settings.MAP_REFINE_FRAMES_USED]), replace=False)
    # b) compute current score with that subset

    def score_particle():
        score = 0
        roi = particle.update_roi()
        particle_img = particle.update_img()
        for f in selected_frames:
            data = model.data[roi[0]:roi[2], roi[1]:roi[3], f]
            model_background = model.stack[roi[0]:roi[2], roi[1]:roi[3], f]
            residual = data - model_background - particle_img
            score += noise_frame_loglikelihood(residual) + logprior_a(particle)
        return score

    current_params = particle.a
    current_score = score_particle()
    print("Starting value:", current_params)
    # c) do the refinement
    it = 0
    PROPOSAL_SIGMA = [settings.MAP_REFINE_XY_SIGMA, settings.MAP_REFINE_XY_SIGMA, settings.MAP_REFINE_RADIUS_SIGMA, settings.MAP_REFINE_INTENSITY_SIGMA]
    while it < settings.MAP_REFINE_ITERATIONS:
        for i in range(4):
            proposed_params = deepcopy(current_params)
            proposed_params[i] += np.random.normal(loc = 0, scale = PROPOSAL_SIGMA[i])
            particle.a = proposed_params
            proposal_score = score_particle()
            if proposal_score > current_score:
                print("Found better params:", particle.a)
                current_score = proposal_score
                current_params = proposed_params
            else:
                particle.a = current_params
        it += 1


def hessian(model, particle):
    def f(x):
        particle.a = x
        roi = particle.update_roi()
        particle.update_img()
        data_crop = model.data[roi[0]:roi[2],roi[1]:roi[3], :]
        model_crop = model.stack[roi[0]:roi[2],roi[1]:roi[3], :]
        return loglikelihood(data_crop, model_crop, particle)

    original_params = deepcopy(particle.a)

    hessian_matrix = np.zeros((4, 4))
    x = np.zeros(4)
    x[0] = particle.a[0]
    x[1] = particle.a[1]
    x[2] = particle.a[2]
    x[3] = particle.a[3]

    for i in range(4):
        dxi = np.zeros(4)
        dxi[i] = settings.NUMERICAL_DERIVATIVE_DELTA[i]
        for j in range(4):
            dxj = np.zeros(4)
            dxj[j] = settings.NUMERICAL_DERIVATIVE_DELTA[j]
            temp = (f(x + dxi + dxj) - f(x + dxj - dxi) - f(x - dxj + dxi) + f(x + dxj - dxi)) / (
                        2 * settings.NUMERICAL_DERIVATIVE_DELTA[i] * settings.NUMERICAL_DERIVATIVE_DELTA[j])
            hessian_matrix[i, j] = temp
    particle.a = original_params
    particle.update_roi()
    particle.update_img()
    return hessian_matrix

def calculate_particle_probabilities(model, particle):
    print("SCORING PARTICLE")
    def log_average_p_from_log_p_list(logp):
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

    # P(D|F) ~= [Laplace approximation for integral over a] + [Sampling in b space]
    # integral over a
    roi = particle.update_roi()
    data_crop = model.data[roi[0]:roi[2], roi[1]:roi[3], :]
    model_crop = model.stack[roi[0]:roi[2], roi[1]:roi[3], :]
    data_minus_model = data_crop - model_crop
    map_logprobability = loglikelihood(data_crop, model_crop, particle)
    hessian_matrix = hessian(model, particle)
    print(hessian_matrix)
    laplace_approximation = map_logprobability #+ np.log(2 * np.pi ** 2) - np.log(np.sqrt(np.abs(np.linalg.det(hessian_matrix)))) # is this correct ..?
    # sum over b space
    llh_off_state = list()
    sample_llh = list()
    for f in range(model.depth):
        llh_off_state.append(noise_frame_loglikelihood(data_minus_model[:, :, f]))
    for s in range(settings.MARKOV_SAMPLES):
        state_sequence = settings.HMM.sample(length = model.depth)
        log_p = settings.HMM.log_probability(state_sequence)
        for f in range(model.depth):
            if state_sequence[f]:
                log_p += noise_frame_loglikelihood(data_minus_model[:, :, f] - particle.img)
            else:
                log_p += llh_off_state[f]
            sample_llh.append(log_p)
    avg_log_p = log_average_p_from_log_p_list(sample_llh)
    state_space_volume = model.depth * np.log(2)
    markov_approximation = np.log(state_space_volume) + avg_log_p - np.log(settings.MARKOV_SAMPLES)

    llh_particle_model = laplace_approximation + markov_approximation
    llh_null_model = 0
    for frame_llh in llh_off_state:
        llh_null_model += frame_llh
    return llh_particle_model, llh_null_model
