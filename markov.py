from pomegranate import *

def GetHMM(p_on, p_off, p_bleach, p_false_emission, p_initial = None):
    global model
    model = HiddenMarkovModel('FPModel')
    d1 = DiscreteDistribution({True: 1.0 - p_false_emission, False: p_false_emission}) # True = LIGHT, False = DARK
    d2 = DiscreteDistribution({False: 1.0 - p_false_emission, True: p_false_emission})
    d3 = DiscreteDistribution({False: 1.0 - p_false_emission, True: p_false_emission})
    s1 = State(d1, name = "b0")
    s2 = State(d2, name = "b1")
    s3 = State(d3, name = "b2")
    model.add_states(s1, s2, s3)
    if p_initial:
        model.add_transition(model.start, s1, p_initial[0])
        model.add_transition(model.start, s2, p_initial[1])
        model.add_transition(model.start, s3, p_initial[2])
    else:
        model.add_transition(model.start, s1, p_on / (p_off + p_on))
        model.add_transition(model.start, s2, p_off / (p_on + p_off))
        model.add_transition(model.start, s3, 0.0)
    model.add_transition(s1, s1, 1.0 - p_off)
    model.add_transition(s1, s2, p_off)
    model.add_transition(s2, s1, p_on)
    model.add_transition(s2, s2, 1.0 - p_on - p_bleach)
    model.add_transition(s2, s3, p_bleach)
    model.bake()
    return model

def GetLogProbability(model, observationSequence):
    return model.log_probability(observationSequence)

def GetObservationSequence(model, length):
    return model.sample(length = length)
