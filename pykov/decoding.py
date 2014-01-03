import numpy

def viterbi(observation, model):
    """
    TODO
    Implements the viterbi algorithm for decoding of an observation sequence
    """
    N = model.N
    T = observation.shape[0]
    q_star = numpy.zeros(T)

    delta = numpy.zeros((T, N))
    psi = numpy.zeros((T, N))

    """ Initialization """
    delta[0, :] = model.pi * model.B[:, observation[0]]

    """ Forward Updates """
    for t in range(1, T):
        temp = model.A * delta[t-1, :]
        psi[t, :] = numpy.argmax(temp, axis=1)
        delta[t, :] = numpy.max(temp, axis=1) * model.B[:, observation[t]]

    """ Termination """
    q_star[T-1] = numpy.argmax(delta[T-1, :])
    p_star = numpy.max(delta[T-1, :])

    """ Backward state sequence """
    for t in range(T-2, -1, -1):
        q_star[t] = psi[t+1, q_star[t+1]]

    return q_star, p_star