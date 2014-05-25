import numpy
import math

def viterbi(observation, model, log=False):
    """
    TODO
    Implements the viterbi algorithm for decoding of an observation sequence

    If `log` is `True`, then it results `log(p(.))` instead of the `p(.)` itself.
    """
    N = model.N
    T = observation.shape[0]
    q_star = numpy.zeros(T)

    delta = numpy.zeros((T, N))
    psi = numpy.zeros((T, N))

    A = numpy.log(model.A)
    B = numpy.log(model.B)

    """ Initialization """
    delta[0, :] = numpy.log(model.pi) + B[:, observation[0]]

    """ Forward Updates """
    for t in range(1, T):
        temp = numpy.zeros((N, N))
        for i in range(N):
            temp[i, :] = A[i, :] + delta[t-1, :]

        psi[t, :] = numpy.argmax(temp, axis=1)
        delta[t, :] = numpy.max(temp, axis=1) + B[:, observation[t]]

    """ Termination """
    q_star[T-1] = numpy.argmax(delta[T-1, :])
    p_star = numpy.max(delta[T-1, :])

    """ Backward state sequence """
    for t in range(T-2, -1, -1):
        q_star[t] = psi[t+1, q_star[t+1]]

    if not log:
        p_star = math.exp(p_star)

    return q_star, p_star