import numpy
import math
from utils import *

def evaluate(observation, model, states=None, log=False):
    """
    TODO
    If you want the real evaluation (you don't know the states) do not set the states.
    Implements the forward algorithm for evaluation of an observation sequence given the HMM model.

    If `log` is `True`, then it results `log(p(observation|model))` instead of the `p(observation|model)` itself.
    """
    N = model.N
    T = observation.shape[0]
    A = numpy.log(model.A)
    B = numpy.log(model.B)

    if states is None:
        alphas = numpy.zeros((T,N))
        
        """ Initialization """
        alphas[0, :] = numpy.log(model.pi) + B[:, observation[0]]

        """ Forward Updates """
        for t in range(1, T):
            for j in range(N):
                temps = numpy.zeros(N)
                for i in range(N):
                    temps[i] = alphas[t-1, i] + A[i, j]
                alphas[t, j] = add_logs(temps) + B[j, observation[t]]

        """ Termination """
        result = add_logs(alphas[T-1, :])
        if log:
            return result
        else:
            return math.exp(result)

    else:
        result = 0
        for i in range(T):
            result += B[states[i], observation[i]]

        if log:
            return result
        else:
            return math.exp(result)