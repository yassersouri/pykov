import numpy
import math
from utils import add_logs, forward_path

def evaluate(observations, model, states=None, log=False):
    """
    TODO
    If you want the real evaluation (you don't know the states) do not set the states.
    Implements the forward algorithm for evaluation of an observation sequence given the HMM model.

    If `log` is `True`, then it results `log(p(observations|model))` instead of the `p(observations|model)` itself.
    """
    N = model.N
    T = observations.shape[0]
    A = numpy.log(model.A)
    B = numpy.log(model.B)

    if states is None:
        alphas = forward_path(observations, numpy.log(model.pi), A, B, T, N)

        """ Termination """
        result = add_logs(alphas[T-1, :])
        if log:
            return result
        else:
            return math.exp(result)

    else:
        result = 0
        for i in range(T):
            result += B[states[i], observations[i]]

        if log:
            return result
        else:
            return math.exp(result)