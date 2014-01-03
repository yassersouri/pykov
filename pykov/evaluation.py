import numpy

def evaluate(observation, model, states=None):
    """
    TODO
    If you want the real evaluation (you don't know the states) do not set the states.
    Implements the forward algorithm for evaluation of an observation sequence given the HMM model.
    """
    if states is None:
        N = model.N
        T = observation.shape[0]

        alphas = numpy.zeros((T,N))
        
        """ Initialization """
        alphas[0, :] = numpy.dot(model.pi, model.B[:, observation[0]])

        """ Forward Updates """
        for t in range(1, T):
            alphas[t, :] = numpy.dot(alphas[t-1, :], model.A) # matrix multiply
            alphas[t, :] = alphas[t, :] * model.B[:, observation[t]]

        """ Termination """
        return alphas[T-1, :].sum()

    else:
        pass