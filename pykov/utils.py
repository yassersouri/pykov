import numpy

def add_logs(to_Add):
    """
    Add to gather some numbers which are in log scale.
    Remember this is sum! and log!
    """
    maximum = numpy.max(to_Add)
    to_Add = to_Add - maximum
    to_Add = numpy.exp(to_Add)
    result = numpy.log(to_Add.sum()) + maximum
    return result

def forward_path(observations, pi, A, B, T, N):
    alphas = numpy.zeros((T,N))
    
    """ Initialization """
    alphas[0, :] = pi + B[:, observations[0]]

    """ Forward Updates """
    for t in range(1, T):
        for j in range(N):
            temps = numpy.zeros(N)
            for i in range(N):
                temps[i] = alphas[t-1, i] + A[i, j]
            alphas[t, j] = add_logs(temps) + B[j, observations[t]]

    return alphas