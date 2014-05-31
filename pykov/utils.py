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

def backward_path(observations, pi, A, B, T, N):
    betas = numpy.zeros((T, N))

    """ Initialization """
    betas[T-1, :] = 0 # because we work in log scale we have initialized them with zero
                      # And you might have noticed this step is useless! but it is here
                      # for convenience.

    """ Backward Updates """
    for t in range(T-2 , -1, -1):
        for i in range(N):
            temps = numpy.zeros(N)
            for j in range(N):
                temps[j] = betas[t+1, j] + A[i, j] + B[j, observations[t+1]]
            betas[t, i] = add_logs(temps)

    return betas

def params_to_vector(pi, A, B):
    pi = pi.ravel()
    A = A.ravel()
    B = B.ravel()

    result = numpy.concatenate([pi, A, B])

    return result