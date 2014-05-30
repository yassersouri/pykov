import numpy
from utils import * 

INITIALIZATION_MODE = "random"
MAX_ITERS = 10
CONV_CRIT = 0.00001

def baum_welch(observations, N, M):
    """
    N is the number of hidden states, M is the number of different possible observations
    """
    T = observations.shape[0]

    # initialize variables
    pi, A, B = initialize_variables(N, M, mode=INITIALIZATION_MODE)
    print 'initialization: Done'

    converge = False

    iter_num = 0
    while not converge:
        iter_num += 1

        # iterate
        new_pi, new_A, new_B = EM_iterate(observations, N, M, T, pi, A, B)
        print 'EM Iteration: %d' % iter_num

        # check convergence
        converge = did_converge(pi, A, B, new_pi, new_A, new_B)
        if iter_num > MAX_ITERS:
            converge = True

    # return variables
    return pi, A, B

def EM_iterate(observations, N, M, T, pi, A, B):
    

def did_converge(pi, A, B, new_pi, new_A, new_B):
    if numpy.linalg.norm(pi - new_pi) > CONV_CRIT:
        return False
    if numpy.linalg.norm(A - new_A) > CONV_CRIT:
        return False
    if numpy.linalg.norm(B - new_B) > CONV_CRIT:
        return False
    return True

def initialize_variables(N, M, mode="random"):
    """
    initializes the model parameters
    It can perform in two modes, "random", "equal"
    """
    A = numpy.zeros((N, N))
    B = numpy.zeros((N, M))
    pi = numpy.zeros(N)

    if mode == "random":
        pi = numpy.random.random(N)
        pi = pi / pi.sum()

        for i in range(N):
            tempA = numpy.random.random(N)
            A[i, :] = tempA / tempA.sum()

            tempB = numpy.random.random(M)
            B[i, :] = tempB / tempB.sum()
    elif mode == "equal":
        pi[:] = 1. / N
        A[:] = 1. / N
        B[:] = 1. / M
    else:
        raise Exception("invalid mode: %s" % mode)

    return pi, A, B