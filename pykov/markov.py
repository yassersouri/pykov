class HMM(object):
    """
    TODO
    """
    
    def __init__(self, pi, A, B):
        """
        TODO

        N: the number of hidden states.
        A: the state transition matrix.
        B: the observation probability distribution.
        pi: the initial probabilities.
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.N = A.shape[0]
        self.check_model()

    def check_model(self):
        """
        Checks the model to see if the parameters are correct.

        It checks the dimention of pi, A and B matrices.
        """
        pass