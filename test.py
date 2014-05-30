import numpy
from pykov import markov, evaluation, decoding, learning


def main():
    pi = numpy.array([1/3., 1/3., 1/3.])
    A = numpy.array([
            [1/3., 1/3., 1/3.],
            [1/3., 1/3., 1/3.],
            [1/3., 1/3., 1/3.]
        ])
    B = numpy.array([
            [1/2., 1/2.],
            [3/4., 1/4.],
            [1/4., 3/4.]
        ])

    O = numpy.array([0, 0, 1, 1, 0])
    Q = numpy.array([1, 1, 2, 2, 1])

    model = markov.HMM(pi, A, B)
    print 'Eval', evaluation.evaluate(O, model)
    print 'Eval with state', evaluation.evaluate(O, model, Q)
    print 'Decode', decoding.viterbi(O, model)

    pi, A, B = learning.baum_welch(O, 3, 2)


if __name__ == '__main__':
    main()