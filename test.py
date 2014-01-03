import numpy
from pykov import markov, evaluation


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

    model = markov.HMM(pi, A, B)
    print evaluation.evaluate(O, model)


if __name__ == '__main__':
    main()