#pykov - python implementation of inference and learning in HMM [Under development]

Here I try to implement inference (evaluation and decoding) and learning  algorithms in python for Hidden Markov Models. These algorithms are fairly easy dynamic programming ones.

##Dependencies

* python 2.7.*
* numpy >= 1.8.0

##Features

###log-scale

All the computations are done in log-scale for more stability and robustness. This way really small probability values can be computed effectively.

###static observation matrix

The matrix `B`, the probability of each observation in each state is static and cannot change. For dynamic matrix `B` one can make small edits in the evaluation algorithm. But I suggest going for a `sum-product` algorithm in more complicated cases.

##TODO

* ~~Implement evaluation using Forward algorithm.~~
* ~~Implement decoding using Viterbi algotithm.~~
* ~~Implement the algorithms using the logarithm scale to make it possible to calculate very large or very small numbers.~~
* Implement learning.
* Test the implementations.
* Covert to a python package.
* Create some documentation.
* Put the package on pypi.
* Implement some actually useful demos like POS tagging.

##Reference
You can read more about HMM algorithms in many references including [Wikipedia](http://en.wikipedia.org/wiki/Hidden_Markov_model), [paper by Rabiner](http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf) or chapter 17 of [Murphy's Book](http://www.cs.ubc.ca/~murphyk/MLbook/).
