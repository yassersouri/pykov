#pykov - python implementation of inference in HMM [Under development]

Here I try to implement inference algorithms (evaluation and decoding) in python for Hidden Markov Models. These algorithms are fairly easy dynamic programming ones. Maybe later I get to implement the learning part.

##Dependencies

* python 2.7.*
* numpy >= 1.8.0

##TODO

* Implement evaluation using Forward-Backward algorithm.
* Implement decoding using Viterbi algotithm.
* Covert to a python package.
* Create some documentation.
* Put the package on pypi.
* Implement learning.
* Implement some actually useful demos like POS tagging.

##Reference
You can read more about HMM algorithms in many references including [wikipedia](http://en.wikipedia.org/wiki/Hidden_Markov_model), [Paper by Rabiner](http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf) or Chapter 17 of [Murphy's Book](http://www.cs.ubc.ca/~murphyk/MLbook/).
