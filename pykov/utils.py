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