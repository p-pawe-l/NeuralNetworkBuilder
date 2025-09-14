import numpy

class ReluActivationFunction:
        @staticmethod
        def activation(x):
                return numpy.maximum(0, x)
        
        @staticmethod
        def derivative(x):
                dx = numpy.array(x, copy = True)
                dx[dx <= 0] = 0
                dx[dx > 0] = 1
                return dx
        
class SigmoidActivationFunction:
        @staticmethod
        def activation(x):
                return 1 / (1 + numpy.exp(-x))
        
        @staticmethod
        def derivative(x):
                return SigmoidActivationFunction.activation(x) * (1 - SigmoidActivationFunction.activation(x))