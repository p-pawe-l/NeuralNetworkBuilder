import numpy

class relu_activation_function: 
        @staticmethod
        def relu(x):
                return numpy.maximum(0, x)
        
        @staticmethod
        def relu_derivative(x):
                dx = numpy.array(x, copy = True)
                dx[dx <= 0] = 0
                dx[dx > 0] = 1
                return dx
        
class sigmoid_activation_function:
        @staticmethod
        def sigmoid(x):
                return (1 / (1 + numpy.exp(-x)))
        
        @staticmethod
        def sigmoid_derivative(x):
                return sigmoid_activation_function.sigmoid(x) * (1 - sigmoid_activation_function.sigmoid(x))