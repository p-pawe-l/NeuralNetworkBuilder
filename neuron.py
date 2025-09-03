import numpy as np
import random

class NoActivationFunctionError(Exception):
        pass

class neuron:
        def __init__(self) -> None:
                self._input: np.float16 = 0.0
                self._output: np.float16 = 0.0
                self._weights: list[np.float16] = []
                
                self._gradient: np.float16 = 1.0
                
                self._activation_function = None
                
                self._bias = 0.0
        
        @property
        def output(self) -> np.float16:
                return self._output
        
        @property
        def input(self) -> np.float16:
                return self._input
        
        @property
        def bias(self) -> np.float16:
                return self._bias
        
        @property
        def gradient(self) -> np.float16:
                return self._gradient
        
        @property
        def weights(self) -> list[np.float16]:
                return self._weights
        
        def generate_weights(self, size: int) -> None:
                for _ in range(size):
                        self._weights.append(random.random() - 0.5) 
        
        def set_activation_function(self, activation_function) -> None:
                self._activation_function = activation_function
        
        def set_input(self, input: np.float16) -> None:
                self._input = input
        
        def set_bias(self, bias: np.float16) -> None:
                self._bias = bias
        
        def set_gradient(self, gradient: np.float16) -> None:
                self._gradient = gradient
                
        def calculate(self) -> None:
                if self._activation_function:
                        self._output = self._activation_function(self._input)
                
                else:
                        self._output = self._input            