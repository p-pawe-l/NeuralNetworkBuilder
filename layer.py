import neuron
import numpy as np
import math

class EmptyLayerError(Exception):
        pass

class LayerTypeError(Exception):
        pass

class NotMatchingElementsError(Exception):
        pass

class layer:
        def __init__(self, neurons: list[neuron.neuron]) -> None:
                self._neurons: list[neuron.neuron] = neurons
                
                self._ac_func: bool = False
                
                self._is_output_layer: bool = False
                self._is_input_layer: bool = False
                
                self._prev: layer | None = None
                self._next: layer | None = None
                
        @property
        def neurons(self) -> list[neuron.neuron]:
                return self._neurons       
        
        def relu(self) -> None:
                if self._is_input_layer:
                        raise LayerTypeError 
                
                if not self._neurons: 
                        raise EmptyLayerError
                
                def relu_func(x):
                        return max(0, x)
                
                self._ac_func = True
                for n in self._neurons:
                     n.set_activation_function(relu_func)   
                     
        def tanh(self) -> None:
                if self._is_input_layer:
                        raise LayerTypeError
                
                if not self._neurons:
                        raise EmptyLayerError

                def tanh_func(x):
                        two_x = 2 * x
                        return (math.e ** two_x - 1)/(math.e ** two_x + 1)
                
                self._ac_func = True
                for n in self._neurons:
                     n.set_activation_function(tanh_func) 
                     
        def sigmoid(self) -> None:          
                if self._is_input_layer:
                        raise LayerTypeError
                  
                if not self._neurons:
                        raise EmptyLayerError

                def sigmoid_func(x):
                        return 1 / (1 + math.e ** (-x))
                
                self._ac_func = True
                for n in self._neurons:
                     n.set_activation_function(sigmoid_func) 
                
        
        def set_next(self, other: object) -> None:
                self._next = other
                
        def give_input(self, input: list[np.float16]):
                if self._is_input_layer:  
                        # Check for dimension correctness
                        if (len(input) == len(self._neurons)):
                                for i in range(len(input)):
                                        self._neurons[i].set_input(input[i])
                        else:
                                raise NotMatchingElementsError
                                       
                else:
                        print("UB")
                        
        def get_output(self) -> list[np.float16]:
                if self._is_output_layer:
                        outputs: list[np.float16] = []
                        for n in self._neurons:
                                outputs.append(n.output)
                        
                        return outputs
                
                else:
                        print("UB")
                        
        def out(self) -> None:
                self._is_output_layer = True 
        def inp(self) -> None:
                self._is_input_layer = True
        
        def apply_biases(self) -> None:
                for n in self._neurons:
                        n.set_input(n.input + n.bias)                
                
        def set_activation_function(self, activation_function) -> None:
                if self._neurons:
                        self._ac_func = True
                        for neuron in self._neurons:
                                neuron.set_activation_function(activation_function)
                else:
                        raise EmptyLayerError
                
        def init_weights(self) -> None:
                if not self._is_output_layer:
                        x: int = len(self._next.neurons)
                        
                        for n in self._neurons:
                                n.generate_weights(x)
        
        def call(self) -> None:
                if not self._is_output_layer:
                        b: int = len(self._next.neurons)
                        
                        for y in range(b):
                                n = self._next.neurons[y]
                                a: int = len(self._neurons)
                                
                                inp: np.float16 = 0.0
                                for x in range(a):
                                        ne: neuron.neuron = self._neurons[x]
                                        w: np.float16 = ne.weights[y]
                                        
                                        inp = inp + ne.input * w
                                
                                n.set_input(inp)
                                        
                 
                
        
                
        