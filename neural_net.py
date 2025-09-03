import layer
import neuron
import numpy as np

class WrongArchitectureError(Exception):
        pass

class neural_network:
        def __init__(self, architecture: list[int]) -> None:
                network_size: int = len(architecture)
                
                if network_size < 2:
                        raise WrongArchitectureError
                
                self._layers: list[layer.layer] = []
                
                for i in range(network_size):
                        layer_size: int = architecture[i]
                        layer_neurons: list[neuron.neuron] = [neuron.neuron() for _ in range(layer_size)]
                        
                        new_layer: layer.layer = layer.layer(layer_neurons)                        
                        self._layers.append(new_layer)
                                
                self._input_layer = self._layers[0]
                self._output_layer = self._layers[-1]
                
                for ii in range(network_size - 1):
                        self._layers[ii].set_next(self._layers[ii + 1])
                        
                for iii in range(network_size):
                        l: layer.layer = self._layers[iii]
                        
                        if iii == 0:
                                l.inp()
                                l.init_weights()
                        elif iii == (network_size - 1):
                                l.out()
                                l.sigmoid()
                        else:
                                l.init_weights()
                                l.relu()
                                
        @property
        def layers(self) -> list[layer.layer]:
                return self._layers
        @property
        def input_layer(self) -> layer.layer:
                return self._input_layer
        @property
        def output_layer(self) -> layer.layer:
                return self._output_layer
        
        @property
        def network_output(self) -> list[np.float16]:
                return self._output_layer.get_output()
                
        def front_prop(self, input: list[np.float16]) -> None:
                self._input_layer.give_input(input)
                
                for l in self._layers:
                        l.call()
                        