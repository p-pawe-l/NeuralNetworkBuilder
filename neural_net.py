import layer_dir.layer_factory as lf
import layer_dir.layer as layer
import numpy

class IncorrectArchitectureError(Exception):
        pass

class IncorrectUnitsNumberError(Exception):
        pass

class architecture_component:
        def __init__(self, units: int, activation_function = None) -> None:
                if units < 0:
                        raise IncorrectUnitsNumberError
                
                self._units: int = units
                self._activation_function = activation_function
        
        @property
        def units(self) -> int:
                return self._units
        @property
        def activation_function(self):
                return self._activation_function


class neural_network:
        def __init__(self, archtitecture: list[architecture_component]) -> None:
                if len(archtitecture) < 2:
                        raise IncorrectArchitectureError
                
                self._architecture: list[architecture_component] = archtitecture
                
                self._input_layer: layer.input_layer | None = None
                self._hidden_layers: list[layer.hidden_layer] = []
                self._output_layer: layer.output_layer | None = None
                
                self._net_output: numpy.ndarray | None = None
        
        
        @property
        def input_layer(self) -> layer.input_layer:
                return self._input_layer
        @property
        def hidden_layers(self) -> list[layer.hidden_layer]:
                return self._hidden_layers
        @property
        def output_layer(self) -> layer.output_layer:
                return self._output_layer
        
                
        def build_input_layer(self) -> None:
                input_layer_input_size: int = self._architecture[0].units
                input_layer_output_size: int = self._architecture[1].units
                
                self._input_layer = lf.layer_factor.build_input_layer(
                        input_size=input_layer_input_size,
                        output_size=input_layer_output_size
                )
                
        def build_hidden_layers(self) -> None:
                control_point: int = 1
                itteration_number: int = len(self._architecture) - 2
                
                for _ in range(itteration_number):
                        hidden_layer_input_size: int = self._architecture[control_point].units
                        hidden_layer_output_size: int = self._architecture[control_point + 1].units
                        hidden_layer_activation_function = self._architecture[control_point].activation_function
                        
                        control_point += 1
                        
                        builded_hidden_layer: layer.hidden_layer = lf.layer_factor.build_hidden_layer(
                                input_size=hidden_layer_input_size,
                                output_size=hidden_layer_output_size,
                                activation_function=hidden_layer_activation_function
                        )    
                        
                        self._hidden_layers.append(builded_hidden_layer)            
        
        def build_output_layer(self) -> None:
                output_layer_input_size: int = self._architecture[-1].units
                output_layer_activation_function = self._architecture[-1].activation_function
                
                self._output_layer = lf.layer_factor.build_output_layer(
                        input_size=output_layer_input_size,
                        activation_function=output_layer_activation_function
                )        
        
        
                
                
                                
                
                
                