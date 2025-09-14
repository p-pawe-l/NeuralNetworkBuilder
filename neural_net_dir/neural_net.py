from jedi.inference.arguments import iterate_argument_clinic

import layer_dir.layer_factory as lf
import layer_dir.layer as layer
import architecture_builder_dir.architecture_component as architecture_component
import numpy

class IncorrectArchitectureError(Exception):
        pass

class NeuralNetwork:
        def __init__(self, architecture: list[architecture_component.ArchitectureComponent]) -> None:
                if len(architecture) < 2:
                        raise IncorrectArchitectureError
                
                self._architecture: list[architecture_component.ArchitectureComponent] = architecture
                
                self._input_layer: layer.InputLayer | None = None
                self._hidden_layers: list[layer.HiddenLayer] = []
                self._output_layer: layer.OutputLayer | None = None
                
                self._net_output: numpy.ndarray | None = None
        
        
        @property
        def input_layer(self) -> layer.InputLayer:
                return self._input_layer
        @property
        def hidden_layers(self) -> list[layer.HiddenLayer]:
                return self._hidden_layers
        @property
        def output_layer(self) -> layer.OutputLayer:
                return self._output_layer
        @property
        def neural_net_output(self) -> numpy.ndarray:
                return self._net_output

        def set_neural_network_output_matrix(self, new_neural_network_output_matrix: numpy.ndarray) -> None:
                self._net_output = new_neural_network_output_matrix
        
                
        def build_input_layer(self) -> None:
                input_layer_input_size: int = self._architecture[0].units
                input_layer_output_size: int = self._architecture[1].units
                
                self._input_layer = lf.LayerFactory.build_input_layer(
                        input_size=input_layer_input_size,
                        output_size=input_layer_output_size
                )
                
        def build_hidden_layers(self) -> None:
                control_point: int = 1
                iteration_number: int = len(self._architecture) - 2
                
                for _ in range(iteration_number):
                        hidden_layer_input_size: int = self._architecture[control_point].units
                        hidden_layer_output_size: int = self._architecture[control_point + 1].units
                        hidden_layer_activation_function = self._architecture[control_point].activation_function
                        hidden_layer_derivative_function = self._architecture[control_point].derivative_function
                        
                        control_point += 1
                        
                        built_hidden_layer: layer.HiddenLayer = (lf.LayerFactory.build_hidden_layer(
                                input_size=hidden_layer_input_size,
                                output_size=hidden_layer_output_size,
                                activation_function=hidden_layer_activation_function,
                                derivative_function=hidden_layer_derivative_function
                        ))
                        
                        self._hidden_layers.append(built_hidden_layer)
        
        def build_output_layer(self) -> None:
                output_layer_input_size: int = self._architecture[-1].units
                output_layer_activation_function = self._architecture[-1].activation_function
                output_layer_derivative_function = self._architecture[-1].derivative_function
                
                self._output_layer = lf.LayerFactory.build_output_layer(
                        input_size=output_layer_input_size,
                        activation_function=output_layer_activation_function,
                        derivative_function=output_layer_derivative_function
                )        
        
        # temporary function -> going to be implemented in neural factory! 
        def build(self) -> None:
                self.build_input_layer()
                self.build_hidden_layers()
                self.build_output_layer()
        
        
        def front_propagation(self, input_array: numpy.ndarray) -> None:
                self.input_layer.set_input_matrix(input_array)
                self.input_layer.layer_front_propagation()

                input_matrix_helper: numpy.ndarray = self.input_layer.layer_output_array
                if self.hidden_layers:
                        for hidden_layer in self.hidden_layers:
                                hidden_layer.set_layer_input_matrix(layer_input_matrix=input_matrix_helper)
                                hidden_layer.layer_front_propagation()

                                input_matrix_helper = hidden_layer.layer_output_matrix

                self.output_layer.set_layer_input_matrix(new_layer_input_matrix=input_matrix_helper)
                self.output_layer.calculate_layer_output()

                neural_network_output_matrix: numpy.ndarray = self.output_layer.post_activation_matrix
                self.set_neural_network_output_matrix(new_neural_network_output_matrix=neural_network_output_matrix)

        def back_propagation(self, target_matrix: numpy.ndarray) -> None:
                self.output_layer.calculate_cost_function_derivative(target_matrix=target_matrix)

                back_propagated_gradient_matrix: numpy.ndarray = self.output_layer.cost_function_derivative_matrix
                previous_layer_matrix: numpy.ndarray = numpy.array([])
                if self.hidden_layers:
                        previous_layer_matrix = self.hidden_layers[-1].post_activation_matrix
                else:
                        previous_layer_matrix = self.input_layer.input_array

                weights_gradient_matrix: numpy.ndarray = self.output_layer.calculate_local_derivative(
                        post_activations_previous_layer_matrix=previous_layer_matrix) * back_propagated_gradient_matrix
