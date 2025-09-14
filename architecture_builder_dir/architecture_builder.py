import architecture_builder_dir.architecture_component as architecture_component
import activation_function_dir.activation_functions as activation_functions
import os
import json


activation_function_map: dict = {
        'relu': activation_functions.ReluActivationFunction,
        'sigmoid': activation_functions.SigmoidActivationFunction
}


class ArchitectureBuilder:
        def __init__(self, architecture_dictionary: dict) -> None:
                def validate_architecture_dictionary(architecutre_dictionary: dict) -> bool:
                        if ('input_layer' not in architecture_dictionary.keys()):
                                return False
                        if ('hidden_layers' not in architecture_dictionary.keys()):
                                return False
                        if ('output_layer' not in architecture_dictionary.keys()):
                                return False
                        
                        return True
                        

                if not validate_architecture_dictionary():
                        raise ValueError("Error building neural network architecture")        
                
                self._architecture_dictionary: dict = architecture_dictionary

        def __init__(self, json_file: str) -> None:
                self._json_file: str = json_file

                def read_json_data(json_file: str) -> dict:
                        with open(os.path.abspath(json_file), mode='r') as json_file_handle:
                                return json.load(json_file_handle)

                self._architecture_dictionary: dict = read_json_data(self._json_file)

        def build_components(self) -> list[architecture_component.ArchitectureComponent]:
                architecture_components: list[architecture_component.ArchitectureComponent] = []

                # Building input layer component
                input_layer_units: int = self._architecture_dictionary['input_layer']['units']
                architecture_components.append(architecture_component.ArchitectureComponent(
                        units=input_layer_units,
                        activation_function=None,
                        derivative_function=None))

                # Building hidden layers components
                hidden_layers_list: list = self._architecture_dictionary['hidden_layers']
                if hidden_layers_list:
                        for hidden_layer_data in hidden_layers_list:
                                hidden_layer_units: int = hidden_layer_data['units']
                                hidden_layer_activation_function = hidden_layer_data["activation_function"]

                                hidden_layer_function = activation_function_map[hidden_layer_activation_function] 
                                
                                architecture_components.append(architecture_component.ArchitectureComponent(
                                        units=hidden_layer_units, 
                                        activation_function=hidden_layer_function.activation,
                                        derivative_function=hidden_layer_function.derivative))
                        
                # Building output layer component
                output_layer_units: int = self._architecture_dictionary['output_layer']['units']
                output_layer_activation_function = self._architecture_dictionary['output_layer']['activation_function']
                
                output_layer_function = activation_function_map[output_layer_activation_function]
                
                architecture_components.append(architecture_component.ArchitectureComponent(
                        units=output_layer_units, 
                        activation_function=output_layer_function.activation,
                        derivative_function=output_layer_function.derivative))
                
                return architecture_components


                
                

        

        

        
        