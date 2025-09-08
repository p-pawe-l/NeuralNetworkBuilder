import architecture_component
import activation_function_dir.activation_functions as activation_functions
import os
import json

class architecture_builder:
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
                self._json_file = json_file

                def read_json_data(json_file: str) -> dict:
                        with open(os.path.abspath(json_file), mode='r') as json_file_handle:
                                return json.loads(json_file_handle)

                self._architecture_dictionary: dict = read_json_data(self._json_file)

        def build_components(self) -> list[architecture_component.architecture_component]:
                architecture_components: list[architecture_component.architecture_component] = []

                # Building input layer component
                input_layer_units: int = self._architecture_dictionary['input']['units']
                architecture_components.append(architecture_component.architecture_component(input_layer_units))

                # Building hidden layers components
                hidden_layers_list: list = self._architecture_dictionary['hidden_layers']
                for hidden_layer_data in hidden_layers_list:
                        hidden_layer_units: int = hidden_layer_data['units']
                        hidden_layer_activation_function = hidden_layer_data["activation_function"]

                        if hidden_layer_activation_function == "relu":
                                hidden_layer_activation_function = activation_functions.relu_activation_function
                        elif hidden_layer_activation_function == "sigmoid":
                                hidden_layer_activation_function = activation_functions.sigmoid_activation_function                  

                        architecture_components.append(architecture_component.architecture_component(hidden_layer_units, hidden_layer_activation_function))
                
                # Building output layer component
                output_layer_units: int = self._architecture_dictionary['output']['units']
                output_layer_activation_function = self._architecture_dictionary['activation_function']
                architecture_components.append(architecture_component.architecture_component(output_layer_units, output_layer_activation_function))
                
                return architecture_components


                
                

        

        

        
        