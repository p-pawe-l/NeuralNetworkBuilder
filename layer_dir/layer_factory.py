from . import layer


class layer_factor:
        
        @staticmethod
        def build_input_layer(input_size: int, output_size: int) -> layer.input_layer:
                input_layer: layer.input_layer = layer.input_layer(
                        input_size=input_size,
                        output_size=output_size
                )
                
                input_layer.init_paramaters()
                return input_layer
                
        @staticmethod
        def build_hidden_layer(input_size: int, output_size: int, activation_function) -> layer.hidden_layer:
                hidden_layer: layer.hidden_layer = layer.hidden_layer(
                        input_size=input_size,
                        output_size=output_size,
                        activation_function=activation_function
                )
                
                hidden_layer.init_paramaters()
                return hidden_layer
                
        @staticmethod
        def build_output_layer(input_size: int, activation_function) -> layer.output_layer:
                output_layer: layer.output_layer = layer.output_layer(
                        input_size=input_size,
                        activation_function=activation_function
                )
                
                output_layer.init_paramaters()
                return output_layer