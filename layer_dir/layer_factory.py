from . import layer


class LayerFactory:
        
        @staticmethod
        def build_input_layer(input_size: int, output_size: int) -> layer.InputLayer:
                input_layer: layer.InputLayer = layer.InputLayer(
                        input_size=input_size,
                        output_size=output_size
                )
                
                input_layer.init_parameters()
                return input_layer
                
        @staticmethod
        def build_hidden_layer(input_size: int, output_size: int, activation_function, derivative_function) -> layer.HiddenLayer:
                hidden_layer: layer.HiddenLayer = layer.HiddenLayer(
                        input_size=input_size,
                        output_size=output_size,
                        activation_function=activation_function,
                        derivative_function=derivative_function
                )
                
                hidden_layer.init_parameters()
                return hidden_layer
                
        @staticmethod
        def build_output_layer(input_size: int, activation_function, derivative_function) -> layer.OutputLayer:
                output_layer: layer.OutputLayer = layer.OutputLayer(
                        input_size=input_size,
                        activation_function=activation_function,
                        derivative_function=derivative_function
                )
                
                output_layer.init_biases()
                return output_layer