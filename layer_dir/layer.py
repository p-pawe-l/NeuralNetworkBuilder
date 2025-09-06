import numpy
         
class NoActivationFunctionError(Exception):
        pass
         
class input_layer:
        def __init__(self, input_size: int, output_size: int) -> None:
                self._type: str = 'input'
        
                self._input_size: int = input_size
                self._output_size: int = output_size
                
                self._outgoing_weights: numpy.ndarray = numpy.array([])
                
                self._input_array: numpy.ndarray = numpy.array([])
                self._layer_output_array: numpy.ndarray = numpy.array([])
        
        
        @property
        def id(self) -> str:
                return self._type
        @property
        def layer_output_array(self) -> numpy.ndarray:
                return self._layer_output_array
        @property
        def input_array(self) -> numpy.ndarray:
                return self._input_array   
        @property
        def outgoing_weights(self) -> numpy.ndarray:
                return self._outgoing_weights     
        
        
        def init_paramaters(self) -> None:
                self._outgoing_weights = numpy.random.randn(self._input_size, self._output_size)
                
        def front(self, user_input_array: numpy.ndarray) -> None:
                linear_output: numpy.ndarray = numpy.dot(user_input_array, self._outgoing_weights) 
                
                self._input_array = user_input_array
                self._layer_output_array = linear_output
        
                
class hidden_layer:
        def __init__(self, input_size: int, output_size: int, activation_function) -> None:
                self._type: str = 'hidden'
                
                self._activation_function = activation_function
                self._input_size: int = input_size
                self._output_size: int = output_size
                
                self._outgoing_weights: numpy.ndarray = numpy.array([])
                self._biases: numpy.ndarray = numpy.array([])
                
                self._layer_input_array: numpy.ndarray = numpy.array([])
                self._pre_activation_output: numpy.ndarray = numpy.array([])
                self._post_activation_output: numpy.ndarray = numpy.array([])


        @property
        def id(self) -> str:
                return self._type 
        @property
        def layer_input_array(self) -> numpy.ndarray:
                return self._layer_input_array
        @property
        def pre_activation_array(self) -> numpy.ndarray:
                return self._pre_activation_output
        @property
        def post_activation_array(self) -> numpy.ndarray:
                return self._post_activation_output
        @property
        def outgoing_weights(self) -> numpy.ndarray:
                return self._outgoing_weights  
        
                         
        def init_paramaters(self) -> None:
                self._outgoing_weights = numpy.random.randn(self._input_size, self._output_size)
                self._biases = numpy.zeros(self._input_size)
                
        def front(self, layer_input_array: numpy.ndarray) -> None:
                linear_output: numpy.ndarray = numpy.dot(layer_input_array, self._outgoing_weights) + self._biases
                
                self._layer_input_array = layer_input_array
                self._pre_activation_output = linear_output
                
        def activate(self) -> None:
                if self._activation_function:
                        self._post_activation_output = self._activation_function(self._pre_activation_output)
                else:
                        raise NoActivationFunctionError
        
        
class output_layer:
        def __init__(self, input_size: int, activation_function) -> None:
                self._type = 'output'
                
                self._activation_function = activation_function
                self._input_size: int = input_size
                
                self._biases: numpy.ndarray = numpy.array([])
                        
                self._layer_input_array: numpy.ndarray = numpy.array([])   
                self._pre_activation_output_array: numpy.ndarray = numpy.array([])
                self._post_activation_output_array : numpy.ndarray = numpy.array([])
        
        
        @property
        def id(self) -> str:
                return self._type
        @property
        def layer_input_array(self) -> numpy.ndarray:
                return self._layer_input_array
        @property
        def pre_activation_output(self) -> numpy.ndarray:
                return self._pre_activation_output_array
        @property
        def post_activation_output_array(self) -> numpy.ndarray:
                return self._post_activation_output_array
        
                
        def init_paramaters(self) -> None:
                self._biases = numpy.zeros(self._input_size)
                
        def calcualte_output(self, layer_input_array: numpy.ndarray) -> None:
                self._pre_activation_output_array = layer_input_array + self._biases
                
                self._layer_input_array = layer_input_array
                
        def activate(self) -> None:
                if self._activation_function:
                        self._post_activation_output_array = self._activation_function(self._pre_activation_output_array)
                else:
                        raise NoActivationFunctionError 