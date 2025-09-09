import numpy
import layer_errors
         
class NoActivationFunctionError(Exception):
        pass


         
class input_layer:
        def __init__(self, input_size: int, output_size: int) -> None:
                # Type of our layer
                self._type: str = 'input'

                # Number of perceptrons in our layer
                self._input_size: int = input_size
                # Number of perceptrons in next layer of our neural_network
                self._output_size: int = output_size
                
                # Matrix that contains outgoing weights from our layer
                self._outgoing_weights: numpy.ndarray = numpy.array([])
                
                # Input matrix provided by user
                self._input_array: numpy.ndarray = numpy.array([])
                # Result of multiplication of Input matrix provided by user and matrix that contains data about outgoing weights
                # This matrix will be used as input matrix for next layer in out neural network
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
                """
                Initializes the hidden layer.

                Args:
                    input_size (int): The number of neurons in the previous layer (input size).
                    output_size (int): The number of neurons in the next layer.
                    activation_function: The activation function object to be used in this layer.
                """
                # Type of our layer
                self._type: str = 'hidden'
                
                # Activation function for our layer
                self._activation_function = activation_function
                # Number of perceptrons in our layer
                self._input_size: int = input_size
                # Number of perceptrons in next layer of our neural network
                self._output_size: int = output_size
                
                # Matrix that contains outgoing weights from our layer
                self._outgoing_weights: numpy.ndarray = numpy.array([])
                # Matrix that contains biases to apply for our layer
                self._biases: numpy.ndarray = numpy.array([])
                
                # Input matrix for hidden layer in out neural network
                self._layer_input_array: numpy.ndarray = numpy.array([])

                # Input matrix with applied biases to it
                self._pre_activation_sum_array: numpy.ndarray = numpy.array([])
                # Transformed input matrix with biases by activtion function
                self._post_activation_sum_array: numpy.ndarray = numpy.array([])

                # Output matrix from multiplication post_activation_sum_array and outgoing_weights 
                # This matrix will be used as input layer matrix for next layer in our neural network 
                self._layer_output_array: numpy.ndarray = numpy.array([])

        """
        Getters methods
        """
        @property
        def layer_type(self) -> str:
                """
                Returns information about type of the layer
                """
                return self._type       
        @property
        def activation_function(self):
                """
                Returns object of proper activation function of the layer
                """
                return self._activation_function
        @property
        def input_size(self) -> int:
                """
                Returns input size of the layer
                Can be interpreted as number of perceptrons in the layer
                """
                return self._input_size
        @property
        def output_size(self) -> int:
                """
                Returns output sze of the layer
                Can be interpreted as number of perceptrons in the next layer
                """
        @property
        def outgoing_weights(self) -> numpy.ndarray:
                """
                Returns outgoing weights from the layer
                If outgoing weights are not initialized this method 
                will raise an error communicating that weighths matrix is not created
                """
                if not self._outgoing_weights:
                        raise layer_errors.NotInitializedWeightsError
                else:
                        return self._outgoing_weights
        @property
        def biases(self) -> numpy.ndarray:
                """
                Returns biases of the layer
                If biases are not initialized this method
                will raise an error communicating that biases matrix in not created
                """
                if not self._biases:
                        raise layer_errors.NotInitiazliedBiasesError
                else:
                        return self._biases
        @property
        def layer_input_array(self) -> numpy.array:
                """
                Returns Matrix that contains input values for the layer
                If matrix for input layer is empty this method 
                will raise and erorr communicating that 
                """
                if not self._layer_input_array:
                        raise layer_errors.EmptyLayerInputMatrixError
                else:
                        return self._layer_input_array
        @property
        def pre_activation_matrix(self) -> numpy.ndarray:
                if not self._pre_activation_sum_array:
                        raise layer_errors.EmptyPreActivationSumMatrixError
                else:
                        return self._pre_activation_sum_array
        @property
        def post_activation_matrix(self) -> numpy.ndarray:
                if not self._post_activation_sum_array:
                        raise layer_errors.EmptyPostActivationSumMatrixError
                else:
                        return self._post_activation_sum_array


        """
        Setters methods
        """
        def set_activation_function(self, activation_function) -> None:
                """Sets the activation function for the layer."""
                self._activation_function = activation_function
        def set_outgoing_weights(self, outgoing_weights: numpy.ndarray) -> None:
                """Sets the outgoing weights matrix for the layer."""
                self._outgoing_weights = outgoing_weights
        def set_biases(self, biases: numpy.ndarray) -> None:
                """Sets the biases vector for the layer."""
                self._biases = biases
        def set_layer_input_matrix(self, layer_input_matrix: numpy.ndarray) -> None:
                """Sets the input matrix for the layer."""
                self._layer_input_array = layer_input_matrix
        def set_pre_activation_matrix(self, pre_activation_matrix: numpy.ndarray) -> None:
                """Sets the pre-activation (weighted sum) matrix."""
                self._pre_activation_sum_array = pre_activation_matrix
        def set_post_activation_matrix(self, post_activation_matrix: numpy.ndarray) -> None:
                """Sets the post-activation (activated) matrix."""
                self._post_activation_sum_array = post_activation_matrix
        def set_layer_output_matrix(self, layer_output_matrix: numpy.ndarray) -> None:
                """Sets the final output matrix of the layer."""
                self._layer_output_array = layer_output_matrix
                
                
        
        def init_weights(self) -> None:
                # Checking if outgoing weights are not initialized
                if not self.outgoing_weights:
                        # Creating random value matrix with provided input and output size
                        initialized_weights: numpy.ndarray = numpy.random.randn(self.input_size, self.output_size)
                        
                        # Assigning newly initiazlied weights to outgoing weights attrribute
                        self.set_outgoing_weights(initialized_weights)
                else:
                        raise layer_errors.ParametersInitializationError               
                
        def init_biases(self) -> None:
                # Checking if biases are not initialized
                if not self.biases:
                        # Creating zero value matrix with provided input size
                        initialized_biases: numpy.ndarray = numpy.zeros(self.input_size)
                        
                        # Assigning newly created biases to biases attribute
                        self.set_biases(initialized_biases)
                else:
                        raise layer_errors.ParametersInitializationError
                
        """
        Helper method to initialize paramters using single function
        """
        def init_parameters(self) -> None:
                # Initializing weights
                self.init_weights()
                
                # Initializing biases
                self.init_biases()
                
                
        def reset_weights(self) -> None:
                if self.outgoing_weights:
                        # Creating new random value matrix with weights
                        new_random_weights: numpy.ndarray = numpy.random.randn(self.input_size, self.output_size)
                        
                        # Assigning this matrix to layer`s weights
                        self.set_outgoing_weights(new_random_weights)
                else:
                        """
                        ERROR PURPOSE:
                        Error informs user that weights parameter shold be initialized before trying to reseting it
                        """
                        raise layer_errors.ParametersInitializationError
                
        def reset_biases(self) -> None:
                if self.biases:
                        # Creating new zero value matrix with biases
                        new_biases: numpy.ndarray = numpy.zeros(self.input_size)
                        
                        # Assigning this matrix to layer`s biases
                        self.set_biases(new_biases)
                else:
                        """
                        ERROR PURPOSE:
                        Error informs user that biases paramater shuold be initiazlied before trying to reseting it
                        """
                        layer_errors.ParametersInitializationError
        
        """
        Helper method to reset paramaters using single function
        """                
        def reset_parameters(self) -> None:
                # Reseting weights
                self.reset_weights()
                
                # Reseting biases
                self.reset_biases()
                
                
        def apply_biases(self) -> None:
                """Applies biases to the layer's input array."""
                
                # Checking if Biases Matrix and Layer Input Matrix are assigned correctly
                if self.biases and self.layer_input_array:
                        # Applying (Summing) Biases to Layer Input Matrix
                        PRE_ACTIVATION_MATRIX: numpy.ndarray = self.layer_input_array + self.biases
                        
                        # Assigning the result to Pre Activation Matrix attribute
                        self.set_pre_activation_matrix(pre_activation_matrix=PRE_ACTIVATION_MATRIX)

        def activate(self) -> None:
                """Applies the activation function to the pre-activation matrix."""
                
                # Cheking if Pre Activation Matrix is assigned correctly
                if self.pre_activation_matrix:
                        # Transofrming Pre Activation Matrix into Post Activation Matrix 
                        # with provided activation function
                        POST_ACTIVATION_MATRIX: numpy.ndarray = self.activation_function.activate(self.pre_activation_matrix)
                        
                        # Assigning the result for Post Activation Matrix attribute
                        self.set_post_activation_matrix(post_activation_matrix=POST_ACTIVATION_MATRIX)

        def layer_front_propagation(self) -> None:
                """Performs a single step of forward propagation for the layer."""
                
                # Applying biases for Layer Input Matrix
                self.apply_biases()
                
                # Activating Layer Input Matrix with activation function
                # and assigning the result of it to Pre Activation Matrix
                self.activate()

                # Calculating Linear Output for next layer in our Neural Network
                LINEAR_OUTPUT: numpy.ndarray = numpy.dot(self.post_activation_matrix, self.outgoing_weights)
                
                # Assigning the result of Matrix multiplication to Layer Output Matrix attribute
                # Layer Output Matrix will be used and Layer Input Matrix in next layer of our neural network
                self.set_layer_output_matrix(layer_output_matrix=LINEAR_OUTPUT)
                
        
class output_layer:
        def __init__(self, input_size: int, activation_function) -> None:
                # Type of our layer
                self._type = 'output'
                
                # Activation function for output layer in our neural network
                self._activation_function = activation_function
                # Input size of output layer in our neural network
                # Input size can be interpretted as number of perceptrons in output layer
                self._input_size: int = input_size
                
                # Biases Matrix that contains values for biases for each perceptron in output layer
                self._biases: numpy.ndarray = numpy.array([])
                        
                # Input Matrix that contains input values for each perceptron in output layer
                self._layer_input_array: numpy.ndarray = numpy.array([])   
                
                # Pre Activation Matrix that contains values after calculations on each perceptron
                # before transforming them by output layer`s activation function
                self._pre_activation_output_array: numpy.ndarray = numpy.array([])
                # Post Actication Matrix that contains values after calculations on each perceptron
                # and addiotionally this values in this matrix are transformed by output layer`s activation function 
                self._post_activation_output_array : numpy.ndarray = numpy.array([])
        
        
        """
        Getters method
        """
        @property
        def layer_type(self) -> str:
                return self._type
        @property
        def activation_function(self):
                return self._activation_function
        @property
        def input_size(self) -> int:
                return self._input_size
        @property
        def biases(self) -> numpy.ndarray:
                return self._biases
        @property
        def layer_input_matrix(self) -> numpy.ndarray:
                return self._layer_input_array
        @property
        def pre_activation_matrix(self) -> numpy.ndarray:
                return self._pre_activation_output_array
        @property
        def post_activation_matrix(self) -> numpy.ndarray:
                return self._post_activation_output_array


        # Needs implementaion
        def init_biases(self) -> None:
                return None