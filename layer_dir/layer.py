import numpy
import layer_dir.layer_errors as layer_errors
         
class NoActivationFunctionError(Exception):
        pass



class InputLayer:
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


        def set_input_matrix(self, new_input_matrix: numpy.ndarray) -> None:
                self._input_array = new_input_matrix
        def set_layer_output_matrix(self, new_layer_output_matrix: numpy.ndarray) -> None:
                self._layer_output_array = new_layer_output_matrix
        def set_outgoing_weights(self, new_outgoing_weights: numpy.ndarray) -> None:
                self._outgoing_weights = new_outgoing_weights


        def init_parameters(self) -> None:
                self._outgoing_weights = numpy.random.randn(self._input_size, self._output_size)
                
        def layer_front_propagation(self) -> None:
                linear_output: numpy.ndarray = numpy.dot(self.input_array, self.outgoing_weights)

                self.set_layer_output_matrix(linear_output)

        
                
class HiddenLayer:
        def __init__(self, input_size: int, output_size: int, activation_function, derivative_function) -> None:
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
                # Derivative function for our layer
                self._derivative_function = derivative_function
                # Number of perceptron in our layer
                self._input_size: int = input_size
                # Number of perceptron in next layer of our neural network
                self._output_size: int = output_size
                
                # Matrix that contains outgoing weights from our layer
                self._outgoing_weights: numpy.ndarray = numpy.array([])
                # Matrix that contains biases to apply for our layer
                self._biases: numpy.ndarray = numpy.array([])
                
                # Input matrix for hidden layer in out neural network
                self._layer_input_array: numpy.ndarray = numpy.array([])

                # Input matrix with applied biases to it
                self._pre_activation_sum_array: numpy.ndarray = numpy.array([])
                # Transformed input matrix with biases by activation function
                self._post_activation_sum_array: numpy.ndarray = numpy.array([])

                # Output matrix from multiplication post_activation_sum_array and outgoing_weights 
                # This matrix will be used as input layer matrix for next layer in our neural network 
                self._layer_output_array: numpy.ndarray = numpy.array([])

                self._back_propagated_gradient_matrix: numpy.ndarray = numpy.array([])
                self._local_derivative_matrix: numpy.ndarray = numpy.array([])

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
        def derivative_function(self):
                return self._derivative_function
        @property
        def input_size(self) -> int:
                """
                Returns input size of the layer
                Can be interpreted as number of perceptron in the layer
                """
                return self._input_size
        @property
        def output_size(self) -> int:
                """
                Returns output sze of the layer
                Can be interpreted as number of perceptron in the next layer
                """
                return self._output_size
        @property
        def outgoing_weights(self) -> numpy.ndarray:
                """
                Returns outgoing weights from the layer
                If outgoing weights are not initialized this method 
                will raise an error communicating that weights matrix is not created
                """
                return self._outgoing_weights
        @property
        def biases(self) -> numpy.ndarray:
                """
                Returns biases of the layer
                If biases are not initialized this method
                will raise an error communicating that biases matrix in not created
                """
                return self._biases
        @property
        def layer_input_array(self) -> numpy.array:
                """
                Returns Matrix that contains input values for the layer
                If matrix for input layer is empty this method 
                will raise and error communicating that
                """
                return self._layer_input_array
        @property
        def pre_activation_matrix(self) -> numpy.ndarray:
                return self._pre_activation_sum_array
        @property
        def post_activation_matrix(self) -> numpy.ndarray:
                return self._post_activation_sum_array
        @property
        def layer_output_matrix(self) -> numpy.ndarray:
                return self._layer_output_array

        @property
        def local_derivative_matrix(self) -> numpy.ndarray:
                return self._local_derivative_matrix
        @property
        def back_propagated_gradient_matrix(self) -> numpy.ndarray:
                return self._back_propagated_gradient_matrix



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
        def set_back_propagated_matrix(self, new_back_propagated_matrix: numpy.ndarray) -> None:
                self._back_propagated_gradient_matrix = new_back_propagated_matrix
                
                
        
        def init_weights(self) -> None:
                # Checking if outgoing weights are not initialized
                if not (self.outgoing_weights.size > 0):
                        # Creating random value matrix with provided input and output size
                        initialized_weights: numpy.ndarray = numpy.random.randn(self.input_size, self.output_size)
                        
                        # Assigning newly initialized weights to outgoing weights attribute
                        self.set_outgoing_weights(initialized_weights)
                else:
                        raise layer_errors.ParametersInitializationError               
                
        def init_biases(self) -> None:
                # Checking if biases are not initialized
                if not (self.biases.size > 0):
                        # Creating zero value matrix with provided input size
                        initialized_biases: numpy.ndarray = numpy.zeros(self.input_size)
                        
                        # Assigning newly created biases to biases attribute
                        self.set_biases(initialized_biases)
                else:
                        raise layer_errors.ParametersInitializationError
                
        """
        Helper method to initialize parameters using single function
        """
        def init_parameters(self) -> None:
                # Initializing weights
                self.init_weights()
                
                # Initializing biases
                self.init_biases()
                
                
        def reset_weights(self) -> None:
                if self.outgoing_weights.size > 0:
                        # Creating new random value matrix with weights
                        new_random_weights: numpy.ndarray = numpy.random.randn(self.input_size, self.output_size)
                        
                        # Assigning this matrix to layer`s weights
                        self.set_outgoing_weights(new_random_weights)
                else:
                        """
                        ERROR PURPOSE:
                        Error informs user that weights parameter should be initialized before trying to resetting it
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
                        Error informs user that biases parameter should be initialized before trying to resetting it
                        """
                        raise layer_errors.ParametersInitializationError
        
        """
        Helper method to reset parameters using single function
        """                
        def reset_parameters(self) -> None:
                # Resetting weights
                self.reset_weights()
                
                # Resetting biases
                self.reset_biases()
                
                
        def apply_biases(self) -> None:
                """Applies biases to the layer's input array."""
                
                # Checking if Biases Matrix and Layer Input Matrix are assigned correctly
                if self.biases.size > 0 and self.layer_input_array.size > 0:
                        # Applying (Summing) Biases to Layer Input Matrix
                        pre_activation_matrix: numpy.ndarray = self.layer_input_array + self.biases
                        
                        # Assigning the result to Pre Activation Matrix attribute
                        self.set_pre_activation_matrix(pre_activation_matrix=pre_activation_matrix)

        def activate(self) -> None:
                """Applies the activation function to the pre-activation matrix."""
                
                # Checking if Pre Activation Matrix is assigned correctly
                if self.pre_activation_matrix.size > 0:
                        # Transforming Pre Activation Matrix into Post Activation Matrix
                        # with provided activation function
                        post_activation_matrix: numpy.ndarray = self.activation_function(self.pre_activation_matrix)
                        
                        # Assigning the result for Post Activation Matrix attribute
                        self.set_post_activation_matrix(post_activation_matrix=post_activation_matrix)

        def layer_front_propagation(self) -> None:
                """Performs a single step of forward propagation for the layer."""
                
                # Applying biases for Layer Input Matrix
                self.apply_biases()
                
                # Activating Layer Input Matrix with activation function
                # and assigning the result of it to Pre Activation Matrix
                self.activate()

                # Calculating Linear Output for next layer in our Neural Network
                linear_output: numpy.ndarray = numpy.dot(self.post_activation_matrix, self.outgoing_weights)
                
                # Assigning the result of Matrix multiplication to Layer Output Matrix attribute
                # Layer Output Matrix will be used and Layer Input Matrix in next layer of our neural network
                self.set_layer_output_matrix(layer_output_matrix=linear_output)

        def calculate_local_derivative(self, previous_layer_post_activation_matrix: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
                def calculate_derivative_of_sums_matrix() -> numpy.ndarray:
                        derivative_sums_matrix: numpy.ndarray = self.derivative_function(self.pre_activation_matrix)

                        return derivative_sums_matrix

                derivative_of_sums: numpy.ndarray = calculate_derivative_of_sums_matrix()
                stacked_matrix: numpy.ndarray = numpy.column_stack([previous_layer_post_activation_matrix for _ in range(self.input_size)])

                local_derivative_matrix_weights: numpy.ndarray = stacked_matrix * derivative_of_sums
                local_derivative_matrix_biases: numpy.ndarray = derivative_of_sums

                return local_derivative_matrix_weights , local_derivative_matrix_biases


class OutputLayer:
        def __init__(self, input_size: int, activation_function, derivative_function) -> None:
                # Type of our layer
                self._type = 'output'
                
                # Activation function for output layer in our neural network
                self._activation_function = activation_function
                self._derivative_function = derivative_function
                # Input size of output layer in our neural network
                # Input size can be interpreted as number of perceptron in output layer
                self._input_size: int = input_size
                
                # Biases Matrix that contains values for biases for each perceptron in output layer
                self._biases: numpy.ndarray = numpy.array([])
                        
                # Input Matrix that contains input values for each perceptron in output layer
                self._layer_input_array: numpy.ndarray = numpy.array([])   
                
                # Pre Activation Matrix that contains values after calculations on each perceptron
                # before transforming them by output layer`s activation function
                self._pre_activation_output_array: numpy.ndarray = numpy.array([])
                # Post Activation Matrix that contains values after calculations on each perceptron
                # and additionally this values in this matrix are transformed by output layer`s activation function
                self._post_activation_output_array : numpy.ndarray = numpy.array([])

                self._cost_function_derivative_matrix: numpy.ndarray = numpy.array([])
                self._local_derivative_matrix: numpy.ndarray = numpy.array([])
        
        
        """
        Getters methods
        """
        @property
        def layer_type(self) -> str:
                return self._type
        @property
        def activation_function(self):
                return self._activation_function
        @property
        def derivative_function(self):
                return self._derivative_function
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

        @property
        def cost_function_derivative_matrix(self) -> numpy.ndarray:
                return self._cost_function_derivative_matrix
        @property
        def derivative_sums_matrix(self) -> numpy.ndarray:
                return self._derivative_of_sums_matrix


        """
        Setters methods
        """
        def set_biases(self, new_biases_matrix: numpy.ndarray) -> None:
                self._biases = new_biases_matrix
        def set_activation_function(self, new_activation_function) -> None:
                self._activation_function = new_activation_function
        def set_layer_input_matrix(self, new_layer_input_matrix) -> None:
                self._layer_input_array = new_layer_input_matrix
        def set_pre_activation_matrix(self, new_pre_activation_matrix: numpy.ndarray) -> None:
                self._pre_activation_output_array = new_pre_activation_matrix
        def set_post_activation_matrix(self, new_post_activation_matrix: numpy.ndarray) -> None:
                self._post_activation_output_array = new_post_activation_matrix
        def set_derivative_cost_function_matrix(self, new_derivative_cost_function_matrix: numpy.ndarray) -> None:
                self._cost_function_derivative_matrix = new_derivative_cost_function_matrix

        def init_biases(self) -> None:
                initialized_biases: numpy.ndarray = numpy.zeros(self.input_size)
                self.set_biases(new_biases_matrix=initialized_biases)

        def apply_biases(self) -> None:
                applied_biases_matrix: numpy.ndarray = self.layer_input_matrix + self.biases
                self.set_pre_activation_matrix(new_pre_activation_matrix=applied_biases_matrix)

        def activate(self) -> None:
                if self.activation_function:
                        activated_matrix: numpy.ndarray = self.activation_function(self.pre_activation_matrix)
                        self.set_post_activation_matrix(new_post_activation_matrix=activated_matrix)
                else:
                        raise NoActivationFunctionError

        def calculate_layer_output(self) -> None:
                self.apply_biases()
                self.activate()



        def calculate_cost_function_derivative(self, target_matrix: numpy.ndarray) -> None:
                error_delta_matrix: numpy.ndarray = target_matrix - self.post_activation_matrix
                derivative_cost_function_matrix: numpy.ndarray = -2 * error_delta_matrix

                self.set_derivative_cost_function_matrix(new_derivative_cost_function_matrix=derivative_cost_function_matrix)

        def calculate_local_derivative(self, post_activations_previous_layer_matrix: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:

                def calculate_derivative_of_sums_matrix() -> numpy.ndarray:
                        derivative_matrix: numpy.ndarray = self.derivative_function(self.pre_activation_matrix)
                        return derivative_matrix

                stacked_matrix: numpy.ndarray = numpy.column_stack([post_activations_previous_layer_matrix for _ in range(self.input_size)])
                derivative_sums_matrix: numpy.ndarray = calculate_derivative_of_sums_matrix()

                local_derivative_matrix: numpy.ndarray = stacked_matrix * derivative_sums_matrix
                local_derivative_biases: numpy.ndarray = derivative_sums_matrix

                return local_derivative_matrix, local_derivative_biases



