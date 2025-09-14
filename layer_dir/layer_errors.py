class NotInitializedWeightsError(Exception):
        """
        Error that communicates to user that layer does not have
        initizalized weights
        """
        pass

class NotInitiazliedBiasesError(Exception):
        """
        Error that communicates to user that layer does not have
        initialized biases
        """
        pass

class EmptyLayerInputMatrixError(Exception):
        """
        Error that communcates to user that layer does not receive 
        any input matrix for calculations
        """
        pass

class EmptyPreActivationSumMatrixError(Exception):
        """
        """
        pass

class EmptyPostActivationSumMatrixError(Exception):
        """
        """
        pass

class ParametersInitializationError(Exception):
        """
        """
        pass

class NoActivationFunctionError(Exception):
        """
        """
        pass

class EmptyLayerOutputMatrixError(Exception):
        """
        """
        pass