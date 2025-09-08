import activation_function_dir.activation_functions as activation_functions

class IncorrectUnitsNumberError(Exception):
        pass

class architecture_component:
        def __init__(self, units: int, activation_function = None, derivative_function = None) -> None:
                if units < 0:
                        raise IncorrectUnitsNumberError
                
                self._units: int = units
                self._activation_function = activation_function
                self._derivative_function = derivative_function
        
        @property
        def units(self) -> int:
                return self._units
        @property
        def activation_function(self):
                return self._activation_function
        @property 
        def derivative_function(self):
                return self._derivative_function