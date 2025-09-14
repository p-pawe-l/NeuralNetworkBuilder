import neural_net_dir.neural_net as neural_net
import architecture_builder_dir.architecture_builder as architecture_builder
import architecture_builder_dir.architecture_component as architecture_component

class NeuralNetworkFactory:

        @staticmethod
        def create_neural_network_from_json_file(json_file: str) -> neural_net.NeuralNetwork:
                neural_network_builder: architecture_builder.ArchitectureBuilder = architecture_builder.ArchitectureBuilder(json_file)
                architecture_array: list[architecture_component.ArchitectureComponent] = neural_network_builder.build_components()

                created_neural_network: neural_net.NeuralNetwork = neural_net.NeuralNetwork(architecture_array)
                created_neural_network.build()

                return created_neural_network

        @staticmethod
        def create_neural_network_from_architecture_array(architecture_array: list[architecture_component.ArchitectureComponent]) -> neural_net.NeuralNetwork:
                created_neural_network: neural_net.NeuralNetwork = neural_net.NeuralNetwork(architecture_array)
                created_neural_network.build()

                return created_neural_network

        @staticmethod
        def create_neural_network_from_dictionary(architecture_dictionary: dict) -> neural_net.NeuralNetwork:
                neural_network_builder: architecture_builder.ArchitectureBuilder = architecture_builder.ArchitectureBuilder(architecture_dictionary)
                architecture_array: list[architecture_component.ArchitectureComponent] = neural_network_builder.build_components()

                created_neural_network: neural_net.NeuralNetwork = neural_net.NeuralNetwork(architecture_array)
                created_neural_network.build()

                return created_neural_network
