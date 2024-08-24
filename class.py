import random

class Node:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.predicted_value = 0
        self.output_value = 0
        self.bias = random.uniform(-0.1, 0.1)  # Initialize with small random biases
        self.inputs = []
        self.outputs = []
        self.bias_gradient_sum = 0 

class Connection:
    def __init__(self, synapse_id):
        self.id = synapse_id
        self.weight = random.uniform(-0.1, 0.1) 
        self.input_value = 0
        self.weight_gradient_sum = 0  

"""neuron_id = 1
synapse_id = 1"""

def initialize_input_neurons(num_neurons):
    global neuron_id
    return [Node(neuron_id + i) for i in range(num_neurons)]

def initialize_hidden_neurons(layers, neurons_per_layer, input_neurons):
    global neuron_id
    network = [input_neurons]
    for _ in range(layers):
        layer = [Node(neuron_id + i) for i in range(neurons_per_layer)]
        network.append(layer)
        neuron_id += neurons_per_layer
    return network

def initialize_output_neurons(num_neurons, network):
    global neuron_id
    output_layer = [Node(neuron_id + i) for i in range(num_neurons)]
    network.append(output_layer)
    return network

def initialize_synapses(network):
    global synapse_id
    for layer_idx in range(len(network) - 1):
        for neuron in network[layer_idx]:
            for next_neuron in network[layer_idx + 1]:
                synapse = Connection(synapse_id)
                synapse_id += 1
                neuron.outputs.append(synapse)
                next_neuron.inputs.append(synapse)
    return network

def set_input_values(input_values, network):
    if len(input_values) != len(network[0]):
        raise ValueError("Input values size doesn't match the number of input neurons")
    for neuron, value in zip(network[0], input_values):
        neuron.predicted_value = value

def set_target_values(target_values, network):
    if len(target_values) != len(network[-1]):
        raise ValueError("Target values size doesn't match the number of output neurons")
    for neuron, target in zip(network[-1], target_values):
        neuron.output_value = target

def forward_propagation(network):
    for layer in network[1:]:
        for neuron in layer:
            total_input = sum(synapse.input_value for synapse in neuron.inputs) + neuron.bias
            neuron.predicted_value = max(0, total_input)  # ReLU activation
            for synapse in neuron.outputs:
                synapse.input_value = neuron.predicted_value * synapse.weight

def backward_propagation(network):
    for neuron in network[-1]:
        error = neuron.predicted_value - neuron.output_value
        activation_derivative = 1 if neuron.predicted_value > 0 else 0  # ReLU derivative
        for synapse in neuron.inputs:
            synapse.weight_gradient_sum += error * activation_derivative * synapse.input_value
        neuron.bias_gradient_sum += error * activation_derivative

    # Hidden layers
    for layer in reversed(network[:-1]):
        for neuron in layer:
            gradient_sum = sum(synapse.weight * synapse.weight_gradient_sum for synapse in neuron.outputs)
            neuron_gradient = gradient_sum * (1 if neuron.predicted_value > 0 else 0)  
            for synapse in neuron.inputs:
                synapse.weight_gradient_sum += neuron_gradient * synapse.input_value
            neuron.bias_gradient_sum += neuron_gradient

def update_weights_and_biases(network, learning_rate, num_examples):
    for layer in network[1:]:
        for neuron in layer:
            for synapse in neuron.inputs:
                synapse.weight -= learning_rate * synapse.weight_gradient_sum / num_examples
                synapse.weight_gradient_sum = 0  
            neuron.bias -= learning_rate * neuron.bias_gradient_sum / num_examples
            neuron.bias_gradient_sum = 0  


def train(network, input_data, target_data, learning_rate, epochs):
    for epoch in range(epochs):
        for inputs, targets in zip(input_data, target_data):
            set_input_values(inputs, network)
            set_target_values(targets, network)
            forward_propagation(network)
            backward_propagation(network)
    
        update_weights_and_biases(network, learning_rate, len(input_data))

