import numpy as np

class CognitronLayer:
    def __init__(self, num_neurons, input_size, activation_function):
        self.weights = np.random.rand(num_neurons, input_size)
        self.bias = np.random.rand(num_neurons, 1)
        self.activation_function = activation_function
    
    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation_function(np.dot(self.weights, input_data) + self.bias)
        return self.output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Create a simple Cognitron network
layer1 = CognitronLayer(num_neurons=4, input_size=2, activation_function=sigmoid)
layer2 = CognitronLayer(num_neurons=1, input_size=4, activation_function=sigmoid)

# Input data
input_data = np.array([[0.2], [0.7]])

# Forward pass
output1 = layer1.forward(input_data)
output2 = layer2.forward(output1)

print("Output of layer 1:", output1)
print("Final output:", output2)
