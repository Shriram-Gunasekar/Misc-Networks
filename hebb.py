import numpy as np

class HebbNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((output_size, input_size))
    
    def train(self, input_data):
        # Hebbian learning rule: W_ij = X_i * Y_j
        for input_vector in input_data:
            input_vector = np.array(input_vector).reshape((1, self.input_size))
            self.weights += np.dot(input_vector.T, input_vector)
    
    def predict(self, input_data):
        input_vector = np.array(input_data).reshape((1, self.input_size))
        output_vector = np.dot(input_vector, self.weights.T)
        return output_vector.squeeze()

# Example usage
input_size = 3
output_size = 2
hebb_net = HebbNetwork(input_size, output_size)

# Training data (patterns)
training_data = [[1, 0, 0], [0, 1, 0], [1, 1, 1]]

# Train the Hebbian network
hebb_net.train(training_data)

# Test the network with a new input
new_input = [1, 0, 1]
output = hebb_net.predict(new_input)
print("Output:", output)
