import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
    
    def train(self, patterns):
        # Hebbian learning rule: W_ij = sum(pattern_i * pattern_j)
        for pattern in patterns:
            pattern = np.array(pattern).reshape((self.num_neurons, 1))
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)  # Set diagonal elements to zero
    
    def predict(self, input_pattern, max_iterations=100):
        input_pattern = np.array(input_pattern).reshape((self.num_neurons, 1))
        prev_pattern = np.copy(input_pattern)
        for _ in range(max_iterations):
            new_pattern = np.sign(np.dot(self.weights, prev_pattern))
            if np.array_equal(new_pattern, prev_pattern):
                break
            prev_pattern = new_pattern
        return new_pattern.squeeze()

# Example usage
num_neurons = 4
hopfield_net = HopfieldNetwork(num_neurons)

# Training patterns (binary vectors)
training_patterns = [[1, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 1]]

# Train the Hopfield network
hopfield_net.train(training_patterns)

# Test the network with a corrupted pattern
corrupted_pattern = [1, 0, 1, 1]
recovered_pattern = hopfield_net.predict(corrupted_pattern)
print("Recovered pattern:", recovered_pattern)
