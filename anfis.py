import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class ANFIS(Model):
    def __init__(self, num_mfs, num_inputs):
        super(ANFIS, self).__init__()
        self.num_mfs = num_mfs
        self.num_inputs = num_inputs
        self.mf_parameters = self.add_weight(shape=(num_mfs, num_inputs),
                                              initializer='random_normal',
                                              trainable=True)
        self.antecedent_layer = layers.Lambda(lambda x: tf.exp(-tf.square(x - self.mf_parameters)))
        self.consequent_layer = layers.Dense(1)

    def call(self, inputs):
        membership_values = self.antecedent_layer(inputs)
        rule_outputs = tf.reduce_prod(membership_values, axis=2)
        weighted_rule_outputs = rule_outputs * self.mf_parameters[:, -1]
        normalized_weights = tf.nn.softmax(weighted_rule_outputs, axis=1)
        aggregated_outputs = tf.reduce_sum(normalized_weights * rule_outputs, axis=1)
        return self.consequent_layer(aggregated_outputs)

# Generate sample data
np.random.seed(42)
X_train = np.random.rand(1000, 4)
y_train = np.sum(X_train, axis=1, keepdims=True)

# Define and compile the ANFIS model
num_mfs = 5  # Number of membership functions
num_inputs = X_train.shape[1]
anfis_model = ANFIS(num_mfs, num_inputs)
anfis_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the ANFIS model
anfis_model.fit(X_train, y_train, epochs=50, batch_size=32)

# Test the model
X_test = np.random.rand(10, 4)
y_test = np.sum(X_test, axis=1, keepdims=True)
test_loss = anfis_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
