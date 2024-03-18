import tensorflow as tf
from tensorflow.keras import layers, models

# Define the Neocognitron model
def build_neocognitron(input_shape):
    model = models.Sequential()
    
    # Layer 1: Convolutional layer with ReLU activation
    model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Layer 2: Convolutional layer with ReLU activation
    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Layer 3: Flatten layer
    model.add(layers.Flatten())
    
    # Layer 4: Dense layer with ReLU activation
    model.add(layers.Dense(16, activation='relu'))
    
    # Layer 5: Output layer with softmax activation
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Create a Neocognitron model instance
neocognitron_model = build_neocognitron((28, 28, 1))  # Assuming input image size of 28x28 pixels

# Compile the model
neocognitron_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
neocognitron_model.summary()
