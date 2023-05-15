import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model architecture
def transformer_model(max_len, vocab_size):
    # Input for word vectors
    input_layer = layers.Input(shape=(50,))
    
    # Transformer layers
    embedding_layer = layers.Embedding(input_dim=37500, output_dim=64)(input_layer)  # Adjust output_dim as needed
    transformer_block = layers.Transformer(num_heads=2, d_model=64, dff=128)(embedding_layer)  # Adjust parameters as needed
    
    # Global average pooling
    pooling_layer = layers.GlobalAveragePooling1D()(transformer_block)
    
    # Dense layers for classification
    dense_layer = layers.Dense(64, activation='relu')(pooling_layer)
    output_layer = layers.Dense(1, activation='sigmoid')(dense_layer)
    
    # Create and compile the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = transformer_model(50,59819)