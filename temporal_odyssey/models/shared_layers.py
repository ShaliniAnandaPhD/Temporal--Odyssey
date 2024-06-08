import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model

def build_shared_layers(input_shape):
    """
    Construct shared convolutional and dense layers for use in hybrid learning models.
    
    Args:
        input_shape (tuple): Shape of the input.
    
    Returns:
        Model: A Keras Model consisting of shared layers.
    """
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    model = Model(inputs, x, name="shared_layers_model")
    
    return model

# Example usage
if __name__ == "__main__":
    input_shape = (64, 64, 3)  # Example input shape for an image with size 64x64 and 3 color channels
    shared_model = build_shared_layers(input_shape)
    shared_model.summary()
