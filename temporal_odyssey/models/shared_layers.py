import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SharedLayers:
    def __init__(self, input_shape, output_shape, num_layers=3, units_per_layer=128, activation='relu', dropout_rate=0.5, l1_reg=0.01, l2_reg=0.01):
        """
        Initialize the SharedLayers class.

        Parameters:
        input_shape (tuple): Shape of the input data.
        output_shape (int): Number of output units.
        num_layers (int): Number of hidden layers. Default is 3.
        units_per_layer (int): Number of units per hidden layer. Default is 128.
        activation (str): Activation function to use. Default is 'relu'.
        dropout_rate (float): Dropout rate for regularization. Default is 0.5.
        l1_reg (float): L1 regularization factor. Default is 0.01.
        l2_reg (float): L2 regularization factor. Default is 0.01.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_layers = num_layers
        self.units_per_layer = units_per_layer
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.model = self.build_model()
        logger.info("SharedLayers initialized with specified architecture and parameters.")

    def build_model(self):
        """
        Build the shared layer model with the specified architecture and parameters.

        Returns:
        Model: Compiled Keras model with shared layers.
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs

        for i in range(self.num_layers):
            x = Dense(self.units_per_layer, activation=self.activation, kernel_regularizer=l1(self.l1_reg) + l2(self.l2_reg))(x)
            x = Dropout(self.dropout_rate)(x)
            x = LayerNormalization()(x)

        outputs = Dense(self.output_shape, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Shared layer model built successfully.")
        return model

    def adapt_layers(self, task_specific_units):
        """
        Adapt the shared layers based on the specific requirements of each task.

        Parameters:
        task_specific_units (int): Number of units for the task-specific layer.

        Returns:
        Model: Compiled Keras model with adapted layers.
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        x = self.model.layers[1](inputs)  # Use the first hidden layer from the shared model
        for layer in self.model.layers[2:]:  # Skip the input layer
            x = layer(x)

        task_specific_output = Dense(task_specific_units, activation='softmax')(x)
        adapted_model = Model(inputs, task_specific_output)
        adapted_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Shared layer model adapted successfully for the specific task.")
        return adapted_model

    def pretrain_on_large_dataset(self, large_dataset, epochs=10, batch_size=32):
        """
        Pre-train the shared layers on a large dataset.

        Parameters:
        large_dataset (tf.data.Dataset): The large dataset to pre-train on.
        epochs (int): Number of epochs for pre-training. Default is 10.
        batch_size (int): Batch size for pre-training. Default is 32.
        """
        self.model.fit(large_dataset, epochs=epochs, batch_size=batch_size)
        logger.info("Shared layer model pre-trained on large dataset.")

    def fine_tune_on_task(self, task_dataset, epochs=10, batch_size=32):
        """
        Fine-tune the shared layers on the specific task dataset.

        Parameters:
        task_dataset (tf.data.Dataset): The task-specific dataset to fine-tune on.
        epochs (int): Number of epochs for fine-tuning. Default is 10.
        batch_size (int): Batch size for fine-tuning. Default is 32.
        """
        self.model.fit(task_dataset, epochs=epochs, batch_size=batch_size)
        logger.info("Shared layer model fine-tuned on task-specific dataset.")

# Example usage
if __name__ == "__main__":
    # Assuming the input shape is (784,) for flattened 28x28 images and 10 output classes
    input_shape = (784,)
    output_shape = 10

    shared_layers = SharedLayers(input_shape, output_shape)

    # Assuming we have a large pre-training dataset and a task-specific dataset
    large_dataset = tf.data.Dataset.from_tensor_slices((np.random.random((10000, 784)), np.random.random((10000, 10))))
    task_dataset = tf.data.Dataset.from_tensor_slices((np.random.random((1000, 784)), np.random.random((1000, 10))))

    # Pre-train the shared layers on a large dataset
    shared_layers.pretrain_on_large_dataset(large_dataset)

    # Fine-tune the shared layers on the task-specific dataset
    shared_layers.fine_tune_on_task(task_dataset)

    # Adapt the shared layers for a specific task
    adapted_model = shared_layers.adapt_layers(task_specific_units=5)
    print("Adapted model summary:")
    adapted_model.summary()

