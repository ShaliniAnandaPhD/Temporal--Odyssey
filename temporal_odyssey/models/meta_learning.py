import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaLearning:
    def __init__(self, input_shape, output_shape, meta_lr=0.001, inner_lr=0.01, num_inner_updates=5, meta_batch_size=32):
        """
        Initialize the MetaLearning class with MAML algorithm.

        Parameters:
        input_shape (tuple): Shape of the input data.
        output_shape (int): Number of output units.
        meta_lr (float): Learning rate for the meta-optimizer. Default is 0.001.
        inner_lr (float): Learning rate for the inner updates. Default is 0.01.
        num_inner_updates (int): Number of inner loop updates. Default is 5.
        meta_batch_size (int): Batch size for the meta-learning process. Default is 32.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_updates = num_inner_updates
        self.meta_batch_size = meta_batch_size
        self.model = self.build_model()
        self.meta_optimizer = Adam(learning_rate=self.meta_lr)
        logger.info("MetaLearning initialized with MAML algorithm.")

    def build_model(self):
        """
        Build the neural network model.

        Returns:
        Model: Compiled Keras model.
        """
        inputs = Input(shape=self.input_shape)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.output_shape, activation='softmax')(x)
        model = Model(inputs, outputs)
        logger.info("Model built successfully.")
        return model

    def meta_train(self, tasks, epochs=100):
        """
        Meta-train the model using MAML on a set of tasks.

        Parameters:
        tasks (list): List of task datasets for meta-training.
        epochs (int): Number of meta-training epochs. Default is 100.
        """
        for epoch in range(epochs):
            meta_gradients = defaultdict(list)
            for task in tasks:
                task_gradients = self.inner_update(task)
                for layer, grad in task_gradients.items():
                    meta_gradients[layer].append(grad)

            # Average gradients over tasks and apply meta-gradient update
            averaged_gradients = {layer: np.mean(gradients, axis=0) for layer, gradients in meta_gradients.items()}
            self.apply_meta_gradients(averaged_gradients)
            logger.info(f"Epoch {epoch+1}/{epochs} completed.")

    def inner_update(self, task):
        """
        Perform the inner loop updates on a single task.

        Parameters:
        task (tuple): Tuple containing task-specific training data (x_train, y_train).

        Returns:
        dict: Gradients with respect to the initial model parameters.
        """
        x_train, y_train = task
        with tf.GradientTape() as tape:
            predictions = self.model(x_train)
            loss = tf.keras.losses.categorical_crossentropy(y_train, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        k_gradients = {layer.name: grad for layer, grad in zip(self.model.layers, gradients)}

        for _ in range(self.num_inner_updates):
            k_gradients = self.apply_inner_update(x_train, y_train, k_gradients)

        return k_gradients

    def apply_inner_update(self, x_train, y_train, k_gradients):
        """
        Apply an inner update using the provided gradients.

        Parameters:
        x_train (ndarray): Training data.
        y_train (ndarray): Training labels.
        k_gradients (dict): Current gradients for the model.

        Returns:
        dict: Updated gradients.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(x_train)
            loss = tf.keras.losses.categorical_crossentropy(y_train, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        updated_gradients = {layer.name: grad - self.inner_lr * k_gradients[layer.name]
                             for layer, grad in zip(self.model.layers, gradients)}

        return updated_gradients

    def apply_meta_gradients(self, meta_gradients):
        """
        Apply the meta-gradient updates to the model.

        Parameters:
        meta_gradients (dict): Meta-gradients to apply.
        """
        for layer in self.model.layers:
            if layer.name in meta_gradients:
                layer.kernel.assign_sub(self.meta_lr * meta_gradients[layer.name])

    def meta_test(self, task):
        """
        Test the model on a specific task using meta-learned parameters.

        Parameters:
        task (tuple): Tuple containing task-specific test data (x_test, y_test).

        Returns:
        float: Test accuracy.
        """
        x_test, y_test = task
        predictions = self.model(x_test)
        accuracy = tf.keras.metrics.categorical_accuracy(y_test, predictions)
        logger.info(f"Test accuracy: {np.mean(accuracy)}")
        return np.mean(accuracy)

# Example usage
if __name__ == "__main__":
    # Assuming the input shape is (784,) for flattened 28x28 images and 10 output classes
    input_shape = (784,)
    output_shape = 10

    meta_learning = MetaLearning(input_shape, output_shape)

    # Assuming we have a list of tasks for meta-training and a specific task for meta-testing
    tasks = [(np.random.random((100, 784)), np.random.random((100, 10))) for _ in range(5)]
    test_task = (np.random.random((100, 784)), np.random.random((100, 10)))

    # Meta-train the model
    meta_learning.meta_train(tasks)

    # Meta-test the model
    meta_learning.meta_test(test_task)

