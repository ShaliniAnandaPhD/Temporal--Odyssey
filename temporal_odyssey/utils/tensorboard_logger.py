import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorBoardLogger:
    def __init__(self, log_dir="logs/"):
        """
        Initialize the TensorBoardLogger.

        Args:
            log_dir (str): Directory to save TensorBoard logs.
        """
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        logger.info("TensorBoardLogger initialized with log directory: %s", log_dir)

    def setup_tensorboard(self):
        """
        Setup TensorBoard logging.
        """
        self.writer.set_as_default()
        logger.info("TensorBoard logging setup complete.")

    def log_to_tensorboard(self, tag, value, step):
        """
        Log metrics and graphs to TensorBoard.

        Args:
            tag (str): The name of the metric.
            value (float): The value of the metric.
            step (int): The current step or epoch.
        """
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        logger.info(f"Logged {tag}: {value} at step {step}")

# Example usage
if __name__ == "__main__":
    import time

    tb_logger = TensorBoardLogger(log_dir="logs/example")
    tb_logger.setup_tensorboard()

    for step in range(100):
        # Simulate some metrics
        loss = 0.1 * step
        accuracy = 1.0 - 0.01 * step

        tb_logger.log_to_tensorboard("loss", loss, step)
        tb_logger.log_to_tensorboard("accuracy", accuracy, step)

        time.sleep(0.1)  # Simulate time between logging
