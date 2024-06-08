import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.metrics import AUC, Precision, Recall

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalMetrics:
    def __init__(self):
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        self.loss = tf.keras.metrics.Mean(name='loss')
        self.auc = AUC(name='auc')
        self.precision = Precision(name='precision')
        self.recall = Recall(name='recall')
        self.true_labels = []
        self.predicted_labels = []

    def update_metrics(self, y_true, y_pred, loss):
        """Updates the metrics with the latest batch of predictions and labels."""
        self.accuracy.update_state(y_true, y_pred)
        self.loss.update_state(loss)
        self.auc.update_state(y_true, y_pred)
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)
        self.true_labels.extend(np.argmax(y_true, axis=1))
        self.predicted_labels.extend(np.argmax(y_pred, axis=1))
        logger.info(f"Metrics updated: loss={loss}, accuracy={self.accuracy.result().numpy()}")

    def reset_metrics(self):
        """Resets the metrics to their initial state."""
        self.accuracy.reset_states()
        self.loss.reset_states()
        self.auc.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.true_labels = []
        self.predicted_labels = []

    def log_metrics(self):
        """Logs the current values of the metrics."""
        logger.info(f"Loss: {self.loss.result().numpy()}")
        logger.info(f"Accuracy: {self.accuracy.result().numpy()}")
        logger.info(f"AUC: {self.auc.result().numpy()}")
        logger.info(f"Precision: {self.precision.result().numpy()}")
        logger.info(f"Recall: {self.recall.result().numpy()}")

    def plot_confusion_matrix(self):
        """Plots the confusion matrix of the true and predicted labels."""
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(set(self.true_labels)))
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        logger.info("Confusion matrix plotted.")

    def classification_report(self):
        """Generates a classification report."""
        report = classification_report(self.true_labels, self.predicted_labels, target_names=[str(i) for i in range(len(set(self.true_labels)))])
        logger.info(f"\nClassification Report:\n{report}")

    def save_metrics(self, filepath):
        """Saves the metrics to a file."""
        try:
            metrics_data = {
                'accuracy': self.accuracy.result().numpy(),
                'loss': self.loss.result().numpy(),
                'auc': self.auc.result().numpy(),
                'precision': self.precision.result().numpy(),
                'recall': self.recall.result().numpy(),
                'true_labels': self.true_labels,
                'predicted_labels': self.predicted_labels
            }
            np.save(filepath, metrics_data)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def load_metrics(self, filepath):
        """Loads the metrics from a file."""
        try:
            metrics_data = np.load(filepath, allow_pickle=True).item()
            self.accuracy.update_state(metrics_data['accuracy'])
            self.loss.update_state(metrics_data['loss'])
            self.auc.update_state(metrics_data['auc'])
            self.precision.update_state(metrics_data['precision'])
            self.recall.update_state(metrics_data['recall'])
            self.true_labels = metrics_data['true_labels']
            self.predicted_labels = metrics_data['predicted_labels']
            logger.info(f"Metrics loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")


# Example usage
if __name__ == "__main__":
    # Simulated example data
    y_true = np.random.randint(0, 2, size=(100, 3))
    y_pred = np.random.rand(100, 3)
    loss = np.random.rand(1)


  
    metrics = MultiModalMetrics()
    metrics.update_metrics(y_true, y_pred, loss)
    metrics.log_metrics()
    metrics.plot_confusion_matrix()
    metrics.classification_report()
    metrics.save_metrics('metrics.npy')
    metrics.load_metrics('metrics.npy')
