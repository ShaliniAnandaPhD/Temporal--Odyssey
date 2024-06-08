import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalMetrics:
    def __init__(self):
        self.results = {}
        logger.info("MultimodalMetrics initialized.")

    def update(self, true_labels, pred_labels, average='weighted', prefix=''):
        """
        Update metrics with new predictions.

        Args:
            true_labels (array-like): True labels.
            pred_labels (array-like): Predicted labels.
            average (str): Averaging method for metrics.
            prefix (str): Prefix for metric keys.
        """
        self.results[prefix + 'accuracy'] = accuracy_score(true_labels, pred_labels)
        self.results[prefix + 'precision'] = precision_score(true_labels, pred_labels, average=average)
        self.results[prefix + 'recall'] = recall_score(true_labels, pred_labels, average=average)
        self.results[prefix + 'f1_score'] = f1_score(true_labels, pred_labels, average=average)
        
        if len(np.unique(true_labels)) == 2:  # Binary classification
            self.results[prefix + 'roc_auc'] = roc_auc_score(true_labels, pred_labels)
        
        logger.info(f"Metrics updated: {self.results}")

    def update_multilabel(self, true_labels, pred_labels, average='macro', prefix=''):
        """
        Update metrics for multi-label classification.

        Args:
            true_labels (array-like): True labels.
            pred_labels (array-like): Predicted labels.
            average (str): Averaging method for metrics.
            prefix (str): Prefix for metric keys.
        """
        self.results[prefix + 'accuracy'] = accuracy_score(true_labels, pred_labels)
        self.results[prefix + 'precision'] = precision_score(true_labels, pred_labels, average=average)
        self.results[prefix + 'recall'] = recall_score(true_labels, pred_labels, average=average)
        self.results[prefix + 'f1_score'] = f1_score(true_labels, pred_labels, average=average)
        self.results[prefix + 'classification_report'] = classification_report(true_labels, pred_labels)
        
        logger.info(f"Multi-label metrics updated: {self.results}")

    def update_confusion_matrix(self, true_labels, pred_labels, labels, prefix=''):
        """
        Update confusion matrix for the predictions.

        Args:
            true_labels (array-like): True labels.
            pred_labels (array-like): Predicted labels.
            labels (list): List of labels.
            prefix (str): Prefix for metric keys.
        """
        self.results[prefix + 'confusion_matrix'] = confusion_matrix(true_labels, pred_labels, labels=labels)
        logger.info(f"Confusion matrix updated for {prefix}")

    def compute_weighted_metrics(self, true_labels, pred_labels, weights, prefix=''):
        """
        Compute weighted metrics to account for class imbalance.

        Args:
            true_labels (array-like): True labels.
            pred_labels (array-like): Predicted labels.
            weights (array-like): Weights for each class.
            prefix (str): Prefix for metric keys.
        """
        self.results[prefix + 'weighted_accuracy'] = np.average(accuracy_score(true_labels, pred_labels), weights=weights)
        self.results[prefix + 'weighted_precision'] = np.average(precision_score(true_labels, pred_labels, average=None), weights=weights)
        self.results[prefix + 'weighted_recall'] = np.average(recall_score(true_labels, pred_labels, average=None), weights=weights)
        self.results[prefix + 'weighted_f1_score'] = np.average(f1_score(true_labels, pred_labels, average=None), weights=weights)
        
        logger.info(f"Weighted metrics updated: {self.results}")

    def visualize_metrics(self):
        """
        Visualize the metrics.
        """
        df = pd.DataFrame.from_dict(self.results, orient='index', columns=['Value'])
        df.plot(kind='bar', figsize=(12, 6))
        plt.title('Performance Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.show()
        logger.info("Metrics visualized.")

    def visualize_confusion_matrix(self, prefix=''):
        """
        Visualize the confusion matrix.

        Args:
            prefix (str): Prefix for metric keys.
        """
        cm = self.results.get(prefix + 'confusion_matrix')
        if cm is not None:
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix {prefix}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            logger.info("Confusion matrix visualized.")

    def statistical_test(self, metric1, metric2, test='t-test'):
        """
        Perform statistical tests to compare two sets of metrics.

        Args:
            metric1 (array-like): First set of metrics.
            metric2 (array-like): Second set of metrics.
            test (str): Type of statistical test ('t-test' or 'anova').

        Returns:
            float: p-value from the statistical test.
        """
        if test == 't-test':
            _, p_value = stats.ttest_ind(metric1, metric2)
        elif test == 'anova':
            _, p_value = stats.f_oneway(metric1, metric2)
        else:
            raise ValueError("Invalid test type. Choose 't-test' or 'anova'.")
        
        logger.info(f"Statistical test ({test}) performed with p-value: {p_value}")
        return p_value

    def log_to_mlflow(self):
        """
        Log the metrics to MLflow.
        """
        for key, value in self.results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            elif isinstance(value, np.ndarray):
                mlflow.log_artifact(pd.DataFrame(value).to_csv(index=False), f"{key}.csv")
            else:
                mlflow.log_param(key, value)
        logger.info("Metrics logged to MLflow.")

# Example usage
if __name__ == "__main__":
    true_labels = np.random.randint(0, 2, 100)
    pred_labels = np.random.randint(0, 2, 100)
    weights = np.random.rand(2)

    mm_metrics = MultimodalMetrics()
    mm_metrics.update(true_labels, pred_labels)
    mm_metrics.update_confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    mm_metrics.compute_weighted_metrics(true_labels, pred_labels, weights)
    mm_metrics.visualize_metrics()
    mm_metrics.visualize_confusion_matrix()
    mm_metrics.statistical_test(true_labels, pred_labels)
    mm_metrics.log_to_mlflow()

