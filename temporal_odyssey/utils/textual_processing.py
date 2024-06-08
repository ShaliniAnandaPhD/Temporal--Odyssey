import logging
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, vocab_size=10000, max_seq_length=100):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.label_encoder = LabelEncoder()
        logger.info("TextProcessor initialized with vocab_size=%d and max_seq_length=%d", vocab_size, max_seq_length)

    def fit_tokenizer(self, texts):
        """
        Fits the tokenizer on the provided texts.
        """
        try:
            self.tokenizer.fit_on_texts(texts)
            logger.info("Tokenizer fitted on %d texts", len(texts))
        except Exception as e:
            logger.error("Failed to fit tokenizer: %s", str(e))

    def texts_to_sequences(self, texts):
        """
        Converts texts to sequences of integers.
        """
        try:
            sequences = self.tokenizer.texts_to_sequences(texts)
            logger.info("Converted %d texts to sequences", len(texts))
            return sequences
        except Exception as e:
            logger.error("Failed to convert texts to sequences: %s", str(e))
            return []

    def pad_sequences(self, sequences):
        """
        Pads sequences to ensure uniform length.
        """
        try:
            padded_sequences = pad_sequences(sequences, maxlen=self.max_seq_length)
            logger.info("Padded sequences to max length %d", self.max_seq_length)
            return padded_sequences
        except Exception as e:
            logger.error("Failed to pad sequences: %s", str(e))
            return np.array([])

    def encode_labels(self, labels):
        """
        Encodes categorical labels into integers.
        """
        try:
            encoded_labels = self.label_encoder.fit_transform(labels)
            logger.info("Encoded %d labels", len(labels))
            return encoded_labels
        except Exception as e:
            logger.error("Failed to encode labels: %s", str(e))
            return np.array([])

    def split_data(self, sequences, labels, test_size=0.2, random_state=42):
        """
        Splits data into training and testing sets.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, random_state=random_state)
            logger.info("Split data into training and testing sets with test_size=%f", test_size)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error("Failed to split data: %s", str(e))
            return np.array([]), np.array([]), np.array([]), np.array([])

# Example usage
if __name__ == "__main__":
    texts = ["This is a test sentence.", "Another test sentence for tokenizer."]
    labels = ["class1", "class2"]

    processor = TextProcessor()
    processor.fit_tokenizer(texts)
    sequences = processor.texts_to_sequences(texts)
    padded_sequences = processor.pad_sequences(sequences)
    encoded_labels = processor.encode_labels(labels)
    X_train, X_test, y_train, y_test = processor.split_data(padded_sequences, encoded_labels)

    print("Sequences:", sequences)
    print("Padded Sequences:", padded_sequences)
    print("Encoded Labels:", encoded_labels)
    print("Train/Test Split:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
