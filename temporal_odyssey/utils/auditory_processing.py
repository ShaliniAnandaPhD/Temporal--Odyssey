import logging
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditoryProcessor:
    def __init__(self, sample_rate=22050, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        logger.info("AuditoryProcessor initialized with sample_rate=%d and n_mfcc=%d", sample_rate, n_mfcc)

    def load_audio(self, file_path):
        """
        Loads an audio file.
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            logger.info("Loaded audio file: %s", file_path)
            return audio, sr
        except Exception as e:
            logger.error("Failed to load audio file %s: %s", file_path, str(e))
            return None, None

    def extract_features(self, audio):
        """
        Extracts MFCC features from audio data.
        """
        try:
            mfccs = librosa.feature.mfcc(audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            logger.info("Extracted MFCC features from audio data")
            return mfccs
        except Exception as e:
            logger.error("Failed to extract MFCC features: %s", str(e))
            return None

    def normalize_features(self, features):
        """
        Normalizes the MFCC features.
        """
        try:
            norm_features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
            logger.info("Normalized MFCC features")
            return norm_features
        except Exception as e:
            logger.error("Failed to normalize MFCC features: %s", str(e))
            return None

    def split_data(self, features, labels, test_size=0.2, random_state=42):
        """
        Splits data into training and testing sets.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
            logger.info("Split data into training and testing sets with test_size=%f", test_size)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error("Failed to split data: %s", str(e))
            return np.array([]), np.array([]), np.array([]), np.array([])

# Example usage
if __name__ == "__main__":
    file_path = "path/to/audio/file.wav"
    labels = ["class1", "class2"]

    processor = AuditoryProcessor()
    audio, sr = processor.load_audio(file_path)
    
    if audio is not None:
        features = processor.extract_features(audio)
        norm_features = processor.normalize_features(features)
        
        # Simulated labels for demonstration purposes
        labels = np.array([0, 1] * (len(norm_features[0]) // 2))
        
        X_train, X_test, y_train, y_test = processor.split_data(norm_features.T, labels)

        print("Features Shape:", features.shape)
        print("Normalized Features Shape:", norm_features.shape)
        print("Train/Test Split:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
