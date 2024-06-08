import logging
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualProcessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        logger.info("VisualProcessor initialized with image_size=%s", image_size)

    def load_image(self, file_path):
        """
        Loads an image file.
        """
        try:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
            logger.info("Loaded image file: %s", file_path)
            return image
        except Exception as e:
            logger.error("Failed to load image file %s: %s", file_path, str(e))
            return None

    def preprocess_image(self, image):
        """
        Preprocesses the image for model input.
        """
        try:
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            logger.info("Preprocessed image for model input")
            return image
        except Exception as e:
            logger.error("Failed to preprocess image: %s", str(e))
            return None

    def normalize_image(self, image):
        """
        Normalizes the image.
        """
        try:
            image = image / 255.0
            logger.info("Normalized image")
            return image
        except Exception as e:
            logger.error("Failed to normalize image: %s", str(e))
            return None

    def split_data(self, images, labels, test_size=0.2, random_state=42):
        """
        Splits data into training and testing sets.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)
            logger.info("Split data into training and testing sets with test_size=%f", test_size)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error("Failed to split data: %s", str(e))
            return np.array([]), np.array([]), np.array([]), np.array([])

# Example usage
if __name__ == "__main__":
    file_path = "path/to/image/file.jpg"
    labels = ["class1", "class2"]

    processor = VisualProcessor()
    image = processor.load_image(file_path)
    
    if image is not None:
        preprocessed_image = processor.preprocess_image(image)
        normalized_image = processor.normalize_image(image)
        
        # Simulated labels for demonstration purposes
        labels = np.array([0, 1] * (len(normalized_image) // 2))
        
        X_train, X_test, y_train, y_test = processor.split_data(np.array([normalized_image]), labels)

        print("Image Shape:", image.shape)
        print("Preprocessed Image Shape:", preprocessed_image.shape)
        print("Normalized Image Shape:", normalized_image.shape)
        print("Train/Test Split:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
