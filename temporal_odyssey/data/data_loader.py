import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Custom Dataset for loading and preprocessing data.

        Args:
            data (np.array): Array of data features.
            labels (np.array): Array of data labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data(file_path, label_column, batch_size=32, shuffle=True):
    """
    Load and preprocess data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        label_column (str): Name of the column to be used as labels.
        batch_size (int, optional): Batch size for the data loader. Default is 32.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.

    Returns:
        DataLoader: PyTorch DataLoader with the processed data.
    """
    # Load data from CSV file
    df = pd.read_csv(file_path)

    # Separate features and labels
    labels = df[label_column].values
    data = df.drop(columns=[label_column]).values

    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Convert to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Create a custom dataset
    dataset = CustomDataset(data, labels)

    # Create a data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def data_augmentation(sample):
    """
    Apply data augmentation techniques to a sample.

    Args:
        sample (dict): A dictionary containing data and label.

    Returns:
        dict: The augmented sample.
    """
    data, label = sample['data'], sample['label']
    
    # Example augmentation 1: Adding Gaussian noise
    noise = torch.randn_like(data) * 0.1
    augmented_data1 = data + noise
    
    # Example augmentation 2: Random rotation
    angle = torch.rand(1) * 360
    rotation_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                                    [torch.sin(angle), torch.cos(angle)]])
    augmented_data2 = torch.matmul(data, rotation_matrix)
    
    # Combine augmented data
    augmented_data = torch.stack([augmented_data1, augmented_data2])
    
    return {'data': augmented_data, 'label': label}

# Example usage
if __name__ == "__main__":
    file_path = 'data/iris.csv'
    label_column = 'species'
    batch_size = 32

    # Load data
    data_loader = load_data(file_path, label_column, batch_size)

    # Iterate through the data loader
    for batch in data_loader:
        data, labels = batch['data'], batch['label']
        print(f"Data shape: {data.shape}")
        print(f"Labels: {labels}")

        # Apply data augmentation
        augmented_sample = data_augmentation(batch)
        augmented_data, augmented_labels = augmented_sample['data'], augmented_sample['label']
        print(f"Augmented data shape: {augmented_data.shape}")
        print(f"Augmented labels: {augmented_labels}")
        print("---")
