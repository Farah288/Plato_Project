import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from plato.config import Config
from plato.datasources import base
from plato.datasources import registry as datasources_registry # Necessary for registration

# --- 0. Custom IID Partitioner Class ---
class IIDPartitioner:
    """A simple class to partition dataset indices IID among clients."""
    def __init__(self, dataset_len, num_clients):
        # Create a list of all indices
        self.all_indices = np.arange(dataset_len)
        self.num_clients = num_clients
        self.client_partitions = {}
        self.split_dataset()

    def split_dataset(self):
        # Shuffle indices
        # FIXED: Access random_seeds from the [server] section
        np.random.shuffle(self.all_indices)
        
        # Calculate split points
        split_points = np.array_split(self.all_indices, self.num_clients)
        
        # Store partitions in a dictionary
        for client_id in range(self.num_clients):
            # Client IDs in Plato start from 1
            self.client_partitions[client_id + 1] = split_points[client_id]

    def get_partition(self, client_id):
        """Returns the indices allocated to a specific client ID."""
        if client_id not in self.client_partitions:
            # This should now only catch invalid IDs > total_clients
            raise ValueError(f"Client ID {client_id} not found in partitions.")
        return self.client_partitions[client_id]


# --- 1. Custom PyTorch Dataset for DR Images and CSV ---
LABEL_TO_FOLDER = {
    0: 'No_DR', 
    1: 'Mild', 
    2: 'Moderate', 
    3: 'Severe', 
    4: 'Proliferate_DR'
}

class DRDataset(Dataset):
    """A standard PyTorch Dataset that loads images and labels from a CSV."""
    def __init__(self, data_path, image_ids, labels, transform=None):
        self.data_path = data_path
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx] 
        folder_name = LABEL_TO_FOLDER[label] 

        # Assuming images are stored in subfolders named after their diagnosis level
        img_path = os.path.join(self.data_path, folder_name, f"{img_id}.png") 
        
        # This is the line that will trigger FileNotFoundError if paths are wrong
        image = Image.open(img_path).convert("RGB") 
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long) 
        
        return image, label


# --- 2. Plato Data Source Class (Entry Point) ---
class DataSource(base.DataSource):
    """Plato's data source for the Diabetic Retinopathy dataset."""

    def __init__(self, client_id=0):
        super().__init__()
        
        config = Config().data
        data_path = config.full_data_path

        # 2. Define Transforms (using ImageNet mean/std for ResNet-18)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), 
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 3. Load Metadata (Labels from CSV)
        labels_file = os.path.join(data_path, config.labels_filename)
        df = pd.read_csv(labels_file)
        
        # FIXED: Use 'image' and 'level' columns from CSV
        image_ids = df['image'].values
        labels = df['level'].values
        
        # 4. Split data into Global Train/Test Sets
        train_indices, test_indices, _, _ = train_test_split(
            np.arange(len(df)), labels, 
            test_size=config.test_ratio, 
            shuffle=True, 
            stratify=labels,
           
        )
        
        full_train_ids = image_ids[train_indices]
        full_train_labels = labels[train_indices]
        full_test_ids = image_ids[test_indices]
        full_test_labels = labels[test_indices]
        
        self.full_trainset = DRDataset(
            os.path.join(data_path, config.train_dir), 
            full_train_ids, full_train_labels, train_transform
        )
        self.testset = DRDataset(
            os.path.join(data_path, config.test_dir), 
            full_test_ids, full_test_labels, test_transform
        )

        # 5. Partition the Training Data for Clients
        num_clients = Config().clients.total_clients
        self.partitioner = IIDPartitioner(
            dataset_len=len(self.full_trainset), 
            num_clients=num_clients
        )


    def num_train_examples(self):
        return len(self.full_trainset)

    def num_test_examples(self):
        return len(self.testset)

    def get_train_set(self, client_id=None, **kwargs):
        """Returns the partitioned training set for a specific client."""
        
        # CRITICAL FIX: Handle None/0 client_id requests for the full dataset
        if client_id is None or client_id == 0:
             return self.full_trainset
        
        # Use the custom partitioner to get the correct indices
        client_indices = self.partitioner.get_partition(client_id)
        
        client_trainset = Subset(self.full_trainset, client_indices)
        client_trainset.dataset.transform = self.full_trainset.transform
        
        return client_trainset

    def get_test_set(self, client_id=None, **kwargs):
        """Returns the global test set (used by all clients/server for validation)."""
        return self.testset
    
