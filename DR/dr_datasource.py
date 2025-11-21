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
from plato.datasources.partitioners import base as partitioners_base


# --- 1. Custom PyTorch Dataset for DR Images and CSV ---
class DRDataset(Dataset):
    """A standard PyTorch Dataset that loads images and labels from a CSV."""
    def __init__(self, data_path, image_ids, labels, transform=None):
        self.data_path = data_path
        self.image_ids = image_ids  # List of image IDs (e.g., '000c1434d8d7')
        self.labels = labels        # List of corresponding diagnosis levels (0-4)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Get Label and Folder Name
        img_id = self.image_ids[idx]
        label = self.labels[idx] 
        # e.g., if label is 2, folder_name is 'Moderate'
        folder_name = LABEL_TO_FOLDER[label] 

        # 2. Construct the Image Path:
        # self.data_path -> points to '/path/to/dr_dataset_root/colored_images'
        # img_path -> /path/to/dr_dataset_root/colored_images/Moderate/img_id.png
        img_path = os.path.join(self.data_path, folder_name, f"{img_id}.png") 
        
        # 3. Load Image
        image = Image.open(img_path).convert("RGB")
        
        # 4. Apply Transformations
        if self.transform:
            image = self.transform(image)
        
        # 5. Return Image and Label
        label = torch.tensor(label, dtype=torch.long) 
        
        return image, label


# --- 2. Plato Data Source Class (Entry Point) ---
class DRDataSource(base.DataSource):
    """Plato's data source for the Diabetic Retinopathy dataset."""

    def __init__(self, client_id=0):
        super().__init__()
        
        # 1. Get configuration
        config = Config().data
        data_path = config.full_data_path

        # 2. Define Transforms (Using ImageNet mean/std for ResNet-18)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize((224, 224)), # Resize to standard ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]   
            ),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # 3. Load Metadata (Labels from CSV)
        labels_file = os.path.join(data_path, config.labels_filename)
        df = pd.read_csv(labels_file)
        
        image_ids = df['id_code'].values
        labels = df['diagnosis'].values
        
        # 4. Split data into Global Train/Test Sets
        # This split is done once and is reproducible due to random_seed
        train_indices, test_indices, _, _ = train_test_split(
            np.arange(len(df)), labels, 
            test_size=config.test_ratio, 
            shuffle=True, 
            stratify=labels,
            random_state=Config().system.random_seeds[0] 
        )
        
        # Extract full datasets using the split indices
        full_train_ids = image_ids[train_indices]
        full_train_labels = labels[train_indices]
        full_test_ids = image_ids[test_indices]
        full_test_labels = labels[test_indices]
        
        # Create full training and testing datasets
        # NOTE: train_dir/test_dir is assumed to point to the folder containing all the .png images (e.g., 'train_images').
        self.full_trainset = DRDataset(
            os.path.join(data_path, config.train_dir), 
            full_train_ids, full_train_labels, train_transform
        )
        self.testset = DRDataset(
            os.path.join(data_path, config.test_dir), 
            full_test_ids, full_test_labels, test_transform
        )

        # 5. Partition the Training Data for Clients
        # CRITICAL FIX: Reads 'sampler' and 'concentration' from the TOML config.
        self.partitioner = partitioners_base.Partitioner(
            self.full_trainset, 
            config.clients, 
            # Reads 'sampler = "iid"' from the config
            config.sampler, 
            # Reads 'concentration' (defaults to 1.0 if not specified for noniid)
            getattr(config, 'concentration', 1.0) 
        )

    def num_train_examples(self):
        """Returns the total number of training examples."""
        return len(self.full_trainset)

    def num_test_examples(self):
        """Returns the total number of testing examples."""
        return len(self.testset)

    def get_train_set(self, client_id, **kwargs):
        """Returns the partitioned training set for a specific client."""
        client_indices = self.partitioner.get_partition(client_id)
        
        # Wrap the full dataset with a Subset using the client's indices
        client_trainset = Subset(self.full_trainset, client_indices)
        
        # Ensures the correct transform is applied
        client_trainset.dataset.transform = self.full_trainset.transform
        
        return client_trainset

    def get_test_set(self, client_id, **kwargs):
        """Returns the global test set (used by all clients/server for validation)."""
        return self.testset