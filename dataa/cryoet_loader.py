# data/cryoet_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mrcfile
from pathlib import Path

class CryoETDataset(Dataset):
    """Dataset for CryoET volumes.
    
    This dataset loads MRC files from a directory structure
    and applies appropriate preprocessing and transforms.
    """
    
    def __init__(self, 
                 root_dir, 
                 transform=None, 
                 target_size=(64, 64, 64),
                 normalize=True):
        """Initialize CryoET dataset.
        
        Args:
            root_dir (str): Root directory containing dataset
            transform (callable, optional): Optional transform to be applied
            target_size (tuple): Target size for volume resizing
            normalize (bool): Whether to normalize data to [0, 1]
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        
        # Find all MRC files in the directory
        self.file_list = list(self.root_dir.glob("**/*.mrc"))
        
        if len(self.file_list) == 0:
            raise ValueError(f"No MRC files found in {root_dir}")
            
        # Extract labels from directory structure
        # Assuming the parent directory name is the class label
        self.labels = [int(file.parent.name) if file.parent.name.isdigit() 
                      else file.parent.name for file in self.file_list]
        
        # Convert string labels to indices
        self.unique_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.indices = [self.label_to_idx[label] for label in self.labels]
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Load and preprocess a volume.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (volume, label) where volume is a tensor and label is an integer
        """
        file_path = self.file_list[idx]
        label = self.indices[idx]
        
        # Load MRC file
        try:
            with mrcfile.open(file_path, permissive=True) as mrc:
                volume = mrc.data.astype(np.float32)
        except Exception as e:
            raise IOError(f"Error loading {file_path}: {e}")
        
        # Handle different MRC formats and orientations
        if volume.ndim == 4:  # Some MRC files might have multiple channels
            volume = volume[0]  # Take the first channel
        
        # Basic preprocessing
        if self.normalize:
            # Normalize to [0, 1]
            volume_min = volume.min()
            volume_max = volume.max()
            if volume_max > volume_min:  # Avoid division by zero
                volume = (volume - volume_min) / (volume_max - volume_min)
        
        # Resize to target size if needed
        # This is a basic implementation - you might need more sophisticated methods
        if volume.shape != self.target_size:
            # Simple resize using interpolation
            # For production, consider using something like scipy.ndimage.zoom
            from scipy.ndimage import zoom
            
            factors = (
                self.target_size[0] / volume.shape[0],
                self.target_size[1] / volume.shape[1],
                self.target_size[2] / volume.shape[2]
            )
            
            volume = zoom(volume, factors, order=1)
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if any
        if self.transform:
            volume_tensor = self.transform(volume_tensor)
        
        return volume_tensor, label
    
    def get_label_names(self):
        """Get the names of the labels.
        
        Returns:
            list: List of label names
        """
        return self.unique_labels

def get_cryoet_loaders(root_dir, batch_size=4, target_size=(64, 64, 64), 
                       num_workers=4, train_val_split=0.8, transform=None):
    """Create data loaders for CryoET data.
    
    Args:
        root_dir (str): Root directory containing dataset
        batch_size (int): Batch size
        target_size (tuple): Target size for volume resizing
        num_workers (int): Number of worker threads for loading
        train_val_split (float): Proportion of data to use for training
        transform (callable, optional): Optional transform to be applied
        
    Returns:
        tuple: (train_loader, val_loader) data loaders
    """
    # Create dataset
    dataset = CryoETDataset(root_dir, transform, target_size)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    train_size = int(train_val_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader