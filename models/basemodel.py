# models/base_model.py
import torch
import torch.nn as nn
import os
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Base class for all models in the project.
    
    Provides common functionality like saving/loading models,
    counting parameters, etc.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        
    @abstractmethod
    def forward(self, x):
        """Forward pass logic"""
        pass
    
    def save(self, path, epoch=None):
        """Save model checkpoint
        
        Args:
            path (str): Directory to save the model
            epoch (int, optional): Current epoch number for checkpoint naming
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        filename = self.__class__.__name__
        if epoch is not None:
            filename = f"{filename}_epoch_{epoch}"
        filename = f"{filename}.pth"
        
        save_path = os.path.join(path, filename)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.__class__.__name__,
            'epoch': epoch
        }, save_path)
        
        return save_path
    
    def load(self, checkpoint_path):
        """Load model from checkpoint
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            
        Returns:
            int: The epoch number of the loaded checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint.get('epoch', None)
    
    def count_parameters(self):
        """Count the number of trainable parameters
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """Print a summary of the model architecture"""
        print(f"Model: {self.__class__.__name__}")
        print(f"Trainable parameters: {self.count_parameters():,}")
        print("Architecture:")
        print(self)