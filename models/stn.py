# models/stn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class STN(BaseModel):
    """Spatial Transformer Network implementation.
    
    This model implements a spatial transformer module that can
    learn to apply affine transformations to input images.
    """
    def __init__(self, 
                 in_channels=1,
                 num_classes=10,
                 use_batch_norm=True, 
                 dropout_rate=0.5):
        """Initialize the STN model.
        
        Args:
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes
            use_batch_norm (bool): Whether to use batch normalization
            dropout_rate (float): Dropout rate for regularization
        """
        super(STN, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Main classification network
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(10)
            self.bn2 = nn.BatchNorm2d(20)
            
        self.dropout = nn.Dropout2d(p=self.dropout_rate)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        # Spatial transformer localization network
        self.loc_conv1 = nn.Conv2d(in_channels, 8, kernel_size=7)
        self.loc_conv2 = nn.Conv2d(8, 10, kernel_size=5)
        
        if self.use_batch_norm:
            self.loc_bn1 = nn.BatchNorm2d(8)
            self.loc_bn2 = nn.BatchNorm2d(10)
            
        self.loc_fc1 = nn.Linear(10 * 3 * 3, 32)
        self.loc_fc2 = nn.Linear(32, 3 * 2)  # 3x2 affine matrix

        # Initialize the weights/bias with identity transformation
        self.loc_fc2.weight.data.zero_()
        self.loc_fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def spatial_transformer(self, x):
        """Spatial transformer network forward function
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Transformed tensor
        """
        # Localization network
        xs = self.loc_conv1(x)
        
        if self.use_batch_norm:
            xs = self.loc_bn1(xs)
            
        xs = F.relu(F.max_pool2d(xs, 2))
        
        xs = self.loc_conv2(xs)
        
        if self.use_batch_norm:
            xs = self.loc_bn2(xs)
            
        xs = F.relu(F.max_pool2d(xs, 2))
        
        # Regressor for the 3 * 2 affine matrix
        xs = xs.view(-1, 10 * 3 * 3)
        xs = F.relu(self.loc_fc1(xs))
        theta = self.loc_fc2(xs)
        
        # Reshape to batch_size * 2 * 3 for the affine transformation
        theta = theta.view(-1, 2, 3)
        
        # Apply affine transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        
        return x_transformed

    def forward(self, x):
        """Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Class predictions
        """
        # Transform the input
        x = self.spatial_transformer(x)
        
        # Regular CNN for classification
        x = self.conv1(x)
        
        if self.use_batch_norm:
            x = self.bn1(x)
            
        x = F.relu(F.max_pool2d(x, 2))
        
        x = self.conv2(x)
        
        if self.use_batch_norm:
            x = self.bn2(x)
            
        x = F.relu(F.max_pool2d(self.dropout(x), 2))
        
        # Flatten the output
        x = x.view(-1, 320)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_transformation_matrices(self, x):
        """Extract the transformation matrices for visualization
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Transformation matrices [B, 2, 3]
        """
        # Localization network
        xs = self.loc_conv1(x)
        
        if self.use_batch_norm:
            xs = self.loc_bn1(xs)
            
        xs = F.relu(F.max_pool2d(xs, 2))
        
        xs = self.loc_conv2(xs)
        
        if self.use_batch_norm:
            xs = self.loc_bn2(xs)
            
        xs = F.relu(F.max_pool2d(xs, 2))
        
        # Regressor for the 3 * 2 affine matrix
        xs = xs.view(-1, 10 * 3 * 3)
        xs = F.relu(self.loc_fc1(xs))
        theta = self.loc_fc2(xs)
        
        # Reshape to batch_size * 2 * 3 for the affine transformation
        return theta.view(-1, 2, 3)
