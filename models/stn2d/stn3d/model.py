# models/stn3d/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class STN3D(BaseModel):
    """3D Spatial Transformer Network for CryoET data.
    
    This model implements a 3D spatial transformer module that can
    learn to apply affine transformations to 3D input volumes.
    """
    def __init__(self, 
                 in_channels=1,
                 num_classes=2,
                 use_batch_norm=True, 
                 dropout_rate=0.5):
        """Initialize the 3D STN model.
        
        Args:
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes
            use_batch_norm (bool): Whether to use batch normalization
            dropout_rate (float): Dropout rate for regularization
        """
        super(STN3D, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # 3D Main classification network
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm3d(16)
            self.bn2 = nn.BatchNorm3d(32)
            self.bn3 = nn.BatchNorm3d(64)
            
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(p=self.dropout_rate)
        
        # The final fc layer size depends on your input size
        # For a 64x64x64 input volume, after 3 max pooling operations
        # the size would be 8x8x8
        self.fc1 = nn.Linear(64 * 8 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # 3D Spatial transformer localization network
        self.loc_conv1 = nn.Conv3d(in_channels, 8, kernel_size=5)
        self.loc_conv2 = nn.Conv3d(8, 16, kernel_size=5)
        
        if self.use_batch_norm:
            self.loc_bn1 = nn.BatchNorm3d(8)
            self.loc_bn2 = nn.BatchNorm3d(16)
            
        # For a 64x64x64 input volume, after 2 max pooling operations
        # with kernel_size=5, the size would be roughly 13x13x13
        self.loc_fc1 = nn.Linear(16 * 13 * 13 * 13, 128)
        self.loc_fc2 = nn.Linear(128, 12)  # 3D affine matrix is 3x4

        # Initialize the weights/bias with identity transformation
        self.loc_fc2.weight.data.zero_()
        # Identity matrix in 3D (3x4): [1,0,0,0, 0,1,0,0, 0,0,1,0]
        self.loc_fc2.bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def spatial_transformer(self, x):
        """3D spatial transformer network forward function
        
        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Transformed tensor
        """
        # Localization network
        xs = self.loc_conv1(x)
        
        if self.use_batch_norm:
            xs = self.loc_bn1(xs)
            
        xs = F.relu(self.pool(xs))
        
        xs = self.loc_conv2(xs)
        
        if self.use_batch_norm:
            xs = self.loc_bn2(xs)
            
        xs = F.relu(self.pool(xs))
        
        # Regressor for the 3 * 4 affine matrix (3D)
        xs = xs.view(xs.size(0), -1)
        xs = F.relu(self.loc_fc1(xs))
        theta = self.loc_fc2(xs)
        
        # Reshape to batch_size * 3 * 4 for the 3D affine transformation
        theta = theta.view(-1, 3, 4)
        
        # Apply 3D affine transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        
        return x_transformed

    def forward(self, x):
        """Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Class predictions
        """
        # Transform the input
        x = self.spatial_transformer(x)
        
        # Regular 3D CNN for classification
        x = self.conv1(x)
        
        if self.use_batch_norm:
            x = self.bn1(x)
            
        x = F.relu(self.pool(x))
        
        x = self.conv2(x)
        
        if self.use_batch_norm:
            x = self.bn2(x)
            
        x = F.relu(self.pool(x))
        
        x = self.conv3(x)
        
        if self.use_batch_norm:
            x = self.bn3(x)
            
        x = F.relu(self.pool(self.dropout(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_transformation_matrices(self, x):
        """Extract the 3D transformation matrices for visualization
        
        Args:
            x (torch.Tensor): Input tensor [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Transformation matrices [B, 3, 4]
        """
        # Localization network
        xs = self.loc_conv1(x)
        
        if self.use_batch_norm:
            xs = self.loc_bn1(xs)
            
        xs = F.relu(self.pool(xs))
        
        xs = self.loc_conv2(xs)
        
        if self.use_batch_norm:
            xs = self.loc_bn2(xs)
            
        xs = F.relu(self.pool(xs))
        
        # Regressor for the 3 * 4 affine matrix (3D)
        xs = xs.view(xs.size(0), -1)
        xs = F.relu(self.loc_fc1(xs))
        theta = self.loc_fc2(xs)
        
        # Reshape to batch_size * 3 * 4 for the affine transformation
        return theta.view(-1, 3, 4)