import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_stn_results(model, test_loader, device='cpu'):
    """Visualize STN transformations"""
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        input_tensor = data.cpu()
        transformed = model.stn(data).cpu()
        
        # Plot results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_tensor[0].squeeze(), cmap='gray')
        plt.title('Original')
        
        plt.subplot(1, 2, 2)
        plt.imshow(transformed[0].squeeze(), cmap='gray')
        plt.title('Transformed')
        plt.show()