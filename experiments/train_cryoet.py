import torch
import torch.optim as optim
from models.stn3d import STN3D
from data.cryoet_loader import get_cryoet_loaders
from utils.visualize_3d import plot_2d_projections

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STN3D(input_shape=(64, 64, 64)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Load cryo-ET data (replace with your directory)
    train_loader = get_cryoet_loaders("/path/to/mrc_files", batch_size=4)
    
    for epoch in range(1, 101):
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_transformed = model(x)
            loss = F.mse_loss(x_transformed, x)  # unsupervised alignment
            loss.backward()
            optimizer.step()
        
        # visualize every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                sample = next(iter(train_loader))[0][0].cpu()  # first sample
                transformed = model(sample.unsqueeze(0).to(device)).cpu()
                plot_2d_projections(transformed.squeeze().numpy())

if __name__ == "__main__":
    train()