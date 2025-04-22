import torch.optim as optim
from models.stn import Net
from data.datasets import get_mnist
from utils.visualize import plot_stn_results

def train(epoch, model, train_loader, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_loader, test_loader = get_mnist()
    
    for epoch in range(1, 21):
        train(epoch, model, train_loader, optimizer, device)
    
    plot_stn_results(model, test_loader, device)

if __name__ == '__main__':
    main()