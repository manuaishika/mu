from torchvision import datasets, transforms

def get_mnist(data_path='./data', batch_size=64):
    """MNIST dataloader with STN-friendly transforms"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, transform=transform),
        batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader