import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F

def load_mnist(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_raw = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_raw = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    X_train = train_raw.data.unsqueeze(1).float() / 255.0
    X_train = transforms.Normalize((0.1307,), (0.3081,))(X_train)
    y_train = F.one_hot(train_raw.targets, 10).float()

    X_test = test_raw.data.unsqueeze(1).float() / 255.0
    X_test = transforms.Normalize((0.1307,), (0.3081,))(X_test)
    y_test = F.one_hot(test_raw.targets, 10).float()
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader