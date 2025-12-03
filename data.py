import torch 
from torch.distributions.constraint_registry import transform_to
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(batch_size=64, test_batch_size = 1000, num_workers = 2): #pytorch data loaders
    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST( #download training dataset 
        root = "./data",
        train= True,
        download= True,
        transform=transform 
    )

    test_dataset = datasets.MNIST( 
        root=".data",
        train = True,
        download = True, 
        transform = transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers = num_workers,
        pin_memory = True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader