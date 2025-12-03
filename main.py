import torch
import torch.optim as optim

from data import get_loaders
from model import Net
from utils import get_device, save_checkpoint
from train import train, test


def main():
    batch_size = 64
    test_batch_size = 1000
    epochs = 5
    learning_rate = 0.01
    momentum = 0.9
    seed = 1
    log_interval = 100
    save_model = True

    # Reproducibility
    torch.manual_seed(seed)

    # Device (CPU or GPU)
    device = get_device()
    print("Using device:", device)

    # Data loaders
    train_loader, test_loader = get_loaders(
        batch_size=batch_size,
        test_batch_size=test_batch_size
    )

    # Model and optimizer
    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)

    # Save final model checkpoint
    if save_model:
        save_checkpoint(model, optimizer, epoch=epochs, path="mnist_cnn.pth")


if __name__ == "__main__":
    main()