'''

Uses Net from Model.py
get_loaders from data.py
helpers from utils.py

'''
import torch
import torch.nn.functional as F
import torch.optim as optim

from data import get_loaders
from model import Net
from utils import get_device, save_checkpoint, accuracy


def train(model, device, train_loader, optimizer, epoch, log_interval=100): # Single training epoch.
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            processed = batch_idx * len(data)
            total = len(train_loader.dataset)
            percent = 100.0 * processed / total
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, processed, total, percent, loss.item()
                )
            )

def test(model, device, test_loader): #Evaluate the model on the test dataset.
    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            test_loss = test_loss + F.nll_loss(output, target, reduction="sum").item() # Sum loss over the batch


            batch_correct, batch_total = accuracy(output, target)
            correct = correct + batch_correct
            total = total + batch_total

    test_loss = test_loss / float(total)
    acc = 100.0 * float(correct) / float(total)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, total, acc
        )
    )


def main():
    # Hyperparameters
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

    # Save final model
    if save_model:
        save_checkpoint(model, optimizer, epoch=epochs, path="mnist_cnn.pth")

if __name__ == "__main__":
    main()