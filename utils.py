# Utility functions for training and evaluation

import torch


def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def save_checkpoint(model, optimizer, epoch, path="mnist_cnn.pth"):
   
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(state, path)
    print("Model checkpoint saved to {}".format(path))


def accuracy(output, target):
    # output: [batch_size, num_classes]
    # target: [batch_size]
    pred = output.argmax(dim=1, keepdim=False)
    correct = pred.eq(target).sum().item()
    total = target.size(0)
    return correct, total