from net import Net
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from load_data import train_loader, test_loader, test_size


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(
                f"loss: {loss:>7f}  [{current:>6d}/{size:>6d}] {100*current/size:1f}%")


def test_loop(dataloader, model, loss_fn):
    # dataloader.dataset is the full dataset, but only 10% is in the test sample
    size = len(dataloader.dataset) * test_size
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


if __name__ == '__main__':
    net = Net()
    net.load_from_file()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.007, momentum=0.9)

    epochs = 20
    prev_loss = 999
    loss_did_increase_once = False
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, net, loss_fn, optimizer)
        curr_loss = test_loop(test_loader, net, loss_fn)

        if curr_loss > prev_loss:
            print('/!\\ loss is increasing')
            net.save_to_file('./saved_model_after_first_loss_increase.dat')
            prev_loss = curr_loss

            if loss_did_increase_once:
                break
            else:
                loss_did_increase_once = True
                continue

        prev_loss = curr_loss
        net.save_to_file()
    print("Done!")
