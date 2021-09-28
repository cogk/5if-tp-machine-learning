import torch
import torch.optim as optim
import torch.nn as nn

from load_data import train_loader

import signal
import time


# class GracefulKiller:
#     kill_now = False
#     def __init__(self):
#         signal.signal(signal.SIGINT, self.exit_gracefully)
#         signal.signal(signal.SIGTERM, self.exit_gracefully)
#     def exit_gracefully(self, *args):
#         self.kill_now = True


def load_train_save(net):
    net.load_from_file()
    net_train(net, save_after_epoch=True)
    net.save_to_file()
    return net


def net_train(net, save_after_epoch=False):
    print('Starting Training')
    n_epochs = 2

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(0, n_epochs):
        if save_after_epoch:
            net.save_to_file()

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[e=%d, i=%5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return net


if __name__ == '__main__':
    from net import Net
    net = Net()
    load_train_save(net)
