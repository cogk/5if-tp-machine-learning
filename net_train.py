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
    try:
        net_train(net, save_between_epochs=True)
    except:
        return net

    net.save_to_file()
    return net


def net_train(net, save_between_epochs=False):
    print('Starting Training')
    n_epochs = 2

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    delta_losses = [0]
    all_losses = []

    for epoch in range(0, n_epochs):
        print('Start of Epoch', epoch + 1, '/', n_epochs)
        epoch_loss = 0.0
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
            l = loss.item()
            running_loss += l
            epoch_loss += l

            if i % 1000 == 999:  # print every 1000 mini-batches
                print('[e=%d, i=%5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        print('End of Epoch %d: loss: %.3f' %
              (epoch + 1, epoch_loss / 2000))

        if epoch > 0:
            delta_losses.append(all_losses[epoch - 1] - running_loss)
        all_losses.append(running_loss)

        if delta_losses[-1] > 0:
            print('/!\\ loss is increasing')
            raise 'loss is increasing'

        if save_between_epochs and epoch < n_epochs - 1:
            net.save_to_file()

    print('Finished Training')
    return net


if __name__ == '__main__':
    from net import Net
    net = Net()
    load_train_save(net)
