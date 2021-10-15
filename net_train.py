import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from load_data import train_loader


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
    n_epochs = 10

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0007, momentum=0.9)

    last_epoch_loss = +999
    for epoch in range(0, n_epochs):
        print('Start of Epoch', epoch + 1, '/', n_epochs)
        epoch_losses = []
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

            if i % 1000 == 999:  # print every 1000 mini-batches
                print('[e=%d, i=%5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                epoch_losses.append(running_loss)
                running_loss = 0.0

        epoch_loss = np.mean(epoch_losses)
        print('End of Epoch %d: loss: %.3f' %
              (epoch + 1, epoch_loss / 1000))

        if last_epoch_loss < epoch_loss:
            print('/!\\ loss is increasing')
            net.save_to_file('./increasing_loss_saved_model.dat')
            raise 'loss is increasing'
        last_epoch_loss = epoch_loss

        if save_between_epochs and epoch < n_epochs - 1:
            net.save_to_file()

    print('Finished Training')
    return net


if __name__ == '__main__':
    from net import Net
    net = Net()
    load_train_save(net)
