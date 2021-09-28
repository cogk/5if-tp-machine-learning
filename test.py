import torch
from net_train import net_train, load_train_save
from load_data import test_loader
from net import Net

if __name__ == '__main__':
    net = Net()
    net.load_from_file()

    correct = 0
    total = 0

    print('Start')
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('{error:1.3f} {correct}/{total}'.format(
                correct=correct,
                total=total,
                error=correct/total
            ))

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
