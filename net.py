import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(6),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )
        self.net3 = nn.Sequential(
            nn.Linear(16 * 6 * 6, 144),
            nn.Dropout(p=0.4),
            nn.PReLU(),
            nn.Linear(144, 32),
            nn.Dropout(p=0.4),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.Dropout(p=0.4),
            nn.PReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        x = x.view(-1, 16 * 6 * 6)
        x = self.net3(x)
        return x

    def save_to_file(self, path='./saved_model.dat'):
        print('Saving model to ' + path)
        torch.save(self.state_dict(), path)

    def load_from_file(self, path='./saved_model.dat'):
        print('Loading model from ' + path)
        try:
            self.load_state_dict(torch.load(path))
            self.eval()
        except:
            print('[!] Not loading file ' + path)
