from torchsampler.imbalanced import ImbalancedDatasetSampler
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0,), std=(1,))])

folders = [
    './train_images',
    # './train_images_extra',
    './train_images_lfw-deepfunneled',
    './train_images_kaggle',
    './test_images',
]
datasets = [
    torchvision.datasets.ImageFolder(folder, transform=transform)
    for folder in folders
]
all_data = torch.utils.data.ConcatDataset(datasets)

# test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

valid_size = 0.1
test_size = 0.1
train_size = 1 - valid_size - test_size
batch_size = 64

num = len(all_data)
indices = list(range(num))

np.random.seed(42)  # Don't forget to seed
np.random.shuffle(indices)

split1 = int(np.floor(valid_size * num))
split2 = int(np.floor((valid_size + test_size) * num))
valid_idx = indices[:split1]
test_idx = indices[split1:split2]
train_idx = indices[split2:]

print('Datasets:')
print('- valid:', len(valid_idx))
print('- test :', len(test_idx))
print('- train:', len(train_idx))

valid_sampler = ImbalancedDatasetSampler(all_data, valid_idx)
test_sampler = ImbalancedDatasetSampler(all_data, test_idx)
train_sampler = ImbalancedDatasetSampler(all_data, train_idx)

train_loader = torch.utils.data.DataLoader(
    all_data, batch_size=batch_size, sampler=train_idx, num_workers=2)
valid_loader = torch.utils.data.DataLoader(
    all_data, batch_size=batch_size, sampler=valid_idx, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    all_data, batch_size=batch_size, sampler=test_idx, num_workers=2)

classes = ('noface', 'face')
