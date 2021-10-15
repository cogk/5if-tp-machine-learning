from torchsampler.imbalanced import ImbalancedDatasetSampler
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

train_dir = './train_images'
# train_dir_extra = './train_images_extra'
train_dir_lfw = './train_images_lfw-deepfunneled'
train_dir_kaggle_nat_img = './train_images_kaggle'
test_dir = './test_images'

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0,), std=(1,))])

train_data_original = torchvision.datasets.ImageFolder(
    train_dir, transform=transform)
# train_data_extra = torchvision.datasets.ImageFolder(
#     train_dir_extra, transform=transform)
train_data_lfw = torchvision.datasets.ImageFolder(
    train_dir_lfw, transform=transform)
train_data_kaggle_nat_img = torchvision.datasets.ImageFolder(
    train_dir_kaggle_nat_img, transform=transform)

train_data = torch.utils.data.ConcatDataset(
    [train_data_original, train_data_lfw, train_data_kaggle_nat_img])

test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

valid_size = 0.2
batch_size = 32

num_train = len(train_data)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

# train_sampler = SubsetRandomSampler(train_new_idx)
train_sampler = ImbalancedDatasetSampler(train_data, train_new_idx)

valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
valid_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True, num_workers=1)
classes = ('noface', 'face')
