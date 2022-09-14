import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import h5py
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def getDataLoaders():
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_train_x.h5', 'r')
    trainset_x = f['x']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_train_y.h5', 'r')
    trainset_y = f['y']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_test_x.h5', 'r')
    testset_x = f['x']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_test_y.h5', 'r')
    testset_y = f['y']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_valid_x.h5', 'r')
    validset_x = f['x']
    f = h5py.File('pcamv1/camelyonpatch_level_2_split_valid_y.h5', 'r')
    validset_y = f['y']
    train_dataloader = torch.utils.data.DataLoader([[trainset_x[i], trainset_y[i]] for i in range(len(trainset_y))],
                                                   shuffle=True, batch_size=100)
    test_dataloader = torch.utils.data.DataLoader([[testset_x[i], testset_y[i]] for i in range(len(testset_y))],
                                                   shuffle=True, batch_size=100)
    valid_dataloader = torch.utils.data.DataLoader([[validset_x[i], validset_y[i]] for i in range(len(validset_y))],
                                                   shuffle=True, batch_size=100)
    return train_dataloader, test_dataloader, valid_dataloader


def test_print(train_dataloader):
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


if __name__ == '__main__':
    train_dataloader, test_dataloader, valid_dataloader = getDataLoaders()
    test_print(train_dataloader)



