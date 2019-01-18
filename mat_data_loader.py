from torchvision import datasets, transforms
import torch
import torch.utils.data as data_utils
from scipy.io import loadmat
import numpy as np
import os

def load_dataset(root_path, ds_name, batch_size):
    file = "dense_" + ds_name + "_decaf7.mat"
    # load raw data from .mat
    dataset_raw = loadmat(os.path.join(root_path, file))

    # train, validation and test data
    x = dataset_raw['fts'].astype('float32')
    y = dataset_raw['label'].astype('int64').squeeze() - 1   #

    train = data_utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    return train_loader

def load_mdataset(root_path, ds_name):
    file = "dense_" + ds_name + "_decaf7.mat"
    # load raw data from .mat
    dataset_raw = loadmat(os.path.join(root_path, file))

    # train, validation and test data
    x = dataset_raw['fts'].astype('float32')
    y = dataset_raw['label'].astype('int64').squeeze() - 1   #
    return (x, y)

##shared：1-10; source_unknown: 11-25; target_unknown: 26-40
def dataset_partition(x, y, source_or_target, ds_name):
    num_classes = np.max(y) + 1  #39+1=40
    print('num_classes=', num_classes)
    samples_per_class = list(np.zeros(num_classes))

    xp = np.zeros((0,4096),dtype=np.float32)
    yp = np.zeros(0, dtype=np.int64)

    for cls in range(num_classes):
        cls_inds_all = (y == cls).nonzero()[0]
        if (source_or_target == 1): #source
            cls_inds=cls_inds_all[0:50]
        else:
            if (ds_name == 'sun'):
                cls_inds=cls_inds_all[0:20]
            else:
                cls_inds=cls_inds_all[0:30]

        samples_per_class[cls] = cls_inds.size
        if (cls<10): #shared
            xp = np.r_[xp, x[cls_inds,:]]
            yp = np.r_[yp, y[cls_inds]]
        elif (cls<25): #source_unknown_class
            if (source_or_target == 1): #(source=1)
                print('cls=',cls)
                # xp = np.r_[xp, x[cls_inds, :]]
                # yc = np.zeros(samples_per_class[cls], dtype=np.int64) + 10
                # yp = np.r_[yp, yc]
        else: #target_unknown_class
            if (source_or_target == 0): #(target=0)
                xp = np.r_[xp, x[cls_inds, :]]
                yc = np.zeros(samples_per_class[cls], dtype=np.int64) + 10
                yp = np.r_[yp, yc]

    print('xp-yp-shape:')
    print(xp.shape)
    print(yp.shape)
    print('samples_per_class:')
    print(samples_per_class)

    return (xp, yp)


def load_training(x, y, batch_size):
    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    return train_loader


def load_and_partition(root_path, ds_name, source_or_target, batch_size):
    x, y = load_mdataset(root_path, ds_name)
    x, y = dataset_partition(x, y, source_or_target, ds_name)
    train_loader = load_training(x, y, batch_size)

    return train_loader


