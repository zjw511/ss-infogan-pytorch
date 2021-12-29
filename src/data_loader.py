
import torch
import os
import sys
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from urllib import request
from torchvision.datasets.cifar import CIFAR10
from PIL import Image
class CIFAR10S(CIFAR10):
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    def __getitem__(self, index):
        
        if self.train :
              index = index % 49500
        else:
              index = index % 500 + 49500
        # print(index)
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_loader(batch_size, root, dataset):

    # data_dir = os.path.join(root, 'data')
    data_dir = '..'
    os.makedirs(data_dir, exist_ok=True)

    if dataset == 'cifar10':
        save_dir = '%s/cifar10'%data_dir
        os.makedirs(save_dir, exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                '%s/cifar10'%data_dir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Lambda(lambda x: x * 2. - 1.)]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        # dataset_unlabel = CIFAR10S(r'./cifar10', transform=transforms.Compose(
        #             [transforms.ToTensor(),
        #             transforms.Lambda(lambda x: x * 2. - 1.)]
        #         ), download=True,train=True)
        # dataloader_unlabel = DataLoader(dataset_unlabel, batch_size=batch_size, shuffle=True, num_workers=4)
        # dataset_label = CIFAR10S(r'./cifar10_S', transform=transforms.Compose(
        #             [transforms.ToTensor(),
        #             transforms.Lambda(lambda x: x * 2. - 1.)]
        #         ), download=True,train=False)
        # dataloader_label = DataLoader(dataset_label, batch_size=batch_size, shuffle=True, num_workers=4)


    elif dataset == 'dsprites':
        save_dir = '%s/1107_dsprites'%data_dir
        os.makedirs(save_dir, exist_ok=True)
        dset = DSpriteDataset(save_dir)
        dataloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=True)

    return dataloader


class DSpriteDataset(Dataset):
    """
    A PyTorch wrapper for the dSprites dataset by
    Matthey et al. 2017. The dataset provides a 2D scene
    with a sprite under different transformations:
    * color
    * shape
    * scale
    * orientation
    * x-position
    * y-position
    """

    def __init__(self, save_dir, transform=None):
        self.transform = transform
        self.file_loc = '%s/dsprites_ndarray_train.npz'%save_dir
        # url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        # try:
        #     dataset_zip = np.load(
        #         self.file_loc, encoding='bytes', allow_pickle=True)
        # except FileNotFoundError:
        #     print("Dsprite Dataset Not Found, Downloading...")
        #     request.urlretrieve(url, self.file_loc)
        #     dataset_zip = np.load(
        #         self.file_loc, encoding='bytes', allow_pickle=True)
        with np.load(self.file_loc, encoding="latin1", allow_pickle=True) as dataset:
            data = torch.tensor(dataset["imgs"])
            targets = torch.tensor(dataset["latents_classes"])
        print("Dsprites Dataset Loaded")
        self.imgs = data
        self.latents_values = targets
        # self.imgs = np.expand_dims(
        #     dataset_zip['imgs'], axis=1).astype(np.float32)
        # self.latents_values = dataset_zip['latents_values']
        # self.latents_classes = dataset_zip['latents_classes']
        # self.metadata = dataset_zip['metadata'][()]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx]
        label = self.latents_values[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample.unsqueeze(0).float(), label
