import pathlib
from pathlib import Path
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class importDataTemporal():
    def __init__(self,config):
        self.localdir = pathlib.Path().absolute().parent
        self.processed_path = Path.joinpath(self.localdir, 'data\\processed')
        self.config = config
        self.images = []

    def getOldRGBDataLoader(self):
        localtransform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        list_train_data = []
        list_test_data = []
        for folder in subfolders:
            folder = [f.path for f in os.scandir(folder) if f.is_dir()]
            root_list_test = []
            root_list_train = []
            for temporalfolder in folder:
                dataroot = temporalfolder + "\\bandTCIRGB"  ##Select first since it is not temporal
                # Create the dataset
                train_data_root = dataroot + "\\Train\\RGBImages";
                test_data_root = dataroot + "\\Test\\RGBImages"
                root_list_train.append(train_data_root)
                root_list_test.append(test_data_root)
            local_train_dataset = ConcatDataset(dset.ImageFolder(root_list_train[0], transform=localtransform))
            local_test_dataset = ConcatDataset(dset.ImageFolder(root_list_test[0], transform=localtransform))
            list_train_data.append(local_train_dataset)
            list_test_data.append(local_test_dataset)
        train_data_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(list_train_data),
                                                        batch_size=self.config.batch_size,
                                                        shuffle=False, num_workers=self.config.workers,
                                                        drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(list_test_data),
                                                       batch_size=self.config.batch_size,
                                                       shuffle=False, num_workers=self.config.workers,
                                                       drop_last=True)

        # Create the dataloader
        return train_data_loader, test_data_loader


    def getTemporalRGBDataLoader(self):
        localtransform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        list_train_data=[]
        list_test_data=[]
        for folder in subfolders:
            folder = [f.path for f in os.scandir(folder) if f.is_dir()]
            root_list_test =[]
            root_list_train=[]
            for temporalfolder in folder:
                dataroot = temporalfolder + "\\bandTCIRGB" ##Select first since it is not temporal
                # Create the dataset
                train_data_root = dataroot + "\\Train\\RGBImages";
                test_data_root= dataroot + "\\Test\\RGBImages"
                root_list_train.append(train_data_root)
                root_list_test.append(test_data_root)
            local_train_dataset = ConcatDataset(dset.ImageFolder(root_list_train[0], transform=localtransform),dset.ImageFolder(root_list_train[1], transform=localtransform),dset.ImageFolder(root_list_train[2], transform=localtransform),dset.ImageFolder(root_list_train[3], transform=localtransform),dset.ImageFolder(root_list_train[4], transform=localtransform))
            local_test_dataset= ConcatDataset(dset.ImageFolder(root_list_test[0], transform=localtransform),dset.ImageFolder(root_list_test[1], transform=localtransform),dset.ImageFolder(root_list_test[2], transform=localtransform),dset.ImageFolder(root_list_test[3], transform=localtransform),dset.ImageFolder(root_list_test[4], transform=localtransform))
            list_train_data.append(local_train_dataset)
            list_test_data.append(local_test_dataset)
        train_data_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(list_train_data),
        batch_size=self.config.batch_size,
        shuffle=False, num_workers=self.config.workers,
        drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(list_test_data),
        batch_size=self.config.batch_size,
        shuffle=False, num_workers=self.config.workers,
        drop_last=True)

        # Create the dataloader
        return train_data_loader,test_data_loader

