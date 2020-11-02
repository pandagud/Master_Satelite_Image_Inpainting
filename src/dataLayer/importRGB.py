import pathlib
from pathlib import Path
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
from PIL import Image
import numpy as np



class SatelliteDataset(Dataset):
    def __init__(self, data,target, transform=None):
        self.data = torch.from_numpy(data)
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

class importData():
    def __init__(self,config):
        self.localdir = pathlib.Path().absolute().parent
        self.processed_path = Path.joinpath(self.localdir, 'data\\processed')
        self.config = config
        self.images = []




    def getRGBDataLoader(self):
        localtransform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.config.run_TCI:
            subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
            for folder in subfolders:
                folder = [f.path for f in os.scandir(folder) if f.is_dir()]
                dataroot = folder[0] + "\\bandTCIRGB"
                # Create the dataset
                train_data = dset.ImageFolder(dataroot + "\\Train\\TCI", transform=localtransform)
                test_data = dset.ImageFolder(dataroot + "\\Test\\TCI", transform=localtransform)
                train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=self.config.batch_size,
                                                                shuffle=False, num_workers=self.config.workers,
                                                                drop_last=True)
                test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=self.config.batch_size,
                                                               shuffle=False, num_workers=self.config.workers,
                                                               drop_last=True)
        else:
            subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
            list_train_data=[]
            list_test_data=[]
            for folder in subfolders:
                folder = [f.path for f in os.scandir(folder) if f.is_dir()]
                dataroot = folder[0] + "\\bandTCIRGB" ##Select first since it is not temporal
                # Create the dataset
                train_data = dset.ImageFolder(dataroot + "\\Train\\RGBImages", transform=localtransform)
                test_data = dset.ImageFolder(dataroot + "\\Test\\RGBImages", transform=localtransform)
                list_train_data.append(train_data)
                list_test_data.append(test_data)
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

    def open_Imagefiles_as_array(self,path):
        from matplotlib.image import imread
        import glob
        import cv2
        path_images = Path.joinpath(self.processed_path, path)
        str_path = str(path_images)
        filelist = glob.glob(str_path + '/*.tiff')
        data = []
        for fname in filelist:
            image = cv2.imread(fname)
            nor_image = (image / np.iinfo(image.dtype).max)
            data.append(nor_image)
        return data

    def open_bandImage_as_array(self, path):
        import glob
        import cv2
        path_images = Path.joinpath(self.processed_path, path)
        str_path = str(path_images)
        filelist = glob.glob(str_path + '/*.tiff')
        data = []
        for fname in filelist:
            image = np.array(Image.open(fname))
            # nor_image = (image / np.iinfo(image.dtype).max)
            ##nor_image = image * 255.0 / image.max()
            ##nor_image= (4095*(image - np.min(image))/np.ptp(image)).astype(int)  #Normalized [0, 4095] https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
            data.append(image)
        return data

    def get_images_for_baseLine(self,):
        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        for folder in subfolders:
            folder = [f.path for f in os.scandir(folder) if f.is_dir()]
            dataroot = folder[0]+"\\bandTCIRGB\\Train"
            if(self.config.run_TCI):
                dataroot = dataroot+"\\TCI\\TCI"
                images = self.open_Imagefiles_as_array(dataroot)
                for i in range(len(images)):
                    self.images.append((images[i]))
            else:
                rgbroot = dataroot+"\\RGBImages\\original_0RGB"
                rgb_images = self.open_bandImage_as_array(rgbroot)
                totalLen = range(len(rgb_images))
                for i in totalLen:
                    self.images.append(rgb_images[i])

        return self.images

