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
        self.data = torch.from_numpy(data).float()
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

        if self.config.run_TCI:
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
            for folder in subfolders:
                dataroot = folder + "\\bandTCIRGB"
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
         train_data = self.get_images_for_baseLine()
         localtransform = transforms.Compose([
             transforms.ToPILImage(),
             transforms.Resize(self.config.image_size),
             transforms.CenterCrop(self.config.image_size),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])
         train_data = np.array(train_data)
         test_target= np.zeros(len(train_data))
         train_data = SatelliteDataset(train_data,test_target,localtransform)
         train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=self.config.batch_size,
                                          shuffle=False, num_workers= self.config.workers,drop_last=True)
         test_data_loader = torch.utils.data.DataLoader(train_data, batch_size=self.config.batch_size,
                                            shuffle=False, num_workers= self.config.workers,drop_last=True)

        # Create the dataloader
        return train_data_loader,test_data_loader

    # def open_files(self, path):
    #     import glob
    #     path_images = Path.joinpath(self.processed_path, path)
    #     str_path = str(path_images)
    #     filelist = glob.glob(str_path+'/*.tiff')
    #     x = np.array([np.array(Image.open(fname)) for fname in filelist])
    #     return x
    def open_bandImage_as_array(self,path):
        import glob
        import cv2
        path_images = Path.joinpath(self.processed_path, path)
        str_path = str(path_images)
        filelist = glob.glob(str_path + '/*.tiff')
        data = []
        for fname in filelist:
            image = np.array(Image.open(fname))
            nor_image = (image / np.iinfo(image.dtype).max)
            ##nor_image = image * 255.0 / image.max()
            ##nor_image= (4095*(image - np.min(image))/np.ptp(image)).astype(int)  #Normalized [0, 4095] https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
            data.append(nor_image)
        return data

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

    def get_images_for_baseLine(self,):
        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        for folder in subfolders:
            dataroot = folder+"\\bandTCIRGB\\Train"
            if(self.config.run_TCI):
                dataroot = dataroot+"\\TCI\\TCI"
                images = self.open_Imagefiles_as_array(dataroot)
                for i in range(len(images)):
                    self.images.append((images[i]))
            else:
                redroot = dataroot+"\\redBand"
                red_images = self.open_bandImage_as_array(redroot)
                blueroot = dataroot+"\\blueBand"
                blue_images = self.open_bandImage_as_array(blueroot)
                greenroot = dataroot+"\\greenBand"
                green_images = self.open_bandImage_as_array(greenroot)
                totalLen = range(len(red_images))
                for i in totalLen:
                    raw_rgb = np.stack([red_images[i],
                                    green_images[i],
                                     blue_images[i]
                                     ], axis=2)
                    self.images .append(raw_rgb)

        return self.images

    # def Load(self):
    #     ## implemented to look into all for folders located in Raw
    #     subfolders = [f.path for f in os.scandir(self.datapath) if f.is_dir()]
    #     for folder in subfolders:
    #         localPath = folder+r"\GRANULE"
    #         localImageFolder = [f.path for f in os.scandir(localPath) if f.is_dir()]
    #         imagefolderPath = localImageFolder[0]+r"\IMG_DATA\R10m"
    #         imagesPath = [f.path for f in os.scandir(imagefolderPath) if f.is_dir()]
    #         imagefiles = [f for f in listdir(imagefolderPath) if isfile(join(imagefolderPath, f))]
    #         unqieImageId = imagefiles[0][:-12]
    #         imagePath = imagefolderPath + "\\"+unqieImageId
    #
    #
    #         with rasterio.open(imagePath +'_B02_10m.jp2', driver='JP2OpenJPEG') as src:
    #             blue = src.read(1, window=Window(0, 0, 256, 256))
    #         print(blue.shape)
    #         band2 = rasterio.open(imagePath +'_B02_10m.jp2', driver='JP2OpenJPEG')  # blue
    #         band3 = rasterio.open(imagePath + '_B03_10m.jp2', driver='JP2OpenJPEG')  # green
    #         band4 = rasterio.open(imagePath + '_B04_10m.jp2', driver='JP2OpenJPEG')  # red
    #         band8 = rasterio.open(imagePath + '_B08_10m.jp2', driver='JP2OpenJPEG')  # nir
    #         # number of raster bands
    #         band4.count
    #         # number of raster columns
    #         band4.width
    #         # number of raster rows
    #         band4.height
    #         # plot band
    #         plot.show(band2)
    #         plot.show(blue)
    #         # type of raster byte
    #         band4.dtypes[0]
    #         # raster sytem of reference
    #         band4.crs
    #         # raster transform parameters
    #         band4.transform
    #         # raster values as matrix array
    #         band4.read(1)
    #         # multiple band representation
    #         plot.show(blue,cmap='Blues')
    #
    #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    #         plot.show(band2, ax=ax1, cmap='Blues')
    #         plot.show(band3, ax=ax2, cmap='Greens')
    #         plot.show(band4, ax=ax3, cmap='Reds')
    #         fig.tight_layout()
    #     print(self.localdir)

