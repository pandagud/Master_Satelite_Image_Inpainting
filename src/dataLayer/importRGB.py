import pathlib
from pathlib import Path
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class SatelliteDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.from_numpy(data).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)
class importData():
    def __init__(self):
        self.localdir = pathlib.Path().absolute().parent
        self.processed_path = Path.joinpath(self.localdir, 'data\\processed')



    def getRBGDataLoader(self,config):

        raw_rgb =self.get_images_array(invert=True)
        localtransform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # Skal eller skal ikke normalize?
        ])
        raw_rgbData = SatelliteDataset(raw_rgb,localtransform)
        test_data_loader = torch.utils.data.DataLoader(raw_rgbData, batch_size=config.batch_size,
                                                       shuffle=False, num_workers=config.workers)
        ## implemented to look into all for folders located in processed
        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        for folder in subfolders:
            dataroot = folder+"\\bandTCIRGB"
            # Create the dataset
            localtransform = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_data = dset.ImageFolder(dataroot+"\\Train", transform = localtransform)
            test_data= dset.ImageFolder(dataroot+"\\Test", transform = localtransform)
            train_data_loader=torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,
                                           shuffle=False, num_workers=config.workers)
            test_data_loader= torch.utils.data.DataLoader(test_data, batch_size=config.batch_size,
                                           shuffle=False, num_workers=config.workers)
        # Create the dataloader
        return train_data_loader,test_data_loader
    def open_files(self, path):
        import glob
        path_images = Path.joinpath(self.processed_path, path)
        str_path = str(path_images)
        filelist = glob.glob(str_path+'/*.tiff')
        x = np.array([np.array(Image.open(fname)) for fname in filelist])
        return x

    def get_images_array(self, invert=False, include_nir=False):
        red_images = self.open_files('T32UPV_20190904T102021\\bandTCIRGB\\Train\\redBand')
        blue_images = self.open_files('T32UPV_20190904T102021\\bandTCIRGB\\Train\\blueBand')
        greenBand = self.open_files('T32UPV_20190904T102021\\bandTCIRGB\\Train\\greenBand')
        if include_nir:
            nirBand = self.open_files('T32UPV_20190904T102021\\bandTCIRGB\\Train\\nirBand')
        test = np.stack([np.array(Image.open(Path.joinpath(self.processed_path,'T32UPV_20190904T102021\\bandTCIRGB\\Train\\redBand\\_01_07.tiff'))),
                            np.array(Image.open(Path.joinpath(self.processed_path,'T32UPV_20190904T102021\\bandTCIRGB\\Train\\blueBand\\_01_05.tiff'))),
                            np.array(Image.open(Path.joinpath(self.processed_path,'T32UPV_20190904T102021\\bandTCIRGB\\Train\\greenBand\\_01_04.tiff')))],axis=2)
        raw_rgb = np.stack([red_images,
                            blue_images,
                            greenBand],axis=2)
        #normalize the data into 0–1 scale
        raw_rgb =(raw_rgb / np.iinfo(raw_rgb.dtype).max)
        # Format is: Samples, Channels, Width, Height
        raw_rgb = np.swapaxes(raw_rgb, 1, 2)
        raw_rgb = np.swapaxes(raw_rgb,2,3)

        return raw_rgb


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
