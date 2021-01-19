import pathlib
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A
import glob
from src.shared.convert import convertToFloat32
from src.shared.convert import _normalize
from src.evalMetrics.eval_helper import remove_outliers


class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list,transform=None):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        image = cv2.imread(self.image_list[i],-1)
        image = convertToFloat32(image)
        image = remove_outliers(image)
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        image = torch.from_numpy(np.array(image).astype(np.float32)).transpose(0, 1).transpose(0, 2).contiguous()
        return image

class NIRImageDataset(Dataset):
    def __init__(self, image_list, NIR_list, transform=None):
        self.image_list = image_list
        self.NIR_list = NIR_list
        self.transform = transform

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        image = cv2.imread(self.image_list[i],-1)
        r,g,b = cv2.split(image)
        NIR = cv2.imread(self.NIR_list[i],-1)
        combined_image= np.stack((r,g,b,NIR), axis=2)
        image = convertToFloat32(combined_image)
        image = remove_outliers(image)
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        image = torch.from_numpy(np.array(image).astype(np.float32)).transpose(0, 1).transpose(0, 2).contiguous()
        return image

class importData():
    def __init__(self,config):
        if config.run_polyaxon:
            self.localdir=config.data_path
        else:
            self.localdir = pathlib.Path().absolute().parent
        self.processed_path = self.localdir /'data' /'processed'
        self.config = config
        self.images = []
        self.names = []

    def getNIRDataLoader(self):
        train_transform = A.Compose([
            A.transforms.HorizontalFlip(p=0.5),
            A.transforms.VerticalFlip(p=0.5),
        ])
        test_trainsform = A.Compose([])
        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        list_train_data = []
        list_train_data_NIR = []
        list_test_data = []
        list_test_data_NIR = []
        for folder in subfolders:
            country = os.path.basename(folder)
            if country == '.ipynb_checkpoints':
                continue
            folder = [f.path for f in os.scandir(folder) if f.is_dir()]
            for dataDir in folder:
                if dataDir.find('ipynb_checkpoints') != -1:
                    continue
                dicName = dataDir
            dataroot = Path(os.path.join(dicName, "bandTCIRGB"))  ##Select first since it is not temporal
            # Create the dataset
            print("adding " + str(dataroot) + " to the dataloader")
            train_data_root = Path.joinpath(dataroot, "Train", "RGBImages", "original_0RGB", "*.tiff")
            train_data_root_NIR= Path.joinpath(dataroot, "Train", "NIRImages", "original_0NIR", "*.tiff")
            test_data_root = Path.joinpath(dataroot, "Test", "RGBImages", "original_0RGB", "*.tiff")
            test_data_root_NIR = Path.joinpath(dataroot, "Test", "NIRImages", "original_0NIR", "*.tiff")
            train_data_list = glob.glob(str(train_data_root))
            train_data_list_NIR=glob.glob(str(train_data_root_NIR))
            test_data_list = glob.glob(str(test_data_root))
            test_data_list_NIR=glob.glob(str(test_data_root_NIR))
            list_train_data = list_train_data + train_data_list
            list_train_data_NIR=list_train_data_NIR+train_data_list_NIR
            list_test_data = list_test_data + test_data_list
            list_test_data_NIR=list_test_data_NIR+test_data_list_NIR
        list_train_data = sorted(list_train_data, key=lambda i: (os.path.basename(i)))
        list_train_data_NIR = sorted(list_train_data_NIR, key=lambda i: (os.path.basename(i)))
        list_test_data = sorted(list_test_data, key=lambda i: (os.path.basename(i)))
        list_test_data_NIR = sorted(list_test_data_NIR, key=lambda i: (os.path.basename(i)))
        train_data = NIRImageDataset(image_list=list_train_data,NIR_list=list_train_data_NIR, transform=train_transform)
        test_data = NIRImageDataset(image_list=list_test_data,NIR_list=list_test_data_NIR, transform=test_trainsform)
        train_data_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=self.config.batch_size,
                                                        shuffle=True, num_workers=self.config.workers,
                                                        drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(test_data,
                                                       batch_size=self.config.batch_size,
                                                       shuffle=False, num_workers=self.config.workers,
                                                       drop_last=True)
        # Create the dataloade
        # Test data is not up!
        print("total train datasize = " + str(len(train_data_loader.dataset)))
        print("total test datasize = " + str(len(test_data_loader.dataset)))

        return train_data_loader, test_data_loader

    # def getGeoRGBDataLoader(self,countries):
    #     train_transform = A.Compose([
    #         A.transforms.HorizontalFlip(p=0.5),
    #         A.transforms.VerticalFlip(p=0.5)
    #     ])
    #     test_trainsform=A.Compose([])
    #     subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
    #     list_train_data=[]
    #     list_test_data=[]
    #     for folder in subfolders:
    #         country = os.path.basename(folder)
    #         if country == '.ipynb_checkpoints':
    #             continue
    #         folder = [f.path for f in os.scandir(folder) if f.is_dir()]
    #         for dataDir in folder:
    #             if dataDir.find('ipynb_checkpoints') != -1:
    #                 continue
    #             dicName=dataDir
    #         dataroot = Path(os.path.join(dicName,"bandTCIRGB")) ##Select first since it is not temporal
    #             # Create the dataset
    #         print("adding "+str(dataroot)+" to the dataloader")
    #         train_data_root = Path.joinpath(dataroot, "Train", "RGBImages", "original_0RGB", "*.tiff")
    #         test_data_root = Path.joinpath(dataroot, "Test", "RGBImages", "original_0RGB", "*.tiff")
    #         train_data_list = glob.glob(str(train_data_root))
    #         test_data_list = glob.glob(str(test_data_root))
    #         total_data_list = train_data_list + test_data_list
    #         total_data_list = AlbumentationImageDataset(image_list=total_data_list, transform=train_transform)
    #         if any(country in s for s in countries):
    #             list_train_data.append(total_data_list)
    #         else :
    #             list_test_data.append(total_data_list)
    #
    #     train_data_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(list_train_data),
    #     batch_size=self.config.batch_size,
    #     shuffle=True, num_workers=self.config.workers,
    #     drop_last=True)
    #     test_data_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(list_test_data),
    #     batch_size=self.config.batch_size,
    #     shuffle=False, num_workers=self.config.workers,
    #     drop_last=True)
    #     # Create the dataloade
    #     # Test data is not up!
    #     print("total train datasize = "+str(len(train_data_loader.dataset)))
    #     print("total test datasize = " + str(len(test_data_loader.dataset)))
    #
    #     return train_data_loader,test_data_loader

    def getTestDataLoader(self,path,testCountries=None,both=None):
        test_trainsform=A.Compose([])
        self.processed_path = path
        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        list_test_data=[]
        for folder in subfolders:
            country = os.path.basename(folder)
            if country == '.ipynb_checkpoints':
                continue
            folder = [f.path for f in os.scandir(folder) if f.is_dir()]
            for dataDir in folder:
                if dataDir.find('ipynb_checkpoints') != -1:
                    continue
                dicName=dataDir
            dataroot = Path(os.path.join(dicName,"bandTCIRGB")) ##Select first since it is not temporal
                # Create the dataset
            print("adding "+str(dataroot)+" to the dataloader")
            test_data_root= Path.joinpath(dataroot,"Test","RGBImages","original_0RGB","*.tiff")
            train_data_rool=Path.joinpath(dataroot,"Train","RGBImages","original_0RGB","*.tiff")
            test_data_list = glob.glob(str(test_data_root))
            train_data_list = glob.glob(str(train_data_rool))
            if both:
                test_data_list = test_data_list + train_data_list
            if testCountries==None:
                list_test_data = list_test_data + test_data_list
            if testCountries and any(country in s for s in testCountries):
                list_test_data = list_test_data + test_data_list
        list_test_data= sorted(list_test_data, key=lambda i: (os.path.basename(i)))
        test_data = AlbumentationImageDataset(image_list=list_test_data,transform=test_trainsform)
        test_data_loader = torch.utils.data.DataLoader(test_data,
        batch_size=self.config.batch_size,
        shuffle=False, num_workers=self.config.workers,
        drop_last=True)
        # Create the dataloade
        # Test data is not up!
        print("total test datasize = " + str(len(test_data_loader.dataset)))

        return test_data_loader

    def getRGBDataLoader(self,trainCountries=None,testCountries=None,randomTest=None):
        if self.config.data_normalize:
            train_transform = A.Compose([
                A.transforms.HorizontalFlip(p=0.5),
                A.transforms.VerticalFlip(p=0.5),
                A.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        else:
            train_transform = A.Compose([
                A.transforms.HorizontalFlip(p=0.5),
                A.transforms.VerticalFlip(p=0.5),
                #A.transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2)
            ])
        test_trainsform=A.Compose([])
        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        list_train_data=[]
        list_test_data=[]
        for folder in subfolders:
            country = os.path.basename(folder)
            if country == '.ipynb_checkpoints':
                continue
            folder = [f.path for f in os.scandir(folder) if f.is_dir()]
            for dataDir in folder:
                if dataDir.find('ipynb_checkpoints') != -1:
                    continue
                dicName=dataDir
            dataroot = Path(os.path.join(dicName,"bandTCIRGB")) ##Select first since it is not temporal
                # Create the dataset
            print("adding "+str(dataroot)+" to the dataloader")
            train_data_root= Path.joinpath(dataroot,"Train","RGBImages","original_0RGB","*.tiff")
            test_data_root= Path.joinpath(dataroot,"Test","RGBImages","original_0RGB","*.tiff")
            train_data_list = glob.glob(str(train_data_root))
            test_data_list = glob.glob(str(test_data_root))
            #train_data = AlbumentationImageDataset(image_list=train_data_list,transform=train_transform,data_normalize=self.config.data_normalize)
            #test_data = AlbumentationImageDataset(image_list=test_data_list,transform=test_trainsform)
            if trainCountries==None:
                list_train_data = list_train_data + train_data_list
            if testCountries==None:
                list_test_data = list_test_data + test_data_list
            if trainCountries and any(country in s for s in trainCountries):
                list_train_data = list_train_data + train_data_list
            if testCountries and any(country in s for s in testCountries):
                list_test_data = list_test_data + test_data_list
        list_train_data = sorted(list_train_data, key=lambda i: (os.path.basename(i)))
        list_test_data= sorted(list_test_data, key=lambda i: (os.path.basename(i)))
        train_data = AlbumentationImageDataset(image_list=list_train_data, transform=train_transform)
        test_data = AlbumentationImageDataset(image_list=list_test_data,transform=test_trainsform)
        train_data_loader = torch.utils.data.DataLoader(train_data,
        batch_size=self.config.batch_size,
        shuffle=True, num_workers=self.config.workers,
        drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(test_data,
        batch_size=self.config.batch_size,
        shuffle=True, num_workers=self.config.workers,
        drop_last=True)
        if randomTest:
            test_data_loader = torch.utils.data.DataLoader(test_data,
                                                           batch_size=self.config.batch_size,
                                                           shuffle=True, num_workers=self.config.workers,
                                                           drop_last=True)
        # Create the dataloade
        # Test data is not up!
        print("total train datasize = "+str(len(train_data_loader.dataset)))
        print("total test datasize = " + str(len(test_data_loader.dataset)))

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
            image = cv2.imread(fname,-1)
            data.append(image)
        return data

    def open_bandImage_as_array_and_names(self, path):
        path_images = Path.joinpath(self.processed_path, path)
        str_path = str(path_images)
        filelist = glob.glob(str_path + '/*.png')
        data = []
        names = []
        for fname in filelist:
            head,tail = os.path.split(fname)
            image = cv2.imread(fname,-1)
            names.append(tail)
            data.append(image)
        return data, names
    def get_images_for_baseLine_train(self,):

        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        for folder in subfolders:
            country = os.path.basename(folder)
            if country == '.ipynb_checkpoints':
                continue
            folder = [f.path for f in os.scandir(folder) if f.is_dir()]
            for dataDir in folder:
                if dataDir.find('ipynb_checkpoints') != -1:
                    continue
                dicName = dataDir
            dataroot = Path(os.path.join(dicName, "bandTCIRGB"))  ##Select first since it is not temporal
            dataroot = dataroot / 'Train'
            if(self.config.run_TCI):
                dataroot = dataroot+"\\TCI\\TCI"
                images = self.open_Imagefiles_as_array(dataroot)
                for i in range(len(images)):
                    self.images.append((images[i]))
            else:
                rgbroot = dataroot /'RGBImages'/'original_0RGB'
                rgb_images, names = self.open_bandImage_as_array_and_names(rgbroot,country)
                totalLen = range(len(rgb_images))
                for i in totalLen:
                    self.images.append(rgb_images[i])
                    self.names.append(names[i])
        return self.images,self.names

    def get_images_for_baseLine(self,):

        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        for folder in subfolders:
            country = os.path.basename(folder)
            if country == '.ipynb_checkpoints':
                continue
            folder = [f.path for f in os.scandir(folder) if f.is_dir()]
            for dataDir in folder:
                if dataDir.find('ipynb_checkpoints') != -1:
                    continue
                dicName = dataDir
            dataroot = Path(os.path.join(dicName, "bandTCIRGB"))  ##Select first since it is not temporal
            dataroot = dataroot / 'Test'
            if(self.config.run_TCI):
                dataroot = dataroot+"\\TCI\\TCI"
                images = self.open_Imagefiles_as_array(dataroot)
                for i in range(len(images)):
                    self.images.append((images[i]))
            else:
                rgbroot = dataroot /'RGBImages'/'original_0RGB'
                rgb_images, names = self.open_bandImage_as_array_and_names(rgbroot)
                totalLen = range(len(rgb_images))
                for i in totalLen:
                    self.images.append(rgb_images[i])
                    self.names.append(names[i])
        return self.images,self.names

    def get_images_for_baseLine_NIR(self, ):

        subfolders = [f.path for f in os.scandir(self.processed_path) if f.is_dir()]
        return_list = []
        return_names = []
        for folder in subfolders:
            country = os.path.basename(folder)
            if country == '.ipynb_checkpoints':
                continue
            folder = [f.path for f in os.scandir(folder) if f.is_dir()]
            for dataDir in folder:
                if dataDir.find('ipynb_checkpoints') != -1:
                    continue
                dicName = dataDir
            dataroot = Path(os.path.join(dicName, "bandTCIRGB"))  ##Select first since it is not temporal
            dataroot = dataroot / 'Test'
            if (self.config.run_TCI):
                dataroot = dataroot + "\\TCI\\TCI"
                images = self.open_Imagefiles_as_array(dataroot)
                for i in range(len(images)):
                    self.images.append((images[i]))
            else:
                rgbroot = dataroot / 'NIRImages' / 'original_0NIR'
                rgb_images, names = self.open_bandImage_as_array_and_names(rgbroot)
                totalLen = range(len(rgb_images))
                for i in totalLen:
                    return_list.append(rgb_images[i])
                    return_names.append(names[i])
        return return_list, return_names




    def getGeneratedImagesDataloader(self,pathToGeneratedImages):
        generated_trainsform = A.Compose([])
        generated_data_root = Path.joinpath(pathToGeneratedImages, "*.tiff")
        print("This is the path to the generated images "+ str(generated_data_root))
        generated_data_root_data_list = glob.glob(str(generated_data_root))
        print("added "+ str(len(generated_data_root_data_list)) +" to the generatedloader")
        generated_data_root_data_list = sorted(generated_data_root_data_list, key=lambda i: (os.path.basename(i)))
        generated_data = AlbumentationImageDataset(image_list=generated_data_root_data_list, transform=generated_trainsform)
        generated_data_loader = torch.utils.data.DataLoader(generated_data,
                                                        batch_size=self.config.batch_size,
                                                        shuffle=False, num_workers=self.config.workers,
                                                        drop_last=True)
        print("total generated datasize = " + str(len(generated_data_loader.dataset)))
        # Create the dataloader
        return generated_data_loader