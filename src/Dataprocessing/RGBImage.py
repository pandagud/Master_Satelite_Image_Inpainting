from PIL import Image
import pathlib
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
import random

import image_slicer
import rasterio
from rasterio.enums import Resampling
import numpy as np
import shutil
import cv2
from src.shared.convert import convertToFloat32,_normalize
from src.shared.visualization import normalize_array
import matplotlib
class prepRGBdata:
    Image.MAX_IMAGE_PIXELS = None
    def __init__(self,config):
        self.localdir = pathlib.Path().absolute().parent
        self.data_path = Path.joinpath(self.localdir, 'data\\raw')
        self.iterim_path = Path.joinpath(self.localdir,'data\\interim')
        self.data_name = "Remove_cloud"
        self.processed_path = Path.joinpath(self.localdir,'data\\processed\\'+ self.data_name)
        self.pixel_values_path = Path.joinpath(self.localdir,'data\\values')
        self.number_tiles = config.number_tiles
        self.image_size = config.image_size
        self.config = config

    def initTestAndValSplit(self,tuples):
        total_tiles = len(tuples)
        index_list = list(range(total_tiles))
        random.shuffle(index_list)
        train_index = index_list[:int((len(index_list) + 1) * .80)]  # Remaining 80% to training set
        test_index = index_list[int((len(index_list) + 1) * .80):]  # Splits 20% data to test set
        return train_index, test_index
    def prepAllInRaw(self):
        subfolders = [f.path for f in os.scandir(self.data_path) if f.is_dir()]
        for folder in subfolders:
            self.data_name = os.path.basename(folder)
            self.processed_path = Path.joinpath(self.localdir, 'data\\processed\\' + self.data_name)
            self.pixel_values_path = self.pixel_values_path / self.data_name
            path = [f.path for f in os.scandir(folder) if f.is_dir()]
            self.LoadRGBdata(path[0])
    def LoadRGBdata(self,path):
        ## implemented to look into all for folders located in Raw
        count = 0
        localPath = path + r"\GRANULE"
        localImageFolder = [f.path for f in os.scandir(localPath) if f.is_dir()]
        imagefolderPath = localImageFolder[0] + r"\IMG_DATA\R10m"
        imagefiles = [f for f in listdir(imagefolderPath) if isfile(join(imagefolderPath, f))]
        unqieImageId = imagefiles[0][:-12]
        imagePath = imagefolderPath + "\\" + unqieImageId
        status = "original_" + str(count)
        temp_img_list = self.convertandSlice(imagePath, unqieImageId, status)
        self.cleanUp(temp_img_list)
        count = count + 1;
        self.firstTime = False
        self.secondTime = True
        print("Just done with number " + str(count))

    def open_bandImage_as_array(self, path):
        import glob
        path_images = Path.joinpath(self.processed_path, path)
        str_path = str(path_images)
        filelist = glob.glob(str_path + '/*.tiff')
        data = []
        for fname in filelist:
            image = cv2.imread(fname, -1)
            image[image>10000]=10000
            data.append(image)
        return data

    def storeRGBDataInTrainAndTest(self,id,status):

        folder = str(self.processed_path)+"\\"+id
        ## Train

        dataroot = folder + "\\bandTCIRGB\\Train"
        redroot = dataroot + "\\redBand"
        red_images = self.open_bandImage_as_array(redroot)
        self.storePixelParams(red_images,"redBand",self.pixel_values_path)

        blueroot = dataroot + "\\blueBand"
        blue_images = self.open_bandImage_as_array(blueroot)
        self.storePixelParams(blue_images, "blueBand", self.pixel_values_path)
        greenroot = dataroot + "\\greenBand"
        green_images = self.open_bandImage_as_array(greenroot)
        self.storePixelParams(green_images, "greenBand", self.pixel_values_path)

        nirroot = dataroot + "\\nirBand"
        nir_images = self.open_bandImage_as_array(nirroot)
        self.storePixelParams(nir_images, "nirBand", self.pixel_values_path)

        RGBimages = []
        for i in range(len(red_images)):
            raw_rgb = np.stack([red_images[i],
                                green_images[i],
                                blue_images[i]
                                ], axis=2)
            RGBimages.append(raw_rgb)
        RGBroot = dataroot + "\\RGBImages\\" + status + "RGB"
        nirRoot = dataroot+ "\\NIRImages\\"+status+"NIR"
        norRGBroot = dataroot + "\\normalizedRGBImages\\" + status + "RGB"
        count = 0
        if not Path.exists(Path(nirRoot)):
            os.makedirs(nirRoot)

        if not Path.exists(Path(norRGBroot)):
            os.makedirs(norRGBroot)
        if os.path.exists(RGBroot):
            for index,i in enumerate(RGBimages):
                cv2.imwrite(RGBroot + "\\" + id + "_train_" + str(count) + ".tiff", i)
                cv2.imwrite(nirRoot + "\\" + id + "_train_" + str(count) + ".tiff", nir_images[index])
                matplotlib.image.imsave(norRGBroot + "\\" + id + "_train_" + str(count) + ".tiff",_normalize(i))
                count = count + 1
        else:
            print("CreatingDir")
            os.makedirs(RGBroot)
            for index,i in enumerate(RGBimages):
                cv2.imwrite(RGBroot + "\\" + id + "_train_" + str(count) + ".tiff", i)
                cv2.imwrite(nirRoot + "\\" + id + "_train_" + str(count) + ".tiff", nir_images[index])
                matplotlib.image.imsave(norRGBroot + "\\" + id + "_train_" + str(count) + ".tiff",_normalize(i))
                count = count + 1

        bands = [redroot, blueroot, greenroot,nirroot]

        self.removeFolder(bands)
        print("Done with Train data")
        ## Test
        dataroot = folder + "\\bandTCIRGB\\Test"
        redroot = dataroot + "\\redBand"
        red_images = self.open_bandImage_as_array(redroot)
        blueroot = dataroot + "\\blueBand"
        blue_images = self.open_bandImage_as_array(blueroot)
        greenroot = dataroot + "\\greenBand"
        green_images = self.open_bandImage_as_array(greenroot)
        nirroot = dataroot + "\\nirBand"
        nir_images = self.open_bandImage_as_array(nirroot)
        self.storePixelParams(nir_images, "nirBand", self.pixel_values_path)

        totalLen = range(len(red_images))
        RGBimages = []
        for i in totalLen:
            raw_rgb = np.stack([red_images[i],
                                green_images[i],
                                blue_images[i]
                                ], axis=2)
            RGBimages.append(raw_rgb)
        RGBroot = dataroot + "\\RGBImages\\" + status + "RGB"
        nirRoot = dataroot + "\\NIRImages\\" + status + "NIR"
        norRGBroot = dataroot + "\\normalizedRGBImages\\" + status + "RGB"
        if not Path.exists(Path(nirRoot)):
            os.makedirs(nirRoot)

        if not Path.exists(Path(norRGBroot)):
            os.makedirs(norRGBroot)
        if os.path.exists(RGBroot):
            for index, i in enumerate(RGBimages):
                cv2.imwrite(RGBroot + "\\" + id + "_test_" + str(count) + ".tiff", i)
                cv2.imwrite(nirRoot + "\\" + id + "_test_" + str(count) + ".tiff", nir_images[index])
                matplotlib.image.imsave(norRGBroot + "\\" + id + "_train_" + str(count) + ".tiff",_normalize(i))
                count = count + 1
        else:
            print("CreatingDir")
            os.makedirs(RGBroot)
            for index, i in enumerate(RGBimages):
                cv2.imwrite(RGBroot + "\\" + id + "_test_" + str(count) + ".tiff", i)
                cv2.imwrite(nirRoot + "\\" + id + "_test_" + str(count) + ".tiff", nir_images[index])
                matplotlib.image.imsave(norRGBroot + "\\" + id + "_train_" + str(count) + ".tiff",_normalize(i))
                count = count + 1
        bands = [redroot, blueroot, greenroot,nirroot]
        self.removeFolder(bands)
        print("Done with Test data")

    def convertandSlice(self,path_org_img,id,status):
        ## Creating box to end at 10752x10752 dim.
        box = (0, 0, 10752, 10752) #(x_offset, Y_offset, width, height)

        ##TCI pictures

        bandTCI_img = rasterio.open(path_org_img + '_TCI_10m.jp2').read()
        bandTCI_img = np.swapaxes(bandTCI_img, 0, 1)
        bandTCI_img = np.swapaxes(bandTCI_img, 1, 2)
        bandTCI_img_path = Path.joinpath(self.iterim_path, id + 'bandTCI.tiff')
        Pil_Image = Image.fromarray(bandTCI_img, mode="I;16")
        cropped = Pil_Image.crop(box)
        cropped.save(str(bandTCI_img_path))
        ## self.resizeOrginalImage(bandTCI_img_path,bandTCI_img_path)

        path = self.createRGBdir(id,'bandTCI')

        tiles =self.sliceData(bandTCI_img_path)
        ## Only cal once!
        train_index, test_index = self.initTestAndValSplit(tiles)

        train, test = self.trainTestSplit(tiles,train_index,test_index)
        train, test = self.storeTrainAndTest(path,test,train,'TCI\\TCI')

        print("Done with TCI")
       #  ## Blue Band
        blueband = rasterio.open(path_org_img + '_B02_10m.jp2').read()
        blueband = np.swapaxes(blueband, 0, 1)
        blueband = np.swapaxes(blueband, 1, 2)
        Pil_Image = Image.fromarray(blueband,mode="I;16")
        cropped = Pil_Image.crop(box)
        band2_img_path = Path.joinpath(self.iterim_path, id + 'band2.tiff')
        cropped.save(str(band2_img_path))

        tiles = self.sliceData(band2_img_path)
        train, test = self.trainTestSplit(tiles,train_index,test_index)
        train, test = self.storeTrainAndTest(path, test, train, 'blueBand')

        print("Done with Blue band")
       #  ## Green band

        band3_img = rasterio.open(path_org_img + '_B03_10m.jp2').read()  # green
        band3_img = np.swapaxes(band3_img, 0, 1)
        band3_img = np.swapaxes(band3_img, 1, 2)
        Pil_Image = Image.fromarray(band3_img, mode="I;16")
        band3_img_path = Path.joinpath(self.iterim_path, id + 'band3.tiff')
        cropped = Pil_Image.crop(box)
        cropped.save(str(band3_img_path))

       ## self.resizeOrginalImage(band3_img_path, band3_img_path)

        tiles = self.sliceData(band3_img_path)
        train, test= self.trainTestSplit(tiles,train_index,test_index)
        self.storeTrainAndTest(path, test, train, 'greenBand')

        print("Done with green band")
       #  ## Red band
        band4_img = rasterio.open(path_org_img + '_B04_10m.jp2').read()    # red
        band4_img = np.swapaxes(band4_img, 0, 1)
        band4_img = np.swapaxes(band4_img, 1, 2)
        Pil_Image = Image.fromarray(band4_img, mode="I;16")
        band4_img_path = Path.joinpath(self.iterim_path, id + 'band4.tiff')
        cropped = Pil_Image.crop(box)
        cropped.save(str(band4_img_path))

       ## self.resizeOrginalImage(band4_img_path, band4_img_path)

        tiles = self.sliceData(band4_img_path)
        train, test= self.trainTestSplit(tiles,train_index,test_index)
        self.storeTrainAndTest(path, test, train, 'redBand')
        print("Done with red band")

        #  ## NIR band
        band8_img = rasterio.open(path_org_img + '_B08_10m.jp2').read()  # NIR
        band8_img = np.swapaxes(band8_img, 0, 1)
        band8_img = np.swapaxes(band8_img, 1, 2)
        Pil_Image = Image.fromarray(band8_img, mode="I;16")
        band8_img_path = Path.joinpath(self.iterim_path, id + 'band8.tiff')
        cropped = Pil_Image.crop(box)
        cropped.save(str(band8_img_path))

        ## self.resizeOrginalImage(band4_img_path, band4_img_path)

        tiles = self.sliceData(band8_img_path)
        train, test = self.trainTestSplit(tiles, train_index, test_index)
        self.storeTrainAndTest(path, test, train, 'nirBand')
        print("Done with nir band")


        self.storeRGBDataInTrainAndTest(id,status)

        return bandTCI_img_path ,band2_img_path,band3_img_path,band4_img_path

    def removeFolder(self,path):
        for i in path :
            try:
                shutil.rmtree(i)
            except OSError as e:
                print("Error: %s : %s" % (i, e.strerror))


    def trainTestSplit(self, tuples,train_index,test_index):
        train = []
        test = []
        for i in train_index:
            train.append(tuples[i])
        for i in test_index:
            test.append(tuples[i])

        train_data = tuple(train)
        test_data = tuple(test)
        return train_data, test_data

    def createRGBdir(self,id,bandNumber):
        localPath = Path.joinpath(self.processed_path,id)
        os.makedirs(localPath,exist_ok=True)
        path_store_img=Path.joinpath(localPath,bandNumber+'RGB')
        os.makedirs(path_store_img, exist_ok=True)
        return path_store_img

    def sliceData(self,path_org_img):

        tiles = image_slicer.slice(path_org_img, self.number_tiles, save=False)
        return tiles

    def storeTrainAndTest(self,path,test_data,train_data,folder_name):

        ## Train
        train_path=Path.joinpath(path,"Train\\"+folder_name)
        if os.path.exists(train_path):
            image_slicer.main.save_tiles(train_data, prefix='', directory=train_path, format='tiff')
        else:
            print("CreatingDir")
            os.makedirs(train_path)
            image_slicer.main.save_tiles(train_data, prefix='', directory=train_path, format='tiff')

        ##Test
        test_data_path = Path.joinpath(path, "Test\\"+folder_name)
        if os.path.exists(test_data_path):
            image_slicer.main.save_tiles(test_data, prefix='', directory=test_data_path, format='tiff')
        else:
            print("CreatingDir")
            os.makedirs(test_data_path)
            image_slicer.main.save_tiles(test_data, prefix='', directory=test_data_path, format='tiff')
        return train_path, test_data_path

    def cleanUp(self,path_to_clean):
        for path in path_to_clean:
            if os.path.exists(path):
                os.remove(path)
            else:
                print("The file does not exist")

    def storePixelParams(self,array,name,filename):
        mean = np.mean(array)
        std = np.std(array)
        max = np.max(array)
        min = np.min(array)
        # Function to save to txt file.
        # filename = filename
        saveString = 'Name : ' + str(name) + '\n' +'Mean: ' + str(
            mean) + '\n' + 'standard deviation: ' + str(
            std) + '\n' + 'Max pixel value: ' + str(
            max) + '\n' + 'Min pixel value ' + str(
            min)
        filename = Path.joinpath(filename,name+".txt")
        if not filename.parent.exists():
            filename.parent.mkdir()
        filename.touch(exist_ok=True)
        # then open, write and close file again
        file = open(filename, 'a')
        file.write(str(saveString))
        # 'Generator loss: ' + str(generatorLoss[0]) + '\n' + 'Generator loss BCE: ' + str(
         # generatorlossBCE[0]) + '\n' + 'Discriminator loss: ' + str(discLossBCE[0]) + '\n')
        file.close()