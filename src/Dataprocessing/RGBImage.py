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

class prepRGBdata:
    Image.MAX_IMAGE_PIXELS = None
    def __init__(self,config):
        self.localdir = pathlib.Path().absolute().parent
        print(self.localdir)
        self.data_path = Path.joinpath(self.localdir, 'data\\raw')
        self.iterim_path = Path.joinpath(self.localdir,'data\\interim')
        self.processed_path = Path.joinpath(self.localdir,'data\\processed')
        self.number_tiles = config.number_tiles
        self.image_size = config.image_size

    def initTestAndValSplit(self,tuples):
        total_tiles = len(tuples)
        index_list = list(range(total_tiles))
        random.shuffle(index_list)
        train_index = index_list[:int((len(index_list) + 1) * .80)]  # Remaining 80% to training set
        test_index = index_list[int((len(index_list) + 1) * .80):]  # Splits 20% data to test set
        return train_index, test_index

    def LoadRGBdata(self):
        ## implemented to look into all for folders located in Raw
        subfolders = [f.path for f in os.scandir(self.data_path) if f.is_dir()]
        for folder in subfolders:
            localPath = folder + r"\GRANULE"
            localImageFolder = [f.path for f in os.scandir(localPath) if f.is_dir()]
            imagefolderPath = localImageFolder[0] + r"\IMG_DATA\R10m"
            imagefiles = [f for f in listdir(imagefolderPath) if isfile(join(imagefolderPath, f))]
            unqieImageId = imagefiles[0][:-12]
            imagePath = imagefolderPath + "\\" + unqieImageId
            temp_img_list = self.convertandSlice(imagePath,unqieImageId)
            self.cleanUp(temp_img_list)


    def resizeOrginalImage(self,input_path,output_path,band=None):
        # ## Converting image to GEOtiff using Raterio. Dim is set to 10752x10752
        # Only working with TCI images
        width = 10752
        height = 10752
        with rasterio.open(input_path) as dataset:
            # resample data to target shape
            data = dataset.read(
                out_shape=(width,height
                ),
                resampling=Resampling.bilinear
            )
            profile=dataset.profile
            profile['width'] = width
            profile['height'] = height
        # writting
        with rasterio.open(output_path, 'w', **profile) as dataset:
            dataset.write(data)

    def convertandSlice(self,path_org_img,id):
        ## Creating box to end at 10752x10752 dim.
        box = (0, 0, 10752, 10752) #(x_offset, Y_offset, width, height)

        ## TCI pictures
        bandTCI_img = Image.open(path_org_img + '_TCI_10m.jp2')
        bandTCI_img_path = Path.joinpath(self.iterim_path, id + 'bandTCI.tiff')
        cropped = bandTCI_img.crop(box)
        cropped.save(str(bandTCI_img_path))
        ## self.resizeOrginalImage(bandTCI_img_path,bandTCI_img_path)

        path = self.createRGBdir(id,'bandTCI')

        tiles =self.sliceData(bandTCI_img_path)
        ## Only cal once!
        train_index, test_index = self.initTestAndValSplit(tiles)

        train, test = self.trainTestSplit(tiles,train_index,test_index)
        train, test = self.storeTrainAndTest(path,test,train,'TCI\\TCI')

        print("Done with TCI")
        ## Blue Band

        band2_img = Image.open(path_org_img + '_B02_10m.jp2') # blue
        band2_img_path = Path.joinpath(self.iterim_path, id + 'band2.tiff')
        cropped = band2_img.crop(box)
        cropped.save(str(band2_img_path))

        tiles = self.sliceData(band2_img_path)
        train, test = self.trainTestSplit(tiles,train_index,test_index)
        train, test = self.storeTrainAndTest(path, test, train, 'blueBand')

        print("Done with Blue band")
        ## Green band
        band3_img = Image.open(path_org_img + '_B03_10m.jp2')  # green
        band3_img_path = Path.joinpath(self.iterim_path, id + 'band3.tiff')
        cropped = band3_img.crop(box)
        cropped.save(str(band3_img_path))


       ## self.resizeOrginalImage(band3_img_path, band3_img_path)

        tiles = self.sliceData(band3_img_path)
        train, test= self.trainTestSplit(tiles,train_index,test_index)
        self.storeTrainAndTest(path, test, train, 'greenBand')

        print("Done with green band")
        ## Red band
        band4_img = Image.open(path_org_img + '_B04_10m.jp2')  # red
        band4_img_path = Path.joinpath(self.iterim_path, id + 'band4.tiff')
        cropped = band4_img.crop(box)
        cropped.save(str(band4_img_path))

       ## self.resizeOrginalImage(band4_img_path, band4_img_path)

        tiles = self.sliceData(band4_img_path)
        train, test= self.trainTestSplit(tiles,train_index,test_index)
        self.storeTrainAndTest(path, test, train, 'redBand')

        return bandTCI_img_path,band2_img_path,band3_img_path,band4_img_path

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

    # def resizeImages(self,path,uniqueId):
        # from tifffile import imread, imwrite
        # from skimage.transform import resize
        # resizeShape = self.image_size,self.image_size
        # count = 0
        # for filename in os.listdir(path):
        #     count += 1
        #     localPath = str(path)+'\\'+filename
        #     data = imread(localPath)
        #     resized_data = resize(data, resizeShape)
        #     imwrite(str(path)+'\\'+uniqueId+str(count)+'.tif', resized_data)

        # import cv2
        # resizeShape = self.image_size,self.image_size
        # count = 0
        # for filename in os.listdir(path):
        #     count+=1
        #     localPath = str(path)+'\\'+filename
        #     img = cv2.imread(localPath,cv2.IMREAD_UNCHANGED)
        #     resized = cv2.resize(img, resizeShape, interpolation=cv2.INTER_AREA)
        #     im = Image.fromarray(resized)
        #     im.save(str(path)+'\\'+uniqueId+str(count)+'.tif')