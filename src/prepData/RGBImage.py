from PIL import Image
import pathlib
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
import random

import image_slicer

class prepRBGdata:
    Image.MAX_IMAGE_PIXELS = None
    def __init__(self,config):
        self.localdir = pathlib.Path().absolute().parent
        print(self.localdir)
        self.data_path = Path.joinpath(self.localdir, 'data\\raw')
        self.iterim_path = Path.joinpath(self.localdir,'data\\interim')
        self.processed_path = Path.joinpath(self.localdir,'data\\processed')
        self.number_tiles = config.number_tiles

    def initTestAndValSplit(self,tuples):
        tiles_list = list(tuples)
        index_list = range(tiles_list.__len__())
        random.shuffle(index_list)
        train_index = index_list[:int((len(index_list) + 1) * .80)]  # Remaining 80% to training set
        test_index = index_list[int((len(index_list) + 1) * .80):]  # Splits 20% data to test set
        train_split,index = tiles_list[:int((len(tiles_list) + 1) * .80)]  # Remaining 80% to training set
        test_split = tiles_list[int((len(tiles_list) + 1) * .80):]  # Splits 20% data to test set
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


    def convertandSlice(self,path_org_img,id):

        bandTCI_img = Image.open(path_org_img + '_TCI_10m.jp2')  # TCI
        bandTCI_img_path = Path.joinpath(self.iterim_path, id + 'bandTCI.tiff')
        bandTCI_img.save(bandTCI_img_path)

        path = self.createRGBdir(id,'bandTCI')

        tiles =self.sliceData(bandTCI_img_path)
        ## Only cal once!
        self.initTestAndValSplit(tiles)
        train, test = self.trainTestSplit(tiles)
        self.storeTrainAndTest(path,test,train,'TCI')

        band2_img  = Image.open(path_org_img+ '_B02_10m.jp2') # blue
        band2_img_path =Path.joinpath(self.iterim_path,id+'band2.tiff')
        band2_img.save(band2_img_path)

        tiles = self.sliceData(band2_img_path)
        test, train = self.trainTestSplit(tiles)
        self.storeTrainAndTest(path, test, train, 'blueBand')
        #
        band3_img = Image.open(path_org_img + '_B03_10m.jp2')  # green
        band3_img_path = Path.joinpath(self.iterim_path, id + 'band3.tiff')
        band3_img.save(band3_img_path)

        tiles = self.sliceData(band3_img_path)
        test, train = self.trainTestSplit(tiles)
        self.storeTrainAndTest(path, test, train, 'greenBand')

        band4_img = Image.open(path_org_img + '_B04_10m.jp2')  # red
        band4_img_path = Path.joinpath(self.iterim_path, id + 'band4.tiff')
        band4_img.save(band4_img_path)

        tiles = self.sliceData(band4_img_path)
        test, train = self.trainTestSplit(tiles)
        self.storeTrainAndTest(path, test, train, 'redBand')

        return bandTCI_img_path,band2_img_path,band3_img_path,band4_img_path

    def trainTestSplit(self, train,test):
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
        test_data_path = Path.joinpath(path, "Test\\Test")
        if os.path.exists(test_data_path):
            image_slicer.main.save_tiles(test_data, prefix='', directory=test_data_path, format='tiff')
        else:
            print("CreatingDir")
            os.makedirs(test_data_path)
            image_slicer.main.save_tiles(test_data, prefix='', directory=test_data_path, format='tiff')
        return

    def cleanUp(self,path_to_clean):
        for path in path_to_clean:
            if os.path.exists(path_to_clean):
                os.remove(path_to_clean)
            else:
                print("The file does not exist")


