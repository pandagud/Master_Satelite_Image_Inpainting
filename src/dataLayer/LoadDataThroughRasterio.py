import albumentations as A
import random
from pathlib import Path
from typing import Dict
from glob import glob
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

class LoadRasterioWindows(Dataset):
    def __init__(self, data_root_path, patch_size=256, split='train', transforms_dict=None, random_seed=42):
        self.transforms_dict = transforms_dict
        self.split: str = split
        split_ratio = 0.8  # The train/val ratio
        tile_height = patch_size
        tile_width = patch_size

        #Define paths of images
        OpticImageB02 = glob(str(Path(data_root_path) / 'Sentinel2') + '/*B02.tif')[0]
        OpticImageB03 = glob(str(Path(data_root_path) / 'Sentinel2') + '/*B03.tif')[0]
        OpticImageB04 = glob(str(Path(data_root_path) / 'Sentinel2') + '/*B04.tif')[0]
        SarImage = glob(str(Path(data_root_path) / 'Sentinel1') + '/*sar.tif')[0]
        # Get the height and width of the Optic image
        with rasterio.open(OpticImageB02) as src:
            image_height = src.height
            image_width = src.width

        # Assert that the dimensions of the rgb image matches the SarImage
        with rasterio.open(SarImage) as src:
            assert image_height == src.height
            assert image_width == src.width

        # Create a dict with paths to images and labels, and with coordinates for tiles (ie. windows)
        num_row_divisions = int(image_height / tile_height)  # int() always rounds down
        num_col_divisions = int(image_width / tile_width)  # int() always rounds down
        self.image_tiles = []
        for row_num in range(num_row_divisions):
            for col_num in range(num_col_divisions):
                # Save information on each tile as a dict to provide flexibility for possible expansions of the dataset.
                # For instance, if additional images are added, or image with different resolutions, such that you want
                # the tiles to be of different sizes, such that the resolutions match (however, in this case the tiles
                # needs to be resized in the applied transformations).
                image_tile = {
                    'OpticImage_path2': OpticImageB02,
                    'OpticImage_path3': OpticImageB03,
                    'OpticImage_path4': OpticImageB04,
                    'SARImage_path': SarImage,
                    'tile_y_coordinate': row_num * tile_height,
                    'tile_x_coordinate': col_num * tile_width,
                    'tile_height': tile_height,
                    'tile_width': tile_width
                }
                self.image_tiles.append(image_tile)

        # Shuffle the tiles and split into train and val data
        random.Random(random_seed).shuffle(self.image_tiles)
        num_tiles = len(self.image_tiles)
        if self.split.lower() == 'train':
            self.image_tiles = self.image_tiles[0:int(split_ratio * num_tiles)]
        elif self.split.lower() == 'test':
            self.image_tiles = self.image_tiles[int(split_ratio * num_tiles):num_tiles]
        else:
            raise ValueError("Split must be either 'train' or 'test'")

    def __len__(self):
        return len(self.image_tiles)

    def __getitem__(self, index):
        image_tile = self.image_tiles[index]
        window_argv = [image_tile['tile_x_coordinate'],
                       image_tile['tile_y_coordinate'],
                       image_tile['tile_width'],
                       image_tile['tile_height']]

        # Optic image is in 3 separate bands
        with rasterio.open(image_tile['OpticImage_path4'], driver="GTiff", dtype=rasterio.uint16) as src:
            r = (src.read(window=Window(*window_argv)))
        with rasterio.open(image_tile['OpticImage_path3'], driver="GTiff", dtype=rasterio.uint16) as src:
            g = (src.read(window=Window(*window_argv)))
        with rasterio.open(image_tile['OpticImage_path2'], driver="GTiff", dtype=rasterio.uint16) as src:
            b = (src.read(window=Window(*window_argv)))
        image = np.vstack((r, g, b))

        # Sar image is in 3 bands
        with rasterio.open(
                image_tile['SARImage_path']) as src:  # Changed, to contain the channels of the pre processed SAR
            VV, VH, VVVH = (src.read(k, window=Window(*window_argv)) for k in (1, 2, 3))
        SARImage = np.array([VV, VH, VVVH])

        # Transform
        if self.transforms_dict is not None:
            transformed = self.transforms_dict(image=image, image0=SARImage)

        # Optic image
        image = transformed['image']
        image = torch.from_numpy(np.array((image).astype(np.float32)))
        image[image > 10000] = 10000
        image = image / 10000

        # SAR Image
        SARImage = transformed['image0']
        SARImage = torch.from_numpy(np.array((SARImage).astype(np.float32)))
        SARImage = (SARImage + 45) / 90
        # SARImage = SARImage/45

        return image, SARImage


def get_dataset(data_root_path, patch_size=256, batch_size=16, transforms_dict=None, use_cuda=False, random_seed=42):
    # Put together the dataset dict
    kwargs = {'pin_memory': False} if use_cuda else {}
    ##Test transforms
    transforms_dict = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        additional_targets={'image0': 'image', 'image1': 'image'}
    )
    transforms_dict_test = A.Compose(
        [

        ],
        additional_targets={'image0': 'image', 'image1': 'image'}
    )

    all_datasets = []
    for i in data_root_path:
        all_datasets.append(LoadRasterioWindows(i,
                                                patch_size=patch_size,
                                                split='train',
                                                transforms_dict=transforms_dict,
                                                random_seed=random_seed))

    final_dataset = torch.utils.data.ConcatDataset(all_datasets)
    train_loader = torch.utils.data.DataLoader(final_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True,
                                               # Necessary sometimes (e.g., when batchnorm layer requires more than a single sample)
                                               **kwargs
                                               )

    all_datasets_test = []
    for i in data_root_path:
        all_datasets_test.append(LoadRasterioWindows(i,
                                                patch_size=patch_size,
                                                split='test',
                                                transforms_dict=transforms_dict_test,
                                                random_seed=random_seed))
    final_dataset_test = torch.utils.data.ConcatDataset(all_datasets_test)
    test_loader = torch.utils.data.DataLoader(final_dataset_test,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True,
                                               # Necessary sometimes (e.g., when batchnorm layer requires more than a single sample)
                                               **kwargs
                                               )
    # kwargs = {'pin_memory': False} if use_cuda else {}
    # validation_loader = torch.utils.data.DataLoader(
    #    LoadRasterioWindows(data_root_path,
    #               patch_size=patch_size,
    #               split='val',
    #               transforms_dict=transforms_dict,
    #               random_seed=random_seed),
    #    batch_size=batch_size,
    #    shuffle=True,
    #    num_workers=6,
    #    **kwargs
    # )
    validation_loader = None

    dataset_type = 'Image inpainting'
    dataset: Dict = {
        'train_dataloader': train_loader,
        'val_dataloader': validation_loader,
        'test_dataloader': test_loader,
        'dataset_type': dataset_type
    }
    return dataset