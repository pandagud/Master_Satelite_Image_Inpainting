import random
from pathlib import Path
from typing import Dict

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

        # Define paths, maybe here make the path to the root only, and define a loop with loading the tiles
        # That go through all folders in each BIOME for to create the dataset
        OpticImage = Path(data_root_path) / 'Optic' / 'S1B_20200404_020990_DSC_139_RGB.tif'
        SarImage = Path(data_root_path) / 'Sar' / 'S1B_20200404_020990_DSC_139_RGB.tif'

        # Get the height and width of the Optic image
        with rasterio.open(OpticImage) as src:
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
                    'OpticImage_path': OpticImage,
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
        elif self.split.lower() == 'val':
            self.image_tiles = self.image_tiles[int(split_ratio * num_tiles):num_tiles]
        else:
            raise ValueError("Split must be either 'train' or 'val'")

    def __len__(self):
        return len(self.image_tiles)

    def __getitem__(self, index):
        image_tile = self.image_tiles[index]
        window_argv = [image_tile['tile_x_coordinate'],
                       image_tile['tile_y_coordinate'],
                       image_tile['tile_width'],
                       image_tile['tile_height']]

        with rasterio.open(image_tile['OpticImage_path']) as src:
            r, g, b = (src.read(k, window=Window(*window_argv)) for k in (1, 2, 3))
        image = np.array([r, g, b])
        #image = np.rollaxis(image, 0, 3)  # Change format from CHW to HWC

        with rasterio.open(
                image_tile['SARImage_path']) as src:  # Changed, to contain the channels of the pre processed SAR
            VV, VH, VVVH = (src.read(k, window=Window(*window_argv)) for k in (1, 2, 3))
            SARImage = np.array([VV, VH, VVVH])

        # Create a dict with the inputs to the transforms (in this case it is only the image)
        input_dict = {'image': image, 'SAR': SARImage}
        # transformed_input_dict = perform_transforms(input_dict, transforms_dict=self.transforms_dict, split=self.split)
        # image = transformed_input_dict['image']
        # label = transformed_input_dict['mask']

        return image, SARImage


def get_dataset(data_root_path, patch_size=256, batch_size=16, transforms_dict=None, use_cuda=False, random_seed=42):
    # Put together the dataset dict
    kwargs = {'pin_memory': False} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        LoadRasterioWindows(data_root_path,
                            patch_size=patch_size,
                            split='train',
                            transforms_dict=transforms_dict,
                            random_seed=random_seed),
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True,  # Necessary sometimes (e.g., when batchnorm layer requires more than a single sample)
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
    test_loader = None

    num_images_in_biomes = 1
    dataset_type = 'segmentation'
    dataset: Dict = {
        'train_dataloader': train_loader,
        'val_dataloader': validation_loader,
        'test_dataloader': test_loader,
        'num_images_in_biomes': num_images_in_biomes,
        'dataset_type': dataset_type
    }
    return dataset
