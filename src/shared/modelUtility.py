import torch
import os
import cv2
from pathlib import Path
from os import remove
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import glob
import shutil
from torchvision.utils import make_grid
from torchvision.utils import save_image
from src.shared.convert import convertToUint16
from src.shared.visualization import normalize_array,normalize_batch_tensor,convert_tensor_batch_to_store_nparray,safe_list_array,convert_tensor_to_nparray, normalize_batch_SAR
import numpy as np
import os
from polyaxon import tracking
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class modelHelper:


    @staticmethod
    def saveModel (name, output_path,model,folderName):
        save_model_name = name
        save_model_path = (output_path / save_model_name).with_suffix('.pt')
        if not save_model_path.parent.exists():
            save_model_path.parent.mkdir()
        onlyfiles = glob.glob(str(os.path.join(save_model_path.parent,'*.pt')))
        if len(onlyfiles) >4:
            removeFile =save_model_path.parent /onlyfiles[0]
            remove(removeFile)
        torch.save(model.state_dict(), save_model_path)
        print(f"Model has been saved to {str(save_model_path)}")
        return save_model_path
    @staticmethod
    def saveToTxt(filename, saveString):
        # Function to save to txt file.
        #filename = filename
        # Creates file if it does not exist, else does nothing
        if not filename.parent.exists():
            filename.parent.mkdir()
        filename.touch(exist_ok=True)
        # then open, write and close file again
        file = open(filename, 'a')
        file.write(str(saveString))
            #'Generator loss: ' + str(generatorLoss[0]) + '\n' + 'Generator loss BCE: ' + str(
            #generatorlossBCE[0]) + '\n' + 'Discriminator loss: ' + str(discLossBCE[0]) + '\n')
        file.close()

    @staticmethod
    def clearFolder(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Error: %s : %s" % (path, e.strerror))

    @staticmethod
    def save_tensor_batch_eval_test(image_tensorReal, image_tensorFake,
                          batchSize, path):
        '''
              Function for visualizing images: Given a tensor of images, number of images, and
              size per image, plots and prints the images in an uniform grid.
        '''
        if not path.parent.exists():
            path.parent.mkdir()

        # fake_images = convert_tensor_batch_to_store_nparray(image_tensorFake,normalize=True)
        # masked_images = convert_tensor_batch_to_store_nparray(image_tensorMasked, normalize=True)
        # count = 0
        # for i in range(len(real_images)):
        #     group_images = []
        #     group_images.append(real_images[i])
        #     group_images.append(fake_images[i])
        #     group_images.append(masked_images[i])
        #     safe_list_array(group_images, str(path) + '_normalize_'+str(count)+'.tiff')
        #     count = count+1
        # image_tensor1 = (image_tensorReal + 1) / 2
        image_unflat1 = image_tensorReal.detach().cpu()
        image_unflat1 = normalize_batch_tensor(image_unflat1)
        # image_tensor2 = (image_tensorFake + 1) / 2
        image_unflat2 = image_tensorFake.detach().cpu()
        image_unflat2 = normalize_batch_tensor(image_unflat2)
        # image_tensor3 = (image_tensorMasked + 1) / 2
        image_unflat1 = torch.cat((image_unflat1, image_unflat2), dim=0)
        image_grid = make_grid(image_unflat1[:batchSize * 2], nrow=batchSize)
        save_image(image_grid, str(path) + '.tiff')

    @staticmethod
    def save_tensor_batch_NIR(image_tensorReal, image_tensorFake, image_tensorMasked,
                          batchSize, path):
        '''
              Function for visualizing images: Given a tensor of images, number of images, and
              size per image, plots and prints the images in an uniform grid.
        '''
        if not path.parent.exists():
            path.parent.mkdir()
        # fake_images = convert_tensor_batch_to_store_nparray(image_tensorFake,normalize=True)
        # masked_images = convert_tensor_batch_to_store_nparray(image_tensorMasked, normalize=True)
        # count = 0
        # for i in range(len(real_images)):
        #     group_images = []
        #     group_images.append(real_images[i])
        #     group_images.append(fake_images[i])
        #     group_images.append(masked_images[i])
        #     safe_list_array(group_images, str(path) + '_normalize_'+str(count)+'.tiff')
        #     count = count+1
        # image_tensor1 = (image_tensorReal + 1) / 2
        image_unflat1 = image_tensorReal.detach().cpu()
        image_unflat1 = normalize_batch_tensor(image_unflat1)
        image_unflat1 = image_unflat1[:,:3, :, :]
        # image_tensor2 = (image_tensorFake + 1) / 2
        image_unflat2 = image_tensorFake.detach().cpu()
        image_unflat2 = normalize_batch_tensor(image_unflat2)
        image_unflat2 = image_unflat2[:,:3, :, :]
        # image_tensor3 = (image_tensorMasked + 1) / 2
        image_unflat3 = image_tensorMasked.detach().cpu()
        image_unflat3 = normalize_batch_tensor(image_unflat3)
        image_unflat3 = image_unflat3[:,:3, :, :]
        image_unflat1 = torch.cat((image_unflat1, image_unflat2, image_unflat3), dim=0)
        image_grid = make_grid(image_unflat1[:batchSize * 3], nrow=batchSize)
        save_image(image_grid, str(path) + '.tiff')

    @staticmethod
    def save_tensor_single_NIR(image_tensorFake,
                           path,path_nir, raw=None):
        '''
              Function for visualizing images: Given a tensor of images, number of images, and
              size per image, plots and prints the images in an uniform grid.
        '''
        if raw:
            image_unflat1 = image_tensorFake.detach().cpu()
            images = convert_tensor_to_nparray(image_unflat1)
            images = convertToUint16(images)
            r,g,b,NIR = cv2.split(images)
            images = np.stack([r,g,b], axis=2)
            cv2.imwrite(str(path), images)
            cv2.imwrite(str(path_nir),NIR)

        else:
            image_unflat1 = image_tensorFake.detach().cpu()
            image_grid = make_grid(image_unflat1, nrow=1)
            save_image(image_grid, path)

    @staticmethod
    def save_tensor_batch_TCI(image_tensorReal, image_tensorFake, image_tensorMasked,
                          batchSize, path):
        '''
              Function for visualizing images: Given a tensor of images, number of images, and
              size per image, plots and prints the images in an uniform grid.
        '''
        if not path.parent.exists():
            path.parent.mkdir()

        # fake_images = convert_tensor_batch_to_store_nparray(image_tensorFake,normalize=True)
        # masked_images = convert_tensor_batch_to_store_nparray(image_tensorMasked, normalize=True)
        # count = 0
        # for i in range(len(real_images)):
        #     group_images = []
        #     group_images.append(real_images[i])
        #     group_images.append(fake_images[i])
        #     group_images.append(masked_images[i])
        #     safe_list_array(group_images, str(path) + '_normalize_'+str(count)+'.tiff')
        #     count = count+1
        # image_tensor1 = (image_tensorReal + 1) / 2
        image_unflat1 = image_tensorReal.detach().cpu()
        # image_tensor2 = (image_tensorFake + 1) / 2
        image_unflat2 = image_tensorFake.detach().cpu()
        # image_tensor3 = (image_tensorMasked + 1) / 2
        image_unflat3 = image_tensorMasked.detach().cpu()
        image_unflat1 = torch.cat((image_unflat1, image_unflat2, image_unflat3), dim=0)
        image_grid = make_grid(image_unflat1[:batchSize * 3], nrow=batchSize)
        save_image(image_grid, str(path) + '.tiff')

    @staticmethod
    def save_tensor_batch(image_tensorReal,image_tensorFake, image_tensorMasked,
                           batchSize,path):
        '''
              Function for visualizing images: Given a tensor of images, number of images, and
              size per image, plots and prints the images in an uniform grid.
        '''
        if not path.parent.exists():
            path.parent.mkdir()

        # fake_images = convert_tensor_batch_to_store_nparray(image_tensorFake,normalize=True)
        # masked_images = convert_tensor_batch_to_store_nparray(image_tensorMasked, normalize=True)
        # count = 0
        # for i in range(len(real_images)):
        #     group_images = []
        #     group_images.append(real_images[i])
        #     group_images.append(fake_images[i])
        #     group_images.append(masked_images[i])
        #     safe_list_array(group_images, str(path) + '_normalize_'+str(count)+'.tiff')
        #     count = count+1
        #image_tensor1 = (image_tensorReal + 1) / 2
        image_unflat1 = image_tensorReal.detach().cpu()
        image_unflat1 = normalize_batch_tensor(image_unflat1)
        #image_tensor2 = (image_tensorFake + 1) / 2
        image_unflat2 = image_tensorFake.detach().cpu()
        image_unflat2 = normalize_batch_tensor(image_unflat2)
        #image_tensor3 = (image_tensorMasked + 1) / 2
        image_unflat3 = image_tensorMasked.detach().cpu()
        image_unflat3 = normalize_batch_tensor(image_unflat3)
        image_unflat1 = torch.cat((image_unflat1, image_unflat2, image_unflat3), dim=0)
        image_grid = make_grid(image_unflat1[:batchSize * 3], nrow=batchSize)
        save_image(image_grid,str(path)+'.tiff')
        plt.imshow(image_grid.permute(1, 2, 0))
        plt.show()

    @staticmethod
    def save_tensor_batchSAR(image_tensorReal, image_tensorSAR, image_tensorOpticFake,
                          batchSize, path):
        '''
              Function for visualizing images: Given a tensor of images, number of images, and
              size per image, plots and prints the images in an uniform grid.
        '''
        if not path.parent.exists():
            path.parent.mkdir()

        # fake_images = convert_tensor_batch_to_store_nparray(image_tensorFake,normalize=True)
        # masked_images = convert_tensor_batch_to_store_nparray(image_tensorMasked, normalize=True)
        # count = 0
        # for i in range(len(real_images)):
        #     group_images = []
        #     group_images.append(real_images[i])
        #     group_images.append(fake_images[i])
        #     group_images.append(masked_images[i])
        #     safe_list_array(group_images, str(path) + '_normalize_'+str(count)+'.tiff')
        #     count = count+1
        # image_tensor1 = (image_tensorReal + 1) / 2
        image_unflat1 = image_tensorReal.detach().cpu()
        image_unflat1 = normalize_batch_tensor(image_unflat1)
        # image_tensor2 = (image_tensorFake + 1) / 2
        image_unflat2 = image_tensorSAR.detach().cpu()
        #image_unflat2 = normalize_batch_SAR(image_unflat2)
        # image_tensor3 = (image_tensorMasked + 1) / 2
        image_unflat3 = image_tensorOpticFake.detach().cpu()
        image_unflat3 = normalize_batch_tensor(image_unflat3)

        image_unflat1 = torch.cat((image_unflat1, image_unflat2, image_unflat3), dim=0)
        image_grid = make_grid(image_unflat1[:batchSize * 3], nrow=batchSize)
        save_image(image_grid, str(path) + '.tiff')
        plt.imshow(image_grid.permute(1, 2, 0))
        plt.show()

    @staticmethod
    def save_tensor_single(image_tensorFake,
                     path, raw=None):
        '''
              Function for visualizing images: Given a tensor of images, number of images, and
              size per image, plots and prints the images in an uniform grid.
        '''
        if raw:
            image_unflat1 = image_tensorFake.detach().cpu()
            images = convert_tensor_to_nparray(image_unflat1)
            images = convertToUint16(images)
            cv2.imwrite(str(path), images)

        else:
            image_unflat1 = image_tensorFake.detach().cpu()
            image_grid = make_grid(image_unflat1, nrow=1)
            save_image(image_grid, path)

    @staticmethod
    def cls():
        os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def saveMetrics(metrics,name,value,polyaxon_experiment,epoch_step):
        metrics[name] = value
        polyaxon_experiment.log_metrics(step=epoch_step, **metrics)

    @staticmethod
    def saveMetricsNewPolyaxon(metrics,name,value,epoch_step):
        os.environ['POLYAXON_NO_OP'] = 'true'
        from polyaxon import tracking
        tracking.init()
        polyaxon_experiment = tracking
        metrics[name] = value
        polyaxon_experiment.log_metrics(step=epoch_step,**metrics)
