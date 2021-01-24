import numpy as np
import torch
def convert_tensor_to_nparray(tensor):
    tensor = tensor.detach().cpu().numpy()*255
    return np.transpose(tensor, (1, 2, 0)).astype(np.uint8)

def remove_outliers(img_array):
    percent_range = [1, 99]
    highestValue_img1 = np.percentile(img_array, percent_range)
    img1 = np.clip(img_array, highestValue_img1.min(), highestValue_img1.max())
    return img1

def remove_outliers_eval(img_array):
    percent_range = [1, 99]
    highestValue_img1 = np.percentile(img_array, percent_range)
    img1 = np.clip(img_array, highestValue_img1.min(), highestValue_img1.max())
    return img1.astype(np.uint8)