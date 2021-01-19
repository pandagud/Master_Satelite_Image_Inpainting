import numpy as np
from skimage.metrics import structural_similarity as ssim
from src.evalMetrics.eval_helper import convert_tensor_to_nparray,remove_outliers_eval
from scipy import stats
class SSIM_SKI:
#structural_similarity
    def __init__(self):
        self.name = "SSIM"
    @staticmethod
    def __call__(img1, img2,tensor=True):
        if tensor:
            img1 = convert_tensor_to_nparray(img1)
            img2 = convert_tensor_to_nparray(img2)
        img1 = remove_outliers_eval(img1)
        img2= remove_outliers_eval(img2)
        minvalue = min(img1.min(), img1.min())
        maxvalue = max(img1.max(),img1.max())
        range= maxvalue - minvalue
        s = ssim(img1, img2,
                          data_range=range, multichannel=True, gaussian_weights=True,
                          win_size=11)
        return s
