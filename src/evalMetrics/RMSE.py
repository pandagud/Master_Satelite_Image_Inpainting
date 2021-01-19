import numpy as np
from src.evalMetrics.eval_helper import convert_tensor_to_nparray,remove_outliers_eval
from skimage.metrics import normalized_root_mse
class RMSE:

    def __init__(self):
        self.name = "RMSE"

    @staticmethod
    def __call__(img1, img2,tensor=True):
        if tensor:
            img1 = convert_tensor_to_nparray(img1)
            img2= convert_tensor_to_nparray(img2)
        img1 = remove_outliers_eval(img1)
        img2 = remove_outliers_eval(img2)
        return normalized_root_mse(img1,img2)
