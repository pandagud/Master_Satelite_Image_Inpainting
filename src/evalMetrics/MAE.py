import torch
import numpy as np
from src.evalMetrics.eval_helper import convert_tensor_to_nparray, remove_outliers_eval
from sklearn.metrics import mean_absolute_error
class MSE:

    def __init__(self):
        self.name = "MAE"

    @staticmethod
    def __call__(img1, img2,tensor=True):
        #The used metrics are the mean absolute error (MAE) in units of Bottom of atmosphere reflectance.
        if tensor:
            img1 = convert_tensor_to_nparray(img1)
            img2= convert_tensor_to_nparray(img2)
        img1 = img1.flatten()
        img2= img2.flatten()
        img1 = remove_outliers_eval(img1)
        img2 = remove_outliers_eval(img2)
        return mean_absolute_error(img1,img2)