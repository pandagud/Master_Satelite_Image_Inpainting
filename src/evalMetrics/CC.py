
import torch
from typing import Optional, Union
import torchvision
from src.evalMetrics.eval_helper import convert_tensor_to_nparray, remove_outliers_eval
from scipy.stats.stats import pearsonr
import numpy as np
from scipy import signal

class CC:
   ## is a measure of the strength of a linear association between x,y
    def __init__(self):
        self.name = "CC"

    @staticmethod
    def __call__(img1, img2,tensor=True):
        ## Cross correlation
        ## No normalization!
        if tensor:
            img1 = convert_tensor_to_nparray(img1)
            img2 = convert_tensor_to_nparray(img2)
        img1 = remove_outliers_eval(img1)
        img2 = remove_outliers_eval(img2)
        r_val= np.corrcoef(img1.flatten(), img2.flatten())
        r_val=r_val[0, 1] ## symmetric correlation matrix where the off-diagonal element is the correlation coefficient.
        # mean_x = torch.mean(x)
        # mean_y = torch.mean(y)
        # xm = x.sub(mean_x)
        # ym = y.sub(mean_y)
        # r_num = xm.dot(ym)
        # r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        # r_val = r_num / r_den
        return r_val