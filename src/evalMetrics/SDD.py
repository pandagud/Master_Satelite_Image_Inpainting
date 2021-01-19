import numpy as np
from src.evalMetrics.eval_helper import convert_tensor_to_nparray,remove_outliers_eval
class SDD:
#Computing the sum of squared differences (SSD) between two images.
    def __init__(self):
        self.name = "SDD"

 ## https://gist.github.com/nimpy/54ccb199c978a5074cdcd35fc696a904
    @staticmethod
    def __call__(img1, img2,tensor=True):
        """Computing the sum of squared differences (SSD) between two images."""
        if tensor:
            img1= convert_tensor_to_nparray(img1)
            img2 = convert_tensor_to_nparray(img2)
        img1 = remove_outliers_eval(img1)
        img2 = remove_outliers_eval(img2)
        return np.sum((img1 - img2) ** 2)