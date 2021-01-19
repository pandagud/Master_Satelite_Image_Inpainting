import numpy as np

def convertToFloat32(array):
    array = array / 10000.
    return array.astype(np.float32)
def convertToUint16(array):
    array = array * 10000.
    return array.astype(np.uint16)

def _normalize_two (x,y):
    minXY = x.min()+y.min()
    maxXY =x.max()+y.max()
    avgMinMax=(minXY+maxXY)/2
    x=x/avgMinMax
    y=y/avgMinMax
    return x,y
def _normalize(M):
    """
    Normalizes M to be in range [0, 1].

    Parameters:
      M: `numpy array`
          1D, 2D or 3D data.

    Returns: `numpy array`
          Normalized data.
    """

    minVal = np.min(M)
    maxVal = np.max(M)

    Mn = M - minVal;

    if maxVal == minVal:
        return np.zeros(M.shape);
    else:
        return Mn / (maxVal-minVal).astype(np.float32)