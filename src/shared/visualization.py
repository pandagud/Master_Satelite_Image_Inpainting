import numpy as np
import torch
from src.shared.convert import convertToUint16
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
def normalize_array_raw(array_image):
    array_image = (array_image/10000.).astype(np.float32)
    array_image = array_image/(4000/10000)
    return array_image

def normalize_array(array_image):
    array_image = array_image/(4000/10000)
    return array_image
def normalize_batch_tensor(batch):
    return torch.div(batch,(4000/10000))

def convert_tensor_to_nparray(tensor):
    tensor = tensor.detach().cpu().numpy()
    return np.transpose(tensor, (1, 2, 0))

def convert_tensor_batch_to_store_nparray(batch,normalize=None):
    batch_array = []
    for i in batch:
        i = i.detach().cpu().numpy()
        i = np.transpose(i, (1, 2, 0))
        if normalize:
            i = normalize_array(i)
        batch_array.append(i)
    return batch_array
def safe_list_array(list_array,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n: int = len(list_array)
    f = plt.figure()
    plt.axis('off')
    plt.axis('equal')
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(list_array[i])

    plt.savefig(name)

def safe_list_array_raw(list_array,names):
    raw_list = []
    for i in list_array:
        i = convertToUint16(i)
        raw_list.append(i)
        mpimg.imsave(names[i], i)


