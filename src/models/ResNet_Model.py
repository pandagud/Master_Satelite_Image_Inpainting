#Resnet gan
import torch
import torch.nn as nn
from torch.nn import init
#from torch.optim import lr_scheduler

class ResNetGenerator(nn.Module):

    def __init__(self, input_n_channels, output_n_channels, n_filters=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_ResNet_blocks=6,
                 padding_type='reflect'):
        super(ResNetGenerator, self).__init__()
        