import torch
from torch import nn, cuda
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


import torch
from torch import nn
# Generator Code
from torch.nn.functional import interpolate


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, kernels, normalize=True):
        super(UNetDown, self).__init__()
        if kernels == 7:
            self.conv = PartialConv2d(in_size, out_size, kernels, stride=2, padding=3, bias=False, return_mask=True, multi_channel=True)
        elif kernels == 5:
            self.conv = PartialConv2d(in_size, out_size, kernels, stride=2, padding=2, bias=False, return_mask=True, multi_channel=True)
        else:
            self.conv = PartialConv2d(in_size, out_size, kernels, stride=2, padding=1, bias=False, return_mask=True, multi_channel=True)
        if normalize:
            self.seq = nn.Sequential(nn.BatchNorm2d(out_size), nn.ReLU())
        else:
            self.seq = nn.ReLU()

    def forward(self, x, mask_in=None):
        conv, mask = self.conv(x, mask_in=mask_in)
        conv = self.seq(conv)
        return conv, mask



class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, kernels, lastLayer=False):
        super(UNetUp, self).__init__()
        self.conv = PartialConv2d(in_size, out_size, kernels, stride=1, padding=1, bias=False, multi_channel=True, return_mask=True)

        if lastLayer:
            self.seq = nn.Tanh()
        else:
            self.seq = nn.Sequential(nn.BatchNorm2d(out_size), nn.LeakyReLU(0.2))

    def forward(self, x, skip_input, mask1, mask2):
        x = interpolate(x, scale_factor=2, mode='nearest')
        out = torch.cat((x, skip_input), dim=1)
        mask = interpolate(mask1, scale_factor=2, mode='nearest')
        mask = torch.cat((mask, mask2), dim=1)

        out,mask = self.conv(out, mask_in=mask)
        out = self.seq(out)
        return out, mask



class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.down1 = UNetDown(3, 64, 3, normalize=False)  # self.down(input)??
        self.down2 = UNetDown(64, 128, 3)
        self.down3 = UNetDown(128, 256, 3)
        self.down4 = UNetDown(256, 512, 3)
        self.down5 = UNetDown(512, 512, 3)
        self.down6 = UNetDown(512, 512, 3)
        self.down7 = UNetDown(512, 512, 3)
        self.down8 = UNetDown(512, 512, 3)
        self.up1 = UNetUp(512+512, 512, 3)
        self.up2 = UNetUp(512+512, 512, 3)
        self.up3 = UNetUp(512+512, 512, 3)
        self.up4 = UNetUp(512+512, 512, 3)
        self.up5 = UNetUp(512+256, 256, 3)
        self.up6 = UNetUp(256+128, 128, 3)
        self.up7 = UNetUp(128+64, 64, 3)
        self.up8 = UNetUp(64+3, 3, 3, lastLayer=True)

    def forward(self, input, mask):
        x1, mask1 = self.down1(input, mask_in=mask)
        x2, mask2 = self.down2(x1, mask_in=mask1)
        x3, mask3 = self.down3(x2, mask_in=mask2)
        x4, mask4 = self.down4(x3, mask_in=mask3)
        x5, mask5 = self.down5(x4, mask_in=mask4)
        x6, mask6 = self.down6(x5, mask_in=mask5)
        x7, mask7 = self.down7(x6, mask_in=mask6)
        x8, mask8 = self.down8(x7, mask_in=mask7)
        x9, mask9 = self.up1(x8, x7, mask8, mask7)
        x10, mask10 = self.up2(x9, x6, mask9, mask6)
        x11, mask11 = self.up3(x10, x5, mask10, mask5)
        x12, mask12 = self.up4(x11, x4, mask11, mask4)
        x13, mask13 = self.up5(x12, x3, mask12, mask3)
        x14, mask14 = self.up6(x13, x2, mask13, mask2)
        x15, mask15 = self.up7(x14, x1, mask14, mask1)
        output, mask16 = self.up8(x15, input, mask15, mask)
        return output

   # def repeat(self, mask, size1, size2):
   #     return torch.cat(
   #         [mask[:, 0].unsqueeze(1).repeat(1, size1, 1, 1), mask[:, 1].unsqueeze(1).repeat(1, size2, 1, 1)], dim=1)

class discriminator(nn.Module):

    # discriminator model
    def __init__(self):
        super(discriminator, self).__init__()

        self.conv1 = self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
    )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
    )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
    )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
    )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv8 = nn.Sequential(

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
    )


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        return conv8