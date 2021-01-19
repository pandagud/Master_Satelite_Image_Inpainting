import torch
import torch.nn as nn
from torchvision import models
#https://github.com/tanimutomo/partialconv/blob/master/src/loss.py
#https://github.com/naoto0804/pytorch-inpainting-with-partial-conv
def gram_matrix(feature_matrix):
    (batch, channel, h, w) = feature_matrix.size()
    feature_matrix = feature_matrix.view(batch, channel, h * w)
    feature_matrix_t = feature_matrix.transpose(1, 2)

    # batch matrix multiplication * normalization factor K_n
    # (batch, channel, h * w) x (batch, h * w, channel) ==> (batch, channel, channel)
    gram = torch.bmm(feature_matrix, feature_matrix_t) / (channel * h * w)

    # size = (batch, channel, channel)
    return gram


def perceptual_loss(h_comp, h_out, h_gt, l1):
    loss = 0.0

    for i in range(len(h_comp)):
        loss += l1(h_out[i], h_gt[i])
        loss += l1(h_comp[i], h_gt[i])

    return loss


def style_loss(h_comp, h_out, h_gt, l1):
    loss = 0.0

    for i in range(len(h_comp)):
        loss += l1(gram_matrix(h_out[i]), gram_matrix(h_gt[i]))
        loss += l1(gram_matrix(h_comp[i]), gram_matrix(h_gt[i]))

    return loss


# computes TV loss over entire composed image since gradient will not be passed backward to input
#def total_variation_loss(image, l1):
#    # shift one pixel and get loss1 difference (for both x and y direction)
#    loss = l1(image[:, :, :, :-1], image[:, :, :, 1:]) + l1(image[:, :, :-1, :], image[:, :, 1:, :])
#    return loss

 # def total_variation_loss(image):
 #     # shift one pixel and get difference (for both x and y direction)
 #    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
 #    return loss

# def total_variation_loss(image):
#     # shift one pixel and get difference (for both x and y direction)
#     loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
#            torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
#     return loss
def dialation_holes(hole_mask):
    b, ch, h, w = hole_mask.shape
    dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(hole_mask)
    torch.nn.init.constant_(dilation_conv.weight, 1.0)
    with torch.no_grad():
        output_mask = dilation_conv(hole_mask)
    updated_holes = output_mask != 0
    return updated_holes.float()

def total_variation_loss(image, mask):
    hole_mask = 1 - mask
    dilated_holes = dialation_holes(hole_mask)
    colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
    rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
    loss = torch.mean(torch.abs(colomns_in_Pset*(image[:, :, :, 1:] - image[:, :, :, :-1]))) + torch.mean(torch.abs(rows_in_Pset*(image[:, :, :1, :] - image[:, :, -1:, :])))
    return loss

class VGG16ExtractorNIR(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        weight = vgg16.features[0].weight.clone()
        vgg16.features[0] = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=3, bias=False)
        with torch.no_grad():
            vgg16.features[0].weight[:, :3] = weight
            vgg16.features[0].weight[:, 3] = vgg16.features[0].weight[:, 0]
        self.max_pooling1 = vgg16.features[:5]
        self.max_pooling2 = vgg16.features[5:10]
        self.max_pooling3 = vgg16.features[10:17]

        for i in range(1, 4):
            for param in getattr(self, 'max_pooling{:d}'.format(i)).parameters():
                param.requires_grad = False

    # feature extractor at each of the first three pooling layers
    def forward(self, image):
        results = [image]
        for i in range(1, 4):
            func = getattr(self, 'max_pooling{:d}'.format(i))
            results.append(func(results[-1]))
        return results[1:]

class VGG16Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.max_pooling1 = vgg16.features[:5]
        self.max_pooling2 = vgg16.features[5:10]
        self.max_pooling3 = vgg16.features[10:17]

        for i in range(1, 4):
            for param in getattr(self, 'max_pooling{:d}'.format(i)).parameters():
                param.requires_grad = False

    # feature extractor at each of the first three pooling layers
    def forward(self, image):
        results = [image]
        for i in range(1, 4):
            func = getattr(self, 'max_pooling{:d}'.format(i))
            results.append(func(results[-1]))
        return results[1:]



class CalculateLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.nir_data:
            self.vgg_extract = VGG16ExtractorNIR()
        else:
            self.vgg_extract = VGG16Extractor()
        self.l1 = nn.L1Loss()
        self.config = config

    def forward(self, input_x, mask, output, ground_truth):
        composed_output = (input_x * mask) + (output * (1 - mask))

        fs_composed_output = self.vgg_extract(composed_output)
        fs_output = self.vgg_extract(output)
        fs_ground_truth = self.vgg_extract(ground_truth)

        loss_dict = dict()

        loss_dict["hole"] = self.l1((1 - mask) * output, (1 - mask) * ground_truth) * self.config.lambdaHole
        loss_dict["valid"] = self.l1(mask * output, mask * ground_truth) * self.config.lambdaValid
        loss_dict["perceptual"] = perceptual_loss(fs_composed_output, fs_output, fs_ground_truth, self.l1) * self.config.lambdaPerceptual
        loss_dict["style"] = style_loss(fs_composed_output, fs_output, fs_ground_truth, self.l1) * self.config.lambdaStyle
        loss_dict["tv"] = total_variation_loss(composed_output,mask) * self.config.lambdaTv

        return loss_dict