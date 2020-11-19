import os

import cv2
import numpy as np
import skimage
import torch
import torchvision.models as models
from torch import nn
from torch.nn import init

from high_resolution_image_inpainting_gan.inpainting_network import (
    PatchDiscriminator,
    PerceptualNet,
)

# ----------------------------------------
#                 Network
# ----------------------------------------
# def create_generator(config):
#     # Initialize the networks
#     generator = GatedGenerator(config)
#     print("Generator is created!")
#     if config.load_name_g:
#         generator.load_state_dict(torch.load(config.load_name_g))
#         print("Load generator %s" % config.load_name_g)
#     else:
#         # Init the networks
#         weights_init(generator, init_type=config.init_type, init_gain=config.init_gain)
#         print("Initialize generator with %s type" % config.init_type)
#     return generator


def create_discriminator(opt):
    # Initialize the networks
    discriminator = PatchDiscriminator(opt)
    print("Discriminator is created!")
    if opt.load_name_d:
        discriminator.load_state_dict(torch.load(opt.load_name_d))
        print("Load generator %s" % opt.load_name_d)
    else:
        weights_init(discriminator, init_type=opt.init_type, init_gain=opt.init_gain)
        print("Initialize discriminator with %s type" % opt.init_type)
    return discriminator


def create_perceptualnet():
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = PerceptualNet()
    # Pre-trained VGG-16
    vgg16 = models.vgg16(pretrained=True)
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    print("Perceptual network is created!")
    return perceptualnet


def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net


def weights_init(net: nn.Module, init_type: str = "kaiming", init_gain: float = 0.02) -> None:
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)


# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt=255):
    # Save image one-by-one
    for i, img in enumerate(img_list):
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        # Save to certain path
        save_img_name = sample_name + "_" + name_list[i] + ".png"
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)


def psnr(pred, target, pixel_max_cnt=255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p


def grey_psnr(pred, target, pixel_max_cnt=255):
    pred = torch.sum(pred, dim=0)
    target = torch.sum(target, dim=0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p


def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel=True)
    return ssim
