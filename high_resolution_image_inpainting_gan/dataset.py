import os

import cv2
import numpy as np
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb
from iglovikov_helper_functions.utils.inpainting_utils import generate_stroke_mask
from torch.utils.data import Dataset

from high_resolution_image_inpainting_gan import utils


class InpaintDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        image = load_rgb(self.imglist[index])
        image = cv2.resize(image, (self.opt.imgsize, self.opt.imgsize))

        mask = generate_stroke_mask((self.opt.imgsize, self.opt.imgsize))

        # the outputs are entire image and mask, respectively
        image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1).contiguous()

        return image, mask


class ValidationSet_with_Known_Mask(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.namelist = utils.get_names(opt.baseroot)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        # image
        imgname = self.namelist[index]
        imgpath = os.path.join(self.opt.baseroot, imgname)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize))

        # mask
        maskpath = os.path.join(self.opt.maskroot, imgname)
        mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        # the outputs are entire image and mask, respectively
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        return img, mask, imgname
