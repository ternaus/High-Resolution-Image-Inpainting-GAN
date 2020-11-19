from pathlib import Path
from typing import List

import albumentations as albu
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from iglovikov_helper_functions.utils.inpainting_utils import generate_stroke_mask
from torch.utils.data import Dataset


class InpaintDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: albu.Compose) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        image = load_rgb(self.imglist[index])
        image = self.transform(image=image)["image"]

        mask = generate_stroke_mask((image.shape[1], image.shape[0]))

        return {"image": tensor_from_rgb_image(image), "mask": tensor_from_rgb_image(mask)}


#
#
# class ValidationSet_with_Known_Mask(Dataset):
#     def __init__(self, opt):
#         self.opt = opt
#         self.namelist = utils.get_names(opt.baseroot)
#
#     def __len__(self):
#         return len(self.namelist)
#
#     def __getitem__(self, index):
#         # image
#         imgname = self.namelist[index]
#         imgpath = os.path.join(self.opt.baseroot, imgname)
#         img = cv2.imread(imgpath)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize))
#
#         # mask
#         maskpath = os.path.join(self.opt.maskroot, imgname)
#         mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
#         # the outputs are entire image and mask, respectively
#         mask = torch.from_numpy(mask.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
#         mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
#         return img, mask, imgname
