from pathlib import Path
from typing import Dict, List

import albumentations as albu
import torch
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from iglovikov_helper_functions.utils.inpainting_utils import generate_stroke_mask
from torch.utils.data import Dataset


class InpaintDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: albu.Compose, length: int = None) -> None:
        self.image_paths = image_paths
        self.transform = transform

        if length is None:
            self.length = len(self.image_paths)
        else:
            self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image = load_rgb(self.image_paths[index])
        image = self.transform(image=image)["image"]

        mask = generate_stroke_mask((image.shape[1], image.shape[0]))
        return {"image": tensor_from_rgb_image(image), "mask": tensor_from_rgb_image(mask)}
