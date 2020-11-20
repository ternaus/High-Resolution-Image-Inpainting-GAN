import torch
from torch import nn

from high_resolution_image_inpainting_gan.inpainting_network import (
    VGG16FeatureExtractor,
)


class Hinge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, y_true: torch.Tensor, y_fake: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.relu(1 - y_true)) + torch.mean(self.relu(y_fake + 1))


class Perceptual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = VGG16FeatureExtractor()
        self.l1 = nn.L1Loss()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        feat_pred = self.extractor(y_pred)
        feat_gt = self.extractor(y_true)

        return sum([self.l1(feat_pred[i], feat_gt[i]) for i in range(3)])
