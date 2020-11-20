import argparse
import os
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import yaml
from addict import Dict as Adict
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

from high_resolution_image_inpainting_gan.dataset import InpaintDataset

image_path = Path(os.environ["IMAGE_PATH"])


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Inpainting(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generator = object_from_dict(self.config["generator"])

        self.losses = {"l1": nn.L1Loss()}

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        return self.generator(**batch)

    def setup(self, stage=0):  # pylint: disable=W0613
        self.image_paths = sorted(image_path.rglob("*.jpg"))
        print("Len train images = ", len(self.image_paths))

    def train_dataloader(self):
        train_aug = from_dict(self.config.train_aug)

        if "epoch_length" not in self.config.train_parameters:
            epoch_length = None
        else:
            epoch_length = self.config.train_parameters.epoch_length

        result = DataLoader(
            InpaintDataset(self.image_paths, train_aug, epoch_length),
            batch_size=self.config.train_parameters.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(result))
        return result

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.config["optimizer_generator"],
            params=[x for x in self.generator.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self.config["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):  # pylint: disable=W0613
        images = batch["image"]
        masks = batch["mask"]

        # Generator output
        first_out, second_out = self.generator(images, masks)

        # forward propagation
        first_out_whole_image = images * (1 - masks) + first_out * masks  # in range [0, 1]
        second_out_whole_image = images * (1 - masks) + second_out * masks  # in range [0, 1]

        # Mask L1 Loss
        first_mask_l1_loss = self.losses["l1"](first_out_whole_image, images)
        second_mask_l1_loss = self.losses["l1"](second_out_whole_image, images)

        total_loss = (
            self.config.loss_weights["mask_l1"] * first_mask_l1_loss
            + self.config.loss_weights["mask_l2"] * second_mask_l1_loss
        )

        self.log("first_mask_l1", first_mask_l1_loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.log("second_mask_l1", second_mask_l1_loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.log("total_loss", total_loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)

        return total_loss

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()


def main():
    args = get_args()

    with open(args.config_path) as f:
        config = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    pl.trainer.seed_everything(config.seed)
    pipeline = Inpainting(config)

    Path(config.checkpoint_callback.filepath).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        config["trainer"],
        logger=WandbLogger(config["experiment_name"]),
        checkpoint_callback=object_from_dict(config["checkpoint_callback"]),
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
