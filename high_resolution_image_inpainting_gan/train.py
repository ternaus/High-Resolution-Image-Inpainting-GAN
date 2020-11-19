# """
# To run this template just do:
# python generative_adversarial_net.py
#
# After a few epochs, launch TensorBoard to see the images being generated at every batch:
#
# tensorboard --logdir default
# """
# import os
# from argparse import ArgumentParser, Namespace
#
# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
#
# from pytorch_lightning.core import LightningModule, LightningDataModule
# from pytorch_lightning.trainer import Trainer
# from high_resolution_image_inpainting_gan import dataset, utils
# from pathlib import Path
#
#
# from high_resolution_image_inpainting_gan.inpainting_network import (
#     GatedGenerator,
#     # PatchDiscriminator,
#     # PerceptualNet,
# )
#
# train_image_path = Path(os.environ["TRAIN_IMAGE_PATH"])
# val_image_path = Path(os.environ["VAL_IMAGE_PATH"])
#
#
# # class Generator(nn.Module):
# #     def __init__(self, latent_dim, img_shape):
# #         super().__init__()
# #         self.img_shape = img_shape
# #
# #         def block(in_feat, out_feat, normalize=True):
# #             layers = [nn.Linear(in_feat, out_feat)]
# #             if normalize:
# #                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
# #             layers.append(nn.LeakyReLU(0.2, inplace=True))
# #             return layers
# #
# #         self.model = nn.Sequential(
# #             *block(latent_dim, 128, normalize=False),
# #             *block(128, 256),
# #             *block(256, 512),
# #             *block(512, 1024),
# #             nn.Linear(1024, int(np.prod(img_shape))),
# #             nn.Tanh()
# #         )
# #
# #     def forward(self, z):
# #         img = self.model(z)
# #         img = img.view(img.size(0), *self.img_shape)
# #         return img
# #
# #
# # class Discriminator(nn.Module):
# #     def __init__(self, img_shape):
# #         super().__init__()
# #
# #         self.model = nn.Sequential(
# #             nn.Linear(int(np.prod(img_shape)), 512),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             nn.Linear(512, 256),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             nn.Linear(256, 1),
# #         )
# #
# #     def forward(self, img):
# #         img_flat = img.view(img.size(0), -1)
# #         validity = self.model(img_flat)
# #
# #         return validity
#
#
# class Inpainting(LightningModule):
#     def __init__(self, config: Namespace):
#         super().__init__()
#
#         self.config = config
#
#         # networks
#         # mnist_shape = (1, 28, 28)
#         # self.generator = Generator(latent_dim=self.config.latent_dim, img_shape=mnist_shape)
#         # self.discriminator = Discriminator(img_shape=mnist_shape)
#
#         # self.generator = utils.create_generator(config)
#
#         self.generator = GatedGenerator(config)
#
#         self.discriminator = utils.create_discriminator(config)
#         self.perceptual_net = utils.create_perceptualnet()
#
#         self.L1Loss = nn.L1Loss()  # reduce=False, size_average=False)
#         self.RELU = nn.ReLU()
#
#         # self.validation_z = torch.randn(8, self.config.latent_dim)
#
#         # self.example_input_array = torch.zeros(2, self.config.latent_dim)
#
#     def forward(self, z):
#         return self.generator(z)
#
#     # @staticmethod
#     # def adversarial_loss(y_hat, y):
#     #     return F.binary_cross_entropy_with_logits(y_hat, y)
#
#     def training_step(self, batch, batch_idx, optimizer_idx):
#         image = batch["image"]
#         mask = batch["mask"]
#
#         first_out, second_out = self.generator(image, mask)
#
#         # forward propagation
#         first_out_wholeimg = image * (1 - mask) + first_out * mask  # in range [0, 1]
#         second_out_wholeimg = image * (1 - mask) + second_out * mask  # in range [0, 1]
#
#         # # train generator
#         # if optimizer_idx == 0:
#         #
#         #     fake_scalar = self.discriminator(second_out_wholeimg.detach(), mask)
#         #     true_scalar = self.discriminator(image, mask)
#         #
#         #     loss_discriminator = torch.mean(self.RELU(1 - true_scalar)) + torch.mean(self.RELU(fake_scalar + 1))
#         #
#         #     self.log("loss_discriminator", loss_discriminator, on_step=True, logger=True, prog_bar=True)
#         #     return loss_discriminator
#         #
#         # # train discriminator
#         # if optimizer_idx == 1:
#         #     # Train Generator
#         #     # Mask L1 Loss
#         #     first_MaskL1Loss = self.L1Loss(first_out_wholeimg, image)
#         #     second_MaskL1Loss = self.L1Loss(second_out_wholeimg, image)
#         #
#         #     # GAN Loss
#         #     fake_scalar = self.discriminator(second_out_wholeimg, mask)
#         #     GAN_Loss = -torch.mean(fake_scalar)
#         #
#         #     # optimizer_g1.zero_grad()
#         #     # first_MaskL1Loss.backward(retain_graph=True)
#         #     # optimizer_g1.step()
#         #
#         #     optimizer_g.zero_grad()
#         #
#         #     # Get the deep semantic feature maps, and compute Perceptual Loss
#         #     img_featuremaps = perceptual_net(img)  # feature maps
#         #     second_out_wholeimg_featuremaps = perceptual_net(second_out_wholeimg)
#         #     second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)
#         #
#         #     loss = (
#         #             0.5 * self.config.lambda_l1 * first_MaskL1Loss
#         #             + self.config.lambda_l1 * second_MaskL1Loss
#         #             + GAN_Loss
#         #             + second_PerceptualLoss * self.config.lambda_perceptual
#         #     )
#         #     loss.backward()
#         #
#         #     optimizer_g.step()
#             #
#             #
#             # # Measure discriminator's ability to classify real from generated samples
#             #
#             # # how well can it label as real?
#             #
#             #
#             #
#             # valid = torch.ones(imgs.size(0), 1)
#             # valid = valid.type_as(imgs)
#             #
#             # real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
#             #
#             # # how well can it label as fake?
#             # fake = torch.zeros(imgs.size(0), 1)
#             # fake = fake.type_as(imgs)
#             #
#             # fake_loss = self.adversarial_loss(
#             #     self.discriminator(self(z).detach()), fake)
#             #
#             # # discriminator loss is the average of these
#             # d_loss = (real_loss + fake_loss) / 2
#             # tqdm_dict = {'d_loss': d_loss}
#             # self.log_dict(tqdm_dict)
#
#             return d_loss
#
#     def configure_optimizers(self):
#         optimizer_g1 = torch.optim.Adam(self.generator.coarse.parameters(), lr=self.config.lr_g)
#         optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.config.lr_g)
#         optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.lr_d)
#
#         return [optimizer_d, optimizer_g1, optimizer_g], []
#
#     # def on_epoch_end(self):
#     #     z = self.validation_z.type_as(self.generator.model[0].weight)
#     #
#     #     # log sampled images
#     #     sample_imgs = self(z)
#     #     grid = torchvision.utils.make_grid(sample_imgs)
#     #     self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
#
#     def setup(self, stage=None):
#         self.train_images = sorted(train_image_path.rglob("*.jpg"))
#
#
#     def train_dataloader(self):
#         return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
#
#
# def main(args: Namespace) -> None:
#     # ------------------------
#     # 1 INIT LIGHTNING MODEL
#     # ------------------------
#     model = Inpainting(args)
#
#     # ------------------------
#     # 2 INIT TRAINER
#     # ------------------------
#     # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
#     # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
#     dm = MNISTDataModule.from_argparse_args(args)
#     trainer = Trainer.from_argparse_args(args)
#
#     # ------------------------
#     # 3 START TRAINING
#     # ------------------------
#     trainer.fit(model, dm)
#
#
# if __name__ == '__main__':
#     parser = ArgumentParser()
#
#     # Add program level args, if any.
#     # ------------------------
#     # Add LightningDataLoader args
#     parser = MNISTDataModule.add_argparse_args(parser)
#     # Add model specific args
#     parser = Inpainting.add_argparse_args(parser)
#     # Add trainer args
#     parser = Trainer.add_argparse_args(parser)
#     # Parse all arguments
#     args = parser.parse_args()
#
#     main(args)
