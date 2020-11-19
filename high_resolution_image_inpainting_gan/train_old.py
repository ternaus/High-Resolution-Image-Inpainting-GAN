import argparse
import datetime
import os
import time
from pathlib import Path

import torch
import yaml
from addict import Dict as Adict
from albumentations.core.serialization import from_dict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from high_resolution_image_inpainting_gan import dataset, utils

train_image_path = Path(os.environ["TRAIN_IMAGE_PATH"])
val_image_path = Path(os.environ["VAL_IMAGE_PATH"])


def WGAN_trainer(config):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    save_folder = Path(config.experiment_name)
    sample_folder = Path(config.sample_path)

    save_folder.mkdrir(exist_ok=True, parents=True)
    sample_folder.mkdrir(exist_ok=True, parents=True)

    # Build networks
    generator = utils.create_generator(config)
    discriminator = utils.create_discriminator(config)
    perceptual_net = utils.create_perceptualnet()

    generator = generator.cuda()
    discriminator = discriminator.cuda()
    perceptual_net = perceptual_net.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()  # reduce=False, size_average=False)
    RELU = nn.ReLU()

    # Optimizers
    optimizer_g1 = torch.optim.Adam(generator.coarse.parameters(), lr=config.lr_g)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # Save the model if pre_train == True
    def save_model(net, epoch, opt, batch=0, is_D=False):
        """Save the model at "checkpoint_interval" and its multiple"""
        if is_D:
            model_name = "discriminator_WGAN_epoch%d_batch%d.pth" % (epoch + 1, batch)
        else:
            model_name = "deepfillv2_WGAN_epoch%d_batch%d.pth" % (epoch + 1, batch)
        model_name = os.path.join(save_folder, model_name)

        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), model_name)
            print("The trained model is successfully saved at epoch %d batch %d" % (epoch, batch))

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(config, transform=from_dict(config.train_aug))
    print(f"The overall number of images equals to {len(trainset)}")

    # Define the dataloader
    dataloader = DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )

    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(config.epochs):
        print("Start epoch ", epoch + 1, "!")
        for batch_idx, (img, mask) in enumerate(tqdm(dataloader)):

            # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W])
            # and put it to cuda
            img = img.cuda()
            mask = mask.cuda()

            # Generator output
            first_out, second_out = generator(img, mask)

            # forward propagation
            first_out_wholeimg = img * (1 - mask) + first_out * mask  # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask  # in range [0, 1]

            optimizer_d.zero_grad()
            fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
            true_scalar = discriminator(img, mask)
            # W_Loss = -torch.mean(true_scalar) + torch.mean(fake_scalar)#+
            # gradient_penalty(discriminator, img, second_out_wholeimg, mask)

            loss_D = torch.mean(RELU(1 - true_scalar)) + torch.mean(RELU(fake_scalar + 1))
            loss_D.backward(retain_graph=True)
            optimizer_d.step()

            # Train Generator
            # Mask L1 Loss
            first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
            second_MaskL1Loss = L1Loss(second_out_wholeimg, img)
            # GAN Loss
            fake_scalar = discriminator(second_out_wholeimg, mask)
            GAN_Loss = -torch.mean(fake_scalar)

            optimizer_g1.zero_grad()
            first_MaskL1Loss.backward(retain_graph=True)
            optimizer_g1.step()

            optimizer_g.zero_grad()

            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptual_net(img)  # feature maps
            second_out_wholeimg_featuremaps = perceptual_net(second_out_wholeimg)
            second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

            loss = (
                0.5 * config.lambda_l1 * first_MaskL1Loss
                + config.lambda_l1 * second_MaskL1Loss
                + GAN_Loss
                + second_PerceptualLoss * config.lambda_perceptual
            )
            loss.backward()

            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = config.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]"
                % (
                    (epoch + 1),
                    config.epochs,
                    (batch_idx + 1),
                    len(dataloader),
                    first_MaskL1Loss.item(),
                    second_MaskL1Loss.item(),
                )
            )
            print(
                "\r[D Loss: %.5f] [Perceptual Loss: %.5f] [G Loss: %.5f] time_left: %s"
                % (loss_D.item(), second_PerceptualLoss.item(), GAN_Loss.item(), time_left)
            )

            if (batch_idx + 1) % 5000 == 0:
                save_model(generator, epoch, config, batch_idx + 1)
                save_model(discriminator, epoch, config, batch_idx + 1, is_D=True)

        # Learning rate decrease
        adjust_learning_rate(config.lr_g, optimizer_g, (epoch + 1), config)
        adjust_learning_rate(config.lr_d, optimizer_d, (epoch + 1), config)

        # Save the model
        save_model(generator, epoch, config)
        save_model(discriminator, epoch, config, is_D=True)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg(
        "-c",
        "--config_path",
        type=Path,
        help="Path to config",
    )

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config_path) as f:
        config = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    WGAN_trainer(config)
