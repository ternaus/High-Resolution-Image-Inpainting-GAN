import argparse

from high_resolution_image_inpainting_gan.trainer import WGAN_trainer


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # General parameters
    arg("--save_path", type=str, default="./models", help="saving path that is a folder")
    arg("--sample_path", type=str, default="./samples", help="training samples path that is a folder")
    arg("--gan_type", type=str, default="WGAN", help="the type of GAN for training")
    arg("--multi_gpu", type=bool, default=False, help="nn.Parallel needs or not")
    arg("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    arg("--cudnn_benchmark", type=bool, default=True, help="True for unchanged input data type")
    arg("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    arg("--load_name_g", type=str, default="", help="load model name")
    arg("--load_name_d", type=str, default="", help="load model name")
    # Training parameters
    arg("--epochs", type=int, default=40, help="number of epochs of training")
    arg("--batch_size", type=int, default=4, help="size of the batches")
    arg("--lr_g", type=float, default=1e-4, help="Adam: learning rate")
    arg("--lr_d", type=float, default=4e-4, help="Adam: learning rate")
    arg("--weight_decay", type=float, default=0, help="Adam: weight decay")
    arg("--lr_decrease_epoch", type=int, default=10, help="lr decrease at certain epoch and its multiple")
    arg("--lr_decrease_factor", type=float, default=0.5, help="lr decrease factor, for classification default 0.1")
    arg("--lambda_l1", type=float, default=256, help="the parameter of L1Loss")
    arg("--lambda_perceptual", type=float, default=100, help="the parameter of FML1Loss (perceptual loss)")
    arg("--num_workers", type=int, default=16, help="number of cpu threads to use during batch generation")
    # Network parameters
    arg("--latent_channels", type=int, default=32, help="latent channels")
    arg("--pad_type", type=str, default="replicate", help="the padding type")
    arg("--activation", type=str, default="elu", help="the activation type")
    arg("--norm1", type=str, default="none", help="normalization type")
    arg("--norm", type=str, default="none", help="normalization type")
    arg("--init_type", type=str, default="kaiming", help="the initialization type")
    arg("--init_gain", type=float, default=0.2, help="the initialization gain")
    # Dataset parameters
    arg(
        "--baseroot",
        type=str,
        default="./dataset/data_large",
        help="the training folder: val_256, test_large, data_256",
    )
    arg("--mask_type", type=str, default="free_form", help="mask type")
    arg("--imgsize", type=int, default=512, help="size of image")
    arg("--margin", type=int, default=10, help="margin of image")
    arg("--bbox_shape", type=int, default=30, help="margin of image for bbox mask")

    return parser.parse_args()


def main():
    args = get_args()

    # """
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    # if opt.multi_gpu == True:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    # else:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = "0" """

    # Enter main function

    WGAN_trainer(args)


if __name__ == "__main__":
    main()
