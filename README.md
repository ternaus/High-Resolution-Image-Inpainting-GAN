# High Resolution Image Inpainting Based on GAN

Refactored implementation of [https://github.com/wangyx240/High-Resolution-Image-Inpainting-GAN](https://github.com/wangyx240/High-Resolution-Image-Inpainting-GAN)

which is an unofficial Pytorch Re-implementation of [https://arxiv.org/abs/2005.09704](Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting)" (CVPR 2020 Oral).

## Implementation
Besides Contextual Residual Aggregation(CRA) and Light-Weight GatedConvolution in the paper, also add Residual network structure, SN-PatchGAN in this project.

Dataset: Download [Places365-Standard](http://places2.csail.mit.edu/download.html) for Training and Testing.

### Training

Define path where images are stored:

```
export TRAIN_PATH=<path to train images>
```

```bash
python train.py -c <path_to_config>
```

For input size of 512x512 and GPU with memory of 11GB, recommended batchsize is 4.

### Acknowledgement & Reference

* [https://github.com/zhaoyuzhi/deepfillv2](https://github.com/zhaoyuzhi/deepfillv2)
* [https://github.com/wangyx240/High-Resolution-Image-Inpainting-GAN](https://github.com/wangyx240/High-Resolution-Image-Inpainting-GAN)
