import torch
from torch import nn

activation_dict = {
    "relu": nn.ReLU(inplace=True),
    "elu": nn.ELU(alpha=1.0, inplace=True),
    "lrelu": nn.LeakyReLU(0.2, inplace=True),
    "prelu": nn.PReLU(),
    "selu": nn.SELU(inplace=True),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "none": None,
}

bn_dict = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d, "none": lambda x: None}

replicate_dict = {"reflect": nn.ReflectionPad2d, "replicate": nn.ReplicationPad2d, "zero": nn.ZeroPad2d}


class Conv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="replicate",
        activation="none",
        norm="none",
        spectral_norm=False,
    ):
        super().__init__()

        self.pad = replicate_dict[pad_type](padding)
        self.norm = bn_dict[norm](out_channels)
        self.activation = activation_dict[activation]

        # Initialize the convolution layers
        if spectral_norm:
            self.conv2d = torch.nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            )
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class DepthWiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int
    ) -> None:
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depth_conv(x)
        return self.point_conv(out)


class GatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        pad_type: str = "replicate",
        activation: str = "elu",
        norm: str = "none",
        single_channel_conv: bool = False,
    ) -> None:
        super().__init__()

        self.pad = replicate_dict[pad_type](padding)
        self.norm = bn_dict[norm](out_channels)
        self.activation = activation_dict[activation]

        # Initialize the convolution layers
        if single_channel_conv:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, 1, kernel_size, stride, padding=0, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = DepthWiseSeparableConv(
                in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
            )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        if self.norm:
            conv = self.norm(conv)
        if self.activation:
            conv = self.activation(conv)
        gated_mask = self.sigmoid(mask)
        return conv * gated_mask
