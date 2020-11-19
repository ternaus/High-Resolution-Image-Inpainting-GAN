import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

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
        sn=False,
    ):
        super().__init__()

        self.pad_type = replicate_dict[pad_type](padding)
        self.norm = bn_dict[norm](out_channels)
        self.activation = activation_dict[activation]

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
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


class TransposeConv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="lrelu",
        norm="none",
        sn=False,
        scale_factor=2,
    ):
        super().__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(
            in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv2d(x)
        return x


class depth_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depth_conv(x)
        return self.point_conv(out)


class SingleChannelConv(nn.Module):
    """
    Normal conv but with output to the 1 channel.
    """

    def __init__(self, in_ch: int, kernel_size: int, stride: int, padding: int, dilation: int) -> None:
        super().__init__()
        self.single_channel_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.single_channel_conv(x)


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

        self.pad_type = replicate_dict[pad_type](padding)
        self.norm = bn_dict[norm](out_channels)
        self.activation = activation_dict[activation]

        # Initialize the convolution layers
        if single_channel_conv:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = SingleChannelConv(in_channels, kernel_size, stride, padding=0, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = depth_separable_conv(
                in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
            )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_in):
        x = self.pad(x_in)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        if self.norm:
            conv = self.norm(conv)
        if self.activation:
            conv = self.activation(conv)
        gated_mask = self.sigmoid(mask)
        return conv * gated_mask


class TransposeGatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        pad_type: str = "zero",
        activation: str = "lrelu",
        norm: str = "none",
        single_channel_conv: bool = False,
        scale_factor: int = 2,
    ):
        super().__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            pad_type,
            activation,
            norm,
            single_channel_conv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.gated_conv2d(x)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
