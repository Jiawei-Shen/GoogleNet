import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


def same_padding_conv(in_channels, out_channels, kernel_size, stride=1, groups=1):
    kh, kw = kernel_size
    ph, pw = kh // 2, kw // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(ph, pw), groups=groups)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        gx = torch.norm(x, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)  # shape: [B, 2, H, W]
        attn = self.sigmoid(self.conv(attn))         # shape: [B, 1, H, W]
        return x * attn                              # 🔁 apply attention to original input



class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class MultiGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate1 = nn.Conv2d(channels, channels, 1)
        self.gate2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.gate1(x)) * torch.tanh(self.gate2(x))


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path):
        super().__init__()
        self.dwconv = same_padding_conv(in_channels, in_channels, kernel_size=(7, 5), stride=1, groups=in_channels)
        self.norm = LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)
        self.act = nn.GELU()
        self.grn = GRN(4 * in_channels)
        self.pwconv2 = nn.Linear(4 * in_channels, out_channels)
        self.cbam = CBAM(out_channels)
        # self.gate = MultiGate(out_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                      nn.GELU())

    def forward(self, x):
        shortcut = x  # save for shortcut
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # shape now [B, out_channels, H, W]

        x = self.cbam(x)

        shortcut = self.proj(shortcut)  # match to out_channels
        return self.act(shortcut + x)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),  # Normalize after Conv
            nn.GELU()          # Non-linearity
        )

    def forward(self, x):
        return self.downsample(x)


class ConvNeXtCBAMClassifier(nn.Module):
    """
    A ConvNeXt-style classifier with a configurable architecture.

    Args:
        in_channels (int): Number of input channels. Default: 4.
        class_num (int): Number of output classes. Default: 2.
        depths (list[int]): Number of blocks in each stage.
                           Default: [3, 3, 27, 3].
        dims (list[int]): Number of channels in each stage.
                         Default: [128, 256, 512, 1024].
    """

    def __init__(self, in_channels=4, class_num=2,
                 depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]):
        super().__init__()

        if not (len(depths) == len(dims) and len(depths) == 4):
            raise ValueError("Both depths and dims must be lists of length 4.")

        # --- Stem ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )

        # --- Stages & Downsampling ---
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        # Build the four stages
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # Build the three downsampling layers
        for i in range(3):
            downsample = DownsampleLayer(dims[i], dims[i + 1])
            self.downsample_layers.append(downsample)

        # --- Classification Head ---
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dims[-1], class_num)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Pass through stem and the four stages with downsampling in between
        x = self.stem(x)

        x = self.stages[0](x)
        x = self.downsample_layers[0](x)
        x = self.stages[1](x)
        x = self.downsample_layers[1](x)
        x = self.stages[2](x)
        x = self.downsample_layers[2](x)
        x = self.stages[3](x)

        # Global pooling and classification
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


if __name__ == "__main__":
    model = ConvNeXtCBAMClassifier(in_channels=4, class_num=2)
    print(model)
