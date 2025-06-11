import torch
import torch.nn as nn
import torch.nn.functional as F


def same_padding_conv(in_channels, out_channels, kernel_size, stride=1, groups=1):
    kh, kw = kernel_size
    ph, pw = kh // 2, kw // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(ph, pw), groups=groups)


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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwconv = same_padding_conv(in_channels, in_channels, kernel_size=(7, 3), stride=1, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)
        self.act = nn.GELU()
        self.grn = GRN(4 * in_channels)
        self.pwconv2 = nn.Linear(4 * in_channels, out_channels)
        self.cbam = CBAM(out_channels)
        # self.gate = MultiGate(out_channels)

        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

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

        # x = self.cbam(x)

        # shortcut = self.proj(shortcut)  # match to out_channels
        return self.act(shortcut + x)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class ConvNeXtCBAMClassifier(nn.Module):
    def __init__(self, in_channels=4, class_num=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4),
            nn.GELU()
        )

        self.stage1 = nn.Sequential(*[ConvNeXtBlock(64, 64) for _ in range(4)])
        self.down1 = DownsampleLayer(64, 128)
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(128, 128) for _ in range(4)])
        self.down2 = DownsampleLayer(128, 256)
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(256, 256) for _ in range(4)])
        self.down3 = DownsampleLayer(256, 512)
        self.stage4 = nn.Sequential(*[ConvNeXtBlock(512, 512) for _ in range(4)])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


if __name__ == "__main__":
    model = ConvNeXtCBAMClassifier(in_channels=4, class_num=2)
    print(model)
