import torch
import torch.nn as nn
import math

class ChannelLayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ChannelLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.conv(y.unsqueeze(1)).squeeze(1)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x


class FuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(FuseReduce, self).__init__()
        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = ChannelLayer(self.out_channels) 

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),
            SpatialAttention(kernel_size=3),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        xl_resized = nn.functional.interpolate(xl, size=xh.shape[2:], mode='bilinear', align_corners=True)
        bottomup_wei = self.bottomup(xl_resized * topdown_wei)
        xs1 = 2 * xl_resized * topdown_wei
        out1 = self.post(xs1)

        xs2 = 2 * xh * bottomup_wei
        out2 = self.post(xs2)
        return out1, out2


if __name__ == '__main__':

    model = FuseReduce(in_high_channels=512, in_low_channels=64)
    xh = torch.randn(1, 512, 16, 16)
    xl = torch.randn(1, 64, 8, 8)
    out1, out2 = model(xh, xl)
    print(out1.shape, out2.shape)
