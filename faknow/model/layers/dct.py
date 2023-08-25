from typing import Tuple, Union

import torch
from torch import nn
"""
layers for Discrete Cosine Transform in MCAN
"""


def conv2d_bn_relu(in_channels,
                   out_channels,
                   kernel_size,
                   stride=1,
                   padding: Union[str, int, Tuple[int, int]] = 0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DctStem(nn.Module):
    def __init__(self, kernel_sizes, num_channels):
        super(DctStem, self).__init__()
        self.convs = nn.Sequential(
            conv2d_bn_relu(in_channels=1,
                           out_channels=num_channels[0],
                           kernel_size=kernel_sizes[0]),
            conv2d_bn_relu(
                in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=kernel_sizes[1],
            ),
            conv2d_bn_relu(
                in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=kernel_sizes[2],
            ),
            nn.MaxPool2d((1, 2)),
        )

    def forward(self, dct_img):
        x = dct_img.unsqueeze(1)
        img = self.convs(x)
        img = img.permute(0, 2, 1, 3)

        return img


class DctInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channel=128,
        branch1_channels=None,
        branch2_channels=None,
        branch3_channels=None,
        branch4_channels=None,
    ):
        super(DctInceptionBlock, self).__init__()

        if branch4_channels is None:
            branch4_channels = [32]
        if branch3_channels is None:
            branch3_channels = [64, 96, 96]
        if branch2_channels is None:
            branch2_channels = [48, 64]
        if branch1_channels is None:
            branch1_channels = [64]

        self.branch1 = conv2d_bn_relu(in_channels=in_channel,
                                      out_channels=branch1_channels[0],
                                      kernel_size=1)

        self.branch2 = nn.Sequential(
            conv2d_bn_relu(in_channels=in_channel,
                           out_channels=branch2_channels[0],
                           kernel_size=1),
            conv2d_bn_relu(
                in_channels=branch2_channels[0],
                out_channels=branch2_channels[1],
                kernel_size=3,
                padding=(0, 1),
            ),
        )

        self.branch3 = nn.Sequential(
            conv2d_bn_relu(in_channels=in_channel,
                           out_channels=branch3_channels[0],
                           kernel_size=1),
            conv2d_bn_relu(
                in_channels=branch3_channels[0],
                out_channels=branch3_channels[1],
                kernel_size=3,
                padding=(0, 1),
            ),
            conv2d_bn_relu(
                in_channels=branch3_channels[1],
                out_channels=branch3_channels[2],
                kernel_size=3,
                padding=(0, 1),
            ),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),
            conv2d_bn_relu(in_channels=in_channel,
                           out_channels=branch4_channels[0],
                           kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = out.permute(0, 2, 1, 3)

        return out
