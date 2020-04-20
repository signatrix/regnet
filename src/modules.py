"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""
import torch.nn as nn


class Head(nn.Module):  # From figure 3

    def __init__(self, num_channels, num_classes):
        super(Head, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Stem(nn.Module): # From figure 3

    def __init__(self, out_channels):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        return x


class XBlock(nn.Module): # From figure 4
    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio=None):
        super(XBlock, self).__init__()
        inter_channels = out_channels // bottleneck_ratio
        groups = inter_channels // group_width

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

        if se_ratio is not None:
            se_channels = in_channels // se_ratio
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Conv2d(inter_channels, se_channels, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(se_channels, inter_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.se = None

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
        self.rl = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x1 = self.conv_block_2(x1)
        if self.se is not None:
            x1 = x1 * self.se(x1)
        x1 = self.conv_block_3(x1)
        if self.shortcut is not None:
            x2 = self.shortcut(x)
        else:
            x2 = x
        x = self.rl(x1 + x2)
        return x


class Stage(nn.Module): # From figure 3
    def __init__(self, num_blocks, in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio):
        super(Stage, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module("block_0", XBlock(in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio))
        for i in range(1, num_blocks):
            self.blocks.add_module("block_{}".format(i),
                                   XBlock(out_channels, out_channels, bottleneck_ratio, group_width, 1, se_ratio))

    def forward(self, x):
        x = self.blocks(x)
        return x