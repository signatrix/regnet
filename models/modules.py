import torch
import torch.nn as nn


class Head(nn.Module):
    """
        Described on page 4, figure 3. If input image's size is 224x224,
        This layer gets input of size 7x7, and produce output of size 1.
    """

    def __init__(self, num_channels, num_classes):
        super(Head, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Stem(nn.Module):
    """
        Described in section 3.2 page 4, figure 3. If input image's size is 224x224,
        This layer gets input of size 224x224, and produce output of size 1.
    """

    def __init__(self, out_channels):
        """
        :param in_channels:
        :param out_channels: w_0
        """
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        return x


class XBlock(nn.Module):
    """
        Described in section 3.2 page 4, figure 4.
    """
    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width, stride):
        """
        :param in_channels: w_i
        :param bottleneck_ratio: b_i
        :param group_width: g_i
        """
        super(XBlock, self).__init__()
        inter_channels = out_channels // bottleneck_ratio

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, groups=group_width, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
        self.rl = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x1 = self.conv_block_2(x1)
        x1 = self.conv_block_3(x1)
        if self.shortcut is not None:
            x2 = self.shortcut(x)
        else:
            x2 = x
        x = self.rl(x1+x2)
        return x

class Stage(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, bottleneck_ratio, group_width):
        super(Stage, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module("block_0", XBlock(in_channels, out_channels, bottleneck_ratio, group_width, 2))
        for i in range(1, num_blocks):
            self.blocks.add_module("block_{}".format(i), XBlock(out_channels, out_channels, bottleneck_ratio, group_width, 1))

    def forward(self, x):
        x = self.blocks(x)
        return x


