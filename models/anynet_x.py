import torch
import torch.nn as nn
from models.modules import Stem, Stage, Head
from models.config import *


class AnyNet(nn.Module):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width):
        super(AnyNet, self).__init__()
        assert len(ls_num_blocks) == 4
        assert len(ls_num_blocks) == len(ls_block_width)
        assert len(ls_num_blocks) == len(ls_bottleneck_ratio)
        assert len(ls_num_blocks) == len(ls_group_width)
        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module("stem", Stem(prev_block_width))

        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in enumerate(zip(ls_num_blocks, ls_block_width,
                                                                          ls_bottleneck_ratio, ls_group_width)):
            # print (i, num_blocks, block_width, bottleneck_ratio, group_width)
            self.net.add_module("stage_{}".format(i), Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width))
            prev_block_width = block_width
        self.net.add_module("head", Head(ls_block_width[-1], NUM_CLASSES))

    def forward(self, x):
        x = self.net(x)
        return x

if __name__ == '__main__':
    ls_num_blocks = [4,8,12,16]
    ls_block_width = [128,256,512,1024]
    ls_bottleneck_ratio = [1,2,2,4]
    ls_group_width = [2,2,2,4]
    model = AnyNet(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width)
    # print (model)
    dummy_images = torch.rand(8, 3, INPUT_RESOLUTION, INPUT_RESOLUTION)
    out = model(dummy_images)
    # print (out)
