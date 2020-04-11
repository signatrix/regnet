"""
@author: Signatrix GmbH
"""
import torch
from torchsummary import summary
from pthflops import count_ops
from src.anynet import AnyNetX
from src.config import INPUT_RESOLUTION

def main():
    ls_num_blocks = [16, 16, 16, 16]
    # Block width must be divisible by bottleneck ratio
    ls_block_width = [128, 128, 128, 16]
    ls_bottleneck_ratio = [1, 2, 2, 4]
    ls_group_width = [2, 2, 2, 4]
    model = AnyNet(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width)
    dummy_images = torch.rand(1, 3, INPUT_RESOLUTION, INPUT_RESOLUTION)
    # model.cuda()
    # dummy_images = dummy_images.cuda()
    out = model(dummy_images)
    count_ops(model, dummy_images, verbose=False)
    # summary(model, (3, INPUT_RESOLUTION, INPUT_RESOLUTION))


if __name__ == '__main__':
    main()