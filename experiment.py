"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""
import torch
from torchsummary import summary
from pthflops import count_ops
from src.regnet import RegNet
from src.config import INPUT_RESOLUTION

def main():
    bottleneck_ratio = 1
    group_width = 16
    initial_width = 32
    slope = 5
    quantized_param = 2.5
    network_depth = 40
    stride = 2
    model = RegNet(initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride)
    # model.cuda()
    dummy_images = torch.rand(1, 3, INPUT_RESOLUTION, INPUT_RESOLUTION)
    # dummy_images = dummy_images.cuda()
    # out = model(dummy_images)
    count_ops(model, dummy_images, verbose=False)
    # for _ in range(100000):
    #     out = model(dummy_images)
    summary(model, (3, INPUT_RESOLUTION, INPUT_RESOLUTION), device="cpu")


if __name__ == '__main__':
    main()