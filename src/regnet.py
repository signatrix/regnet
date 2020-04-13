import numpy as np
from src.anynet import AnyNetXe

class RegNet(AnyNetXe):
    def __init__(self, initial_width, slope, quantized_param, network_depth):
        parameterized_width = initial_width + slope * np.arange(network_depth)  # From equation 2
        parameterized_block = np.log(parameterized_width/initial_width)/np.log(quantized_param)  # From equation 3
        parameterized_block = np.round(parameterized_block)
        quantized_width = initial_width * np.power(quantized_param, parameterized_block)
        # We need to convert quantized_width to make sure that it is divisible by 8
        quantized_width = 8 * np.round(quantized_width/8)
        ls_block_width, ls_num_blocks = np.unique(quantized_width.astype(np.int))
        ls_block_width = ls_block_width.tolist()
        ls_num_blocks = ls_num_blocks.tolist()

        # super(AnyNetXe, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width)