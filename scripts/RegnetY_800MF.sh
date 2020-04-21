#!/bin/bash
python train.py --bottleneck_ratio 1 \
                --group_width 16 \
                --initial_width 56 \
                --slope 39 \
                --quantized_param 2.4 \
                --network_depth 14 \
                --stride 2 \
                --se_ratio 4