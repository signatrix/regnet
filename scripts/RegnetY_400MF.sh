#!/bin/bash
python train.py --bottleneck_ratio 1 \
                --group_width 8 \
                --initial_width 48 \
                --slope 28 \
                --quantized_param 2.1 \
                --network_depth 16 \
                --stride 2 \
                --se_ratio 4