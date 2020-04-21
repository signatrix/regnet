#!/bin/bash
python train.py --bottleneck_ratio 1 \
                --group_width 16 \
                --initial_width 48 \
                --slope 33 \
                --quantized_param 2.3 \
                --network_depth 15 \
                --stride 2 \
                --se_ratio 4