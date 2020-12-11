#!/bin/bash

# DDP
python -m torch.distributed.launch --nproc_per_node=1 examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --multiprocessing-distributed
