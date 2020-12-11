#!/bin/bash

workers=4

# DDP
python -m torch.distributed.launch --nproc_per_node=1 examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --multiprocessing-distributed

# DDP + mixed_precision
python -m torch.distributed.launch --nproc_per_node=1 examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --multiprocessing-distributed --mixed-precision

# DDP + workers
python -m torch.distributed.launch --nproc_per_node=1 examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --multiprocessing-distributed --n-workers "${workers}"

# DDP + mixed_precision + workers
python -m torch.distributed.launch --nproc_per_node=1 examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --multiprocessing-distributed --mixed-precision --n-workers "${workers}"
