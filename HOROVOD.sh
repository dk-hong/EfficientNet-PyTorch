#!/bin/bash

gpu_num=4
workers=4

# HOROVOD
horovodrun -np "${gpu_num}" python examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --use-horovod

# HOROVOD + mixed_precision
horovodrun -np "${gpu_num}" python examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --use-horovod --mixed-precision

# HOROVOD + workers
horovodrun -np "${gpu_num}" python examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --use-horovod --n-workers "${workers}"

# HOROVOD + mixed_precision + workers
horovodrun -np "${gpu_num}" python examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --use-horovod --mixed-precision --n-workers "${workers}"