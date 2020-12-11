#!/bin/bash

gpu_num=4

# HOROVOD
horovodrun -np "${gpu_num}" python examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --use-horovod