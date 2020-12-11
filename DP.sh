#!/bin/bash

workers=4

# DP
python examples/imagenet/main.py --data /data/image/ -a efficientnet-b0

# DP + mixed_precision
python examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --mixed-precision

# DP + workers
python examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --n-workers "${workers}"

# DP + mixed_precision + workers
python examples/imagenet/main.py --data /data/image/ -a efficientnet-b0 --mixed-precision --n-workers "${workers}"