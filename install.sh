#!/bin/bash

# while these networks can run on CPU it is *highly* recommended
# that you use a GPU, torch-geometric supports nVidia CUDA.
# We recommend using CUDA 10.2, the default below
# if you need to change this please see the list of available wheels here:
# https://github.com/rusty1s/pytorch_geometric#pytorch-160
CUDA=cu102

pip install cython
pip install git+https://github.com/eldridgejm/unionfind.git
pip install torch
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric
