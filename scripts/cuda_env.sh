#!/usr/bin/env bash
# CUDA paths supplied by the shared research environment. Source this file.
SHARED_CUDA_SITE=/backup01/zhangyihan/envs/research/lib/python3.11/site-packages/nvidia
SHARED_ENV_LIB=/backup01/zhangyihan/envs/research/lib
SHARED_CUDA_LIBS="$SHARED_CUDA_SITE/cublas/lib:$SHARED_CUDA_SITE/cuda_cupti/lib:$SHARED_CUDA_SITE/cuda_nvrtc/lib:$SHARED_CUDA_SITE/cuda_runtime/lib:$SHARED_CUDA_SITE/cudnn/lib:$SHARED_CUDA_SITE/cufft/lib:$SHARED_CUDA_SITE/cufile/lib:$SHARED_CUDA_SITE/curand/lib:$SHARED_CUDA_SITE/cusolver/lib:$SHARED_CUDA_SITE/cusparse/lib:$SHARED_CUDA_SITE/cusparselt/lib:$SHARED_CUDA_SITE/nccl/lib:$SHARED_CUDA_SITE/nvjitlink/lib:$SHARED_CUDA_SITE/nvshmem/lib:$SHARED_CUDA_SITE/nvtx/lib"
export LD_LIBRARY_PATH="$SHARED_ENV_LIB:$SHARED_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
