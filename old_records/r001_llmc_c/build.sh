#!/bin/bash
# Builds llm.c's train_gpt2.cu (BF16, cuDNN attention, NCCL, no MPI) into $BUILD/train_gpt2cu.
#   NVCC     the compiler (CUDA 12.6)
#   NVIDIA   a pip "nvidia" package dir with cuDNN 9, NCCL 2 and cuBLAS 12 (the torch 2.10 venv's)
#   CUDNN_FE cudnn-frontend's checkout (v1.7.0, the release llm.c used in October 2024; cloned into $BUILD if unset)
#   BUILD    the output directory (default build/ next to this script)
#   CARGO    cargo (for cpuval/, the CPU validation build: a Rust static library)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-12.6/bin/nvcc}
NVIDIA=${NVIDIA:-/sphinx/u/brunborg/venvs/record-torch210/lib/python3.12/site-packages/nvidia}
BUILD=${BUILD:-$HERE/build}
mkdir -p "$BUILD/lib"
if [ -z "$CUDNN_FE" ]; then
  CUDNN_FE=$BUILD/cudnn-frontend
  [ -d "$CUDNN_FE" ] || git clone -q --depth 1 --branch v1.7.0 https://github.com/NVIDIA/cudnn-frontend.git "$CUDNN_FE"
fi
# the wheels ship versioned .so files only
ln -sf "$NVIDIA/cublas/lib/libcublas.so.12" "$BUILD/lib/libcublas.so"
ln -sf "$NVIDIA/cublas/lib/libcublasLt.so.12" "$BUILD/lib/libcublasLt.so"
ln -sf "$NVIDIA/cudnn/lib/libcudnn.so.9" "$BUILD/lib/libcudnn.so"
ln -sf "$NVIDIA/nccl/lib/libnccl.so.2" "$BUILD/lib/libnccl.so"
CUDA=$(dirname "$(dirname "$NVCC")")
ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90"
FLAGS="--threads=0 -t=0 --use_fast_math -std=c++17 -O3 $ARCH -DMULTI_GPU -DENABLE_BF16 -DENABLE_CUDNN"
INC="-I$HERE -I$NVIDIA/nccl/include -I$NVIDIA/cudnn/include -I$CUDNN_FE/include -I$CUDA/include"
RPATH="$NVIDIA/cudnn/lib:$NVIDIA/nccl/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_nvrtc/lib:$NVIDIA/nvjitlink/lib"
LIBS="-L$BUILD/lib -lcublas -lcublasLt -lnccl -lcudnn -L$CUDA/lib64/stubs -lnvidia-ml -Xlinker -rpath -Xlinker $RPATH"
CARGO=${CARGO:-cargo}
"$CARGO" build --release --quiet --manifest-path "$HERE/cpuval/Cargo.toml" --target-dir "$BUILD/cpuval"
LIBS="$LIBS $BUILD/cpuval/release/libcpuval.a -lpthread -ldl -lm"
if [ ! -f "$BUILD/cudnn_att.o" ] || [ "$HERE/llmc/cudnn_att.cpp" -nt "$BUILD/cudnn_att.o" ]; then
  $NVCC -c $FLAGS $INC "$HERE/llmc/cudnn_att.cpp" -o "$BUILD/cudnn_att.o"
fi
$NVCC $FLAGS "$HERE/train_gpt2.cu" "$BUILD/cudnn_att.o" $INC $LIBS -o "$BUILD/train_gpt2cu"
echo "built $BUILD/train_gpt2cu"
