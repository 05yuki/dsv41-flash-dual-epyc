#!/usr/bin/env bash
# Rebuild kt_kernel_ext for venv-dsv41 from source/ktransformers-gemma4/kt-kernel
# (the tree the venv's extension was built from on 08-28, plus local patches
# such as the task-queue timing of 09-13) and install it into the venv, keeping
# the previous .so beside it as .bak. Same toolchain and sysroot as the
# next-pair builds.
set -Eeuo pipefail
root=$HOME/KTransformers
venv="$root/venv-dsv41"
src="$root/source/ktransformers-gemma4/kt-kernel"
bld="$root/runtime/kt-dsv41-build"
export PATH="$venv/bin:/usr/local/cuda-13.1/bin:$PATH"
export NVCC_PREPEND_FLAGS="-U_GNU_SOURCE -D_DEFAULT_SOURCE -include $root/tools/cuda13-glibc-compat.h"
cmake -S "$src" -B "$bld" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON \
 -DKTRANSFORMERS_USE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
 -DCMAKE_CUDA_ARCHITECTURES=120 -DPYTHON_EXECUTABLE="$venv/bin/python" \
 -DCMAKE_PREFIX_PATH="$root/runtime/sysroot/usr" \
 -DCMAKE_CXX_FLAGS="-I$root/runtime/sysroot/usr/include -I$root/runtime/sysroot/usr/include/x86_64-linux-gnu" \
 -DCMAKE_LIBRARY_PATH="$root/runtime/sysroot/usr/lib/x86_64-linux-gnu"
cmake --build "$bld" --target kt_kernel_ext -j 16
so=$(ls "$bld"/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so)
dst="$venv/lib/python3.12/site-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
[ -f "$dst.bak" ] || cp "$dst" "$dst.bak"
cp "$so" "$dst"
echo "installed $(ls -la "$dst")"
