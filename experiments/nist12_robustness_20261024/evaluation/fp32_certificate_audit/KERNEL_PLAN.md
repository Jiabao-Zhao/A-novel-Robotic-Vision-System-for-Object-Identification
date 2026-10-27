# Unexecuted ordered-FP32 kernel draft

`ordered_top2.cu` is a new standalone audit asset. It has not been compiled or
launched, and production imports none of this directory. No CUDA context or
Torch process was initialized for the availability inspection.

## Existing local tools

Metadata/file inspection found these packages in the existing WSL virtualenv:

- Torch2.11.0, Triton3.6.0, Numba0.67.0.
- NVIDIA NVRTC13.0.88 and CUDA runtime13.0.96.
- cuda-bindings13.3.1 and cuda-pathfinder1.8.0.
- `/usr/bin/g++`; no `nvcc` executable on PATH; no CuPy package.
- Existing NVRTC library:
  `/home/jiabao/.venvs/lerobot-libero/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13`.
- Existing CUDA driver: `/usr/lib/wsl/lib/libcuda.so.1`.

An isolated NVRTC/driver-API process could compile/launch the draft without
installing anything or importing Torch. NVRTC compilation itself does not need
a GPU, according to [NVIDIA's NVRTC manual](https://docs.nvidia.com/cuda/nvrtc/index.html).
Compilation and GPU testing remain subject to the coordinator's separate go-ahead.

## Draft arithmetic and reduction

One warp compares one query exhaustively against all original CAD points. Four
warps per128-thread block is the initial proposed launch shape. Each thread
maintains two minima with distinct original IDs; warp shuffles merge those pairs
using only exact floating-point comparisons. Equal-distance different IDs yield
equal first/second distances, forcing CPU fallback under the margin rule.

Each squared distance uses explicit `__fsub_rn`, `__fmul_rn`, and `__fadd_rn`
intrinsics in x/y/z order. They have round-to-nearest semantics in the
[CUDA math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).
Proposed NVRTC options are `--std=c++17`, `--gpu-architecture=compute_86`,
`--fmad=false`, `--ftz=false`, `--prec-div=true`, and `--prec-sqrt=true`.
No fast-math, GEMM norm expansion, approximate top-k, TF32 or tensor-core
operation is used. The generated PTX must still be inspected before execution
to confirm ordered RN sub/mul/add and absence of FMA/reassociation/FTZ.

Host checks must cover original FP64 finite coordinates, coordinate cap2^60,
point_count>=2, supported index/grid bounds and block shape. Nonfinite/negative
kernel outputs or invalid indices make the complete row fall back. Conversion
and certificate threshold calculations must follow `DERIVATION.md`; GPU output
alone cannot certify itself. The original fixed-build SciPy tree remains the
fallback and final parity reference.

## Bounded test required before any production consideration

1. Independent proof review, then explicit coordinator go-ahead for one isolated
   GPU process with strict host/device memory limits and guaranteed teardown.
2. Compile and inspect PTX. Preserve source/options/PTX hashes and compiler log.
3. Test random, duplicate/equidistant, near-tie, unique-zero, underflow and
   overflow-guard cases, then the same<=4,096 cached query rows. Check raw GPU
   first/second values against CPU ordered-FP32 emulation. Check every final
   CPU-recomputed/certified-or-fallback distance and index bitwise against fixed
   SciPy1.18.1. Any mismatch stops the candidate.
4. Time end-to-end host conversion, transfers, kernel, transfer back, outward
   bound/certificate work, CPU selected-distance recomputation and original-tree
   fallback, against four-thread CPU SciPy. Separate initialization/compile time
   from warm timing, and report both. Stop if end-to-end timing loses.
5. Only if it wins, consider larger bounded rows and a cached complete-bank
   replay. Passing the small test is not permission for a production proxy or a
   full fit, nor proof of whole-study speed.

The previous FP64 eager-GPU path was6.21times slower on its cached test. This
new unexecuted draft changes arithmetic/performance strategy but provides no
evidence yet that the loss can be reversed.
