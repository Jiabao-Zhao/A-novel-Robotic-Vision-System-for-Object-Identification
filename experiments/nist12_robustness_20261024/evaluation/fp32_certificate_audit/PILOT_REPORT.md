# Bounded FP32 GPU pilot: exact parity, no speed win at this size

The coordinator authorized one isolated live-load pilot after independent proof
review. Scientific fit jobs continued unchanged. No production proxy was
installed, model imported, pose fit run or new accuracy annotation read.

The successful receipt is
`gpu_pilot/20261024T195248234436Z/result.json`. The earlier
`gpu_pilot/20261024T195217269592Z` attempt failed during NVRTC compilation before
CUDA initialization because the loader could not find the already-installed
builtins library by soname. The isolated pilot then preloaded that same existing
library by absolute path; no package, linker configuration or environment change
was needed. Both attempts remain saved.

## Exactness and bounds

All4,096 unique parity rows pass the pinned SciPy1.18.1/NumPy2.2.6 final
distance/index byte checks. Raw GPU first/second FP32 values match CPU ordered
FP32 emulation, including ties, a selected subnormal-scale point with distant
competitors, the coordinate-cap boundary, strided conversion and multibatch
normalization. Raw tie-broken GPU IDs also match the emulation's first argmin.
The actual kernel used explicit separate RN operations; generated PTX contains
15sub.rn.f32,15mul.rn.f32 and10add.rn.f32 instructions and no FMA/float-MAD/FTZ
arithmetic. Source/options/compiler/builtins/PTX hashes are in the receipt.

The cached label-free case certifies2,431/3,072 rows(79.13%);641 use the original
four-thread CPU tree. Every ambiguity and guard failure retains exact fallback.
This fraction applies only to the saved raw-pixel/previous-predicted-pose sample,
not to all voxel clouds or all ICP iterations.

## Live-load timing

Each warm hybrid call includes both CAD and query FP32 conversion/upload, exact
outward-bound calculation, kernel/synchronization, returned arrays, CPU selected
distance recomputation and four-thread original-tree fallback. CPU tree
construction and reusable device-buffer allocation are setup costs. Three
alternating timing passes reuse the same3,072 cached queries.

| Method | Three wall times(ms) | Median(ms) |
| --- | --- | ---: |
| Original SciPy, four threads | 5.846 /3.396 /3.379 | **3.396** |
| FP32 GPU certificate + exact CPU fallback | 7.075 /5.765 /3.255 | **5.765** |

Median native/hybrid speed is0.589times: the GPU path loses at this small size
under the continuing live CPU workload. The last pass is close and timings vary;
three passes do not establish a stable dense-size crossover. No larger query or
complete-fit test was authorized or run.

Reported setup components are0.324s for NVRTC compilation/PTX inspection and
0.494s for CUDA initialization/module/buffer setup. These component timers do not
cover every process-start/import, library-load, binary-hash or C-helper compile
cost. They are excluded from the warm timing above; total process startup was
not separately instrumented.

## Resource limits and teardown

Peak explicit device buffers:213,828bytes, below128MiB. Peak process RSS:
254,857,216bytes(243.05MiB), starting from69,881,856bytes(66.64MiB), an increase
of176.41MiB and below the768MiB additional-host limit. These tracked device
buffers exclude driver/context-managed memory, which is not exposed by this
allocation counter; available device memory after initialization is recorded.

All buffers were freed, the module unloaded and context destroyed in `finally`,
with no cleanup errors and zero tracked device bytes retained. The process
exited successfully. Torch was never imported. CPU fit jobs were never paused.

Production remains on the validated CPU query proxy. Any larger pilot or cached
whole-bank replay requires a separate coordinator decision; this receipt does
not establish whole-study acceleration or authorize deployment.
