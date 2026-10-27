# Dense-voxel query scaling: exact parity and1.6times query speed

The separately authorized live-load pilot completed in
`gpu_pilot_large/20261024T195757142266Z`. Every output distance and original CAD
point index matches four-thread SciPy bitwise at all three sizes and all three
warm repetitions. An additional4,096-row spread subset also matches ordered
CPU-FP32 raw first/second distances and first IDs. The kernel, reviewed proof,
extension hashes and generated PTX hash are unchanged from the small pilot.
No pose fit, model request, production change, accuracy-label read or fit-job
pause occurred.

The input is now actual native voxel means of the full noisy observation:
2,094,540 raw pixels produce1,279,564 voxels for
`R00__depth_noise_2mm/proposal0`. Integer-spaced sampling selects262,144 distinct
rows across that full array. Nested strides16/4/1 supply16,384/65,536/262,144
rows. A previously saved nominal predicted Gear pose transforms observations
into native CAD coordinates; no simulator pose is used. The exact observation
key, full voxel signature, sampled rows, CAD points, predicted pose and all
source/input/output hashes are preserved under `prepared/` and in the receipt.

| Query rows | SciPy4 median(ms) | Complete hybrid median(ms) | Native/hybrid speed | Certified rows |
| --- | ---: | ---: | ---: | ---: |
|16,384 |22.338 |20.445 |1.09times |9,944(60.69%) |
|65,536 |88.912 |54.919 |1.62times |39,529(60.32%) |
|262,144 |278.478 |173.353 |1.61times |158,095(60.31%) |

The hybrid timings include normalization/FP guards, exact outward error-bound
work, both CAD/query conversion and upload on every call, kernel/synchronization,
download, certificate calculation, exact selected-distance CPU recomputation and
original four-thread fallback. All scientific jobs continued, so these are
live-load measurements, not isolated CPU throughput or full-fit acceleration.
At262,144 rows,104,049 rows(39.69%) still require the CPU tree.

Median measured components at262,144 rows:

| Component | Milliseconds |
| --- | ---: |
| Original CPU fallback |123.055 |
| GPU kernel and synchronization |21.728 |
| CPU selected-distance recomputation |13.137 |
| Certificate |5.023 |
| Coordinate maximum and exact bound |4.159 |
| FP32 conversion |1.740 |
| Download/output allocation |1.607 |
| Normalization and FP check |1.476 |
| Upload |1.186 |

Component medians are calculated separately and need not sum to the median
end-to-end time. CPU fallback is the main measured remaining cost. The
coordinate maximum here spans the entire query batch, so its conservative bound
is looser for many rows than the same reviewed formula evaluated per row would
be. No tighter-bound implementation or additional measurement was performed.

The exact voxel preparation ran in a separate CPU process that exited before
CUDA initialization. It took4.265s and peaked633.72MiB total RSS,
567.92MiB above its baseline, with a704MiB additional address-space limit. The
GPU pilot parent peaked247.70MiB total RSS,179.77MiB above baseline. Both pass
the768MiB incremental host cap; explicit device buffers peak7.12MiB, below128MiB.
NVRTC compilation/PTX inspection took0.239s and CUDA context/module/buffer setup
0.203s, reported separately from warm calls. These component setup timers do not
include every process/import/hash/helper-compilation cost.

All buffers were freed, the module unloaded and context destroyed in `finally`,
with no cleanup errors and zero tracked device bytes retained. The process
exited successfully and never imported Torch. This supports further consideration
of a bounded cached-bank parity replay, but no production installation or whole
study completion estimate follows from these query timings.
