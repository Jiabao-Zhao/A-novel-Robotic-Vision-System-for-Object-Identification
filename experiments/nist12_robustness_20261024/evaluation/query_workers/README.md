# Nearest-neighbor worker audit

This audit changes no frozen kernel, fit, cache key or model. It runs only
nearest-neighbor queries, using SciPy 1.18.1, synthetic arrays and bounded
observed inputs with a previously saved predicted transform. No pose search or
accuracy annotation is used.

The installed `cKDTree.query` documentation describes `workers` as the number
of parallel workers. Its mathematical parameters remain `k=1`, `eps=0`, `p=2`
and an infinite distance upper bound. The wheel includes a compiled extension
and Python delegating wrapper, but not the Cython/C++ kernel sources.

The official tagged [Cython wrapper](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/_ckdtree.pyx)
splits input rows into disjoint contiguous ranges and gives each thread its own
output slice. All ranges call the same kernel with the same query parameters.
The [C++ query kernel](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/spatial/ckdtree/src/query.cxx)
processes each row independently, creating its search heap and node memory pool
inside the single-point function. No distance reduction or shared tie-breaking
operation spans query rows. For a fixed tree, point ordering and installed build,
this supports changing row scheduling without changing a row's calculations.
This is a source-based inference, not an API guarantee across SciPy versions.

`busy_host.json` records ten synthetic cases, including repeated coordinates,
equidistant ties, three-neighbor outputs, single/empty inputs, noncontiguous and
float32 input, bounded distances, and multidimensional/seed-shaped batches.
Workers 1, 2, 4 and 8 match in array shape, dtype and raw bytes for distances
and indices. A 1,024-row cached-input check also matches for every worker count.

The larger benchmark uses 131,072 foreground raw pixels from
`R00__depth_noise_2mm`, proposal 0, transformed using an existing nominal
predicted pose. The CAD tree has 10,651 native voxel samples. These are bounded
input queries, not an additional dense voxelization or fit. On the busy host,
one versus two workers takes median 0.476 versus 0.477 seconds: approximately
1.00× speed, with no measured benefit under contention. Peak process RSS is
95.3 MB, approximately 26.3 MB above the post-import baseline.

Any expanded CPU-window receipt is separate. Microbenchmark speed must not be
reported as whole-fit acceleration: projection and the remaining fit operations
are outside this audit. The first busy-host measurements do not establish
four-process throughput.

In the coordinated isolated window, `expanded.json` records full 131,072-row
distance/index arrays matching bitwise for workers 1, 2, 4 and 8. Median query
times are 0.2085, 0.09879, 0.05068 and 0.02938 seconds, respectively: 2.11×,
4.11× and 7.10× faster than one worker. Peak RSS remains approximately 95.3 MB.

A separate coordinated window measures four forked processes directly on the
same cached queries. Each trial executes sixteen batches (2,097,152 query rows),
with alternating thread-count order. Four processes with one query thread each
take median 0.8831 seconds; four processes with four query threads each take
0.4120 seconds: **2.14× query throughput**. Every returned distance and index
array matches the one-thread reference bitwise. Peak aggregate PSS is 73.2 MB;
summed RSS is 319.6 MB because fork-shared pages are counted repeatedly. The
250 MiB aggregate PSS guard passes. The second window's automatic release marker
resumes the verified fit processes; this script sends no process signals.

`four_process_throughput.json` contains all trial timings and memory accounting.
This repeated-array microbenchmark establishes neither full-fit speed nor a
whole-study completion time. Any whole-fit estimate must use actual completed
required fits. No query proxy or fitting change was installed by this audit.
