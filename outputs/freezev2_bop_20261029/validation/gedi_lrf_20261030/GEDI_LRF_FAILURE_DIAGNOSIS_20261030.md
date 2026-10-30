# Original GeDi planar-LRF failure: CPU evidence

The three failed CAD onboardings and the sampled observed-mask failure are
caused by an undefined tangent direction in the original GeDi local reference
frame (LRF), before the neural network or CUDA is used. They are not explained
by nonfinite input clouds, a bad checkpoint, missing dependencies, CUDA memory
or a dead worker. The healthy finite-feature experiment can continue unchanged.

## Source mechanism

External source revision `b3dd86776750d8221f89d39975118da9839b39f7`, `gedi.py`
SHA256 `50f23311ad3173c92f834c637e63c9c0bb6fbd8f12c58dd69773a7aba6350249`.
The exact [original LRF](https://github.com/fabiopoiesi/gedi/blob/b3dd86776750d8221f89d39975118da9839b39f7/gedi.py#L140)
computes a normal by covariance SVD. It weights each projected tangent vector
by squared normal displacement (`beta = norm * norm`), sums these vectors into
`vi_c`, then normalizes it at line184 without an epsilon or degeneracy branch.
Although its constructor stores `eps=1e-12`, the original forward never uses it.

For an exactly planar neighborhood, every normal displacement is exactly zero.
Consequently all beta weights and `vi_c` are zero, irrespective of how many
valid planar support points exist. `vi_c / ||vi_c||` becomes0/0 and the LRF and
network-input patch contain NaNs. This is an undefined in-plane direction, not
small-norm underflow. A perfect plane can provide its normal but this particular
weighted tangent rule has no direction to normalize.

## Exact cached-CAD replay

CPU-only original Open3D radius search and AST-extracted original `LRF` were
executed. The diagnostic replay preserves float32 input conversion, NumPy seed0,
500-query batches, original4000-neighbor sampling/padding, original512-point
LRF output sampling and the .3→.4diameter RNG sequence. It loads no descriptor
model. The active source/checkpoints/protocol and cached geometry are read-only.
The diagnostic uses2 CPU threads and reports `torch.cuda.is_initialized()==False`.

All input queries/support are finite, with5000 queries and20,000 support points
per CAD. Every nonfinite patch was explained by exactly zero `vi_c`; no norm
underflow witness occurred.

| CAD | Nonfinite .3diameter LRF patches | Nonfinite .4diameter replay patches | First .3 witness index |
|---|---:|---:|---:|
|27|12/5000|0/5000|452|
|28|10/5000|0/5000|904|
|30|6/5000|0/5000|163|
|1 (successful control)|0/5000|0/5000|none|

The real frozen adapter aborts validation after nonfinite .3 output, so it never
reaches .4 for these CADs. The .4 figures are an explicitly hypothetical CPU
continuation with the same sampling order, not successful actual descriptors or
permission to substitute the larger radius.

The rejection of an entire CAD cache after any nonfinite row is this independent
implementation's strict finite-output guard. It is distinct from the original
LRF's per-query NaNs and is not verified FreeZeV2 author handling. Most queried
rows remain finite. Reported baseline scores must retain this guard policy and
its failure counts as implementation limitations.

First witnesses have constant sampled Z, smallest covariance singular value0,
normal offsets0 and `vi_c=[0,0,0]`:

| CAD | Query point Z (m) | Distinct neighbors in radius | Sampled Z range |
|---|---:|---:|---:|
|27|.028000000864|2455|0|
|28|.024199999869|2655|0|
|30|.025299999863|1939|0|

Padding to4000 does not create the undefined plane; the original radius support
is already entirely planar. No GT poses or pose quality were inspected.

## Observed failure and synthetic witness

The actual failed candidate is T-LESS scene3/image476, object8, CNOS-FastSAM
record7747, under `datasets/tless/images/000003_000476/obj_000008/cnos-fastsam_007747`.
Its saved mask, original sensor depth/calibration and unchanged original
`observed_clouds` function reconstruct91 sparse queries. Query32 is the sole
nonfinite LRF patch at both radii. It has5/.3 and6/.4 radius neighbors, all at
exact Z2.3081998825m; smallest covariance singular value0, normal offsets0 and
`vi_c=[0,0,0]`. This proves the same pre-network defect in one actual observed
candidate; it does not establish how often it occurs across the full dataset.

A synthetic exact plane with4000 support points and two queries gives NaN input
patches for2/2 queries when run through the unmodified original LRF. This holds
even with its stored `eps=1e-12`, because that parameter is unused.

## Faithful-source choices and limits

The unchanged-source baseline should retain its precise failure states and all
finite predictions. Reinstalling weights, restarting GPU workers, changing CUDA
batches or retrying identical samples cannot supply a direction to this zero
vector. Do not mix a repair into the active frozen run.

No verified FreeZeV2 author inference implementation has supplied a planar-LRF
policy. Therefore no alternative below can be called a demonstrated original
FreeZeV2 behavior:

- **Strict original behavior:** retain the failure and report its effect on
  coverage, with the complete official target denominator. This preserves the
  current declared method and source evidence.
- **Versioned invalid-descriptor exclusion:** a separate experiment can retain
  only finite descriptor rows consistently across points, modalities and both
  radii, with explicit per-CAD/per-mask exclusion counts. This avoids inventing
  a descriptor but changes the5000-point/sparse-point recipe and may leave
  insufficient correspondences. It is a documented robustness variant, not an
  invisible infrastructure repair.
- **Versioned defined tangent rule:** use an independently specified
  deterministic tangent construction for degenerate planes, validated on
  synthetic rigid rotations/partial planes before any score evaluation. This
  changes the original GeDi input frame and may affect learned descriptor
  invariance. Only a verified author-released rule could establish author parity.

An epsilon alone maps the zero vector to a zero tangent, producing a rank-deficient
LRF rather than a valid orthonormal frame. `nan_to_num`, random features/jitter,
resampling until success, radius substitution or silent float/threshold changes
are not established faithful fixes. No such action was taken. Any versioned
handling must be frozen using geometric/synthetic checks, not BOP test-GT tuning.

## Retained audits

- `gedi_lrf_cpu_diagnosis_20261030.json`, SHA256
  `e690c25883e11aeff7c11ee20a34334b24a91b0d7b4e96b53429dc964b8116dc`;
  exact CAD radius replays, point indices/examples, source/geometry hashes;
  runtime9.78s.
- `gedi_lrf_cpu_additional_diagnosis_20261030.json`, SHA256
  `dfdb50f27fcb82dade4d22f06e7fca834cceaf34fe8e01b51f8fe4a5c37e5d46`;
  exact synthetic-plane and failed observed candidate evidence.
- Reproducible CPU-only diagnostic helpers are adjacent. They do not load GeDi
  weights or DINOv2, access test GT or modify the active run.
