# Independent review of the isolated query proxy

Source review passes for the separately authorized **single cached R00 / Gear_Large / proposal 0 bank replay**, conditional on a fresh geometry-agent memory clearance. This is not production approval or permission to fit another scientific key. No GPU query, fit, model call or new accuracy annotation was used in this review.

The current sources address the earlier findings: the fingerprint covers opaque node/split state as well as point order and bounds; cache budgets are checked before CAD snapshots/device allocation; oversized worker integers retain the original SciPy error; and the original pilot, floating-point helper, compiler libraries and PTX are pinned. The saved CPU-only receipt records 13 passing tests against the reviewed proxy/test hashes. Those tests use a fake FP32 backend and are not GPU evidence.

## Query semantics and exactness

- Only a fresh bare original nonperiodic 3D cKDTree is wrapped. Nested proxies are rejected. The original tree remains the fallback; point order is retained, separate CAD buffers cannot overwrite each other, and every accelerated call fingerprints the current tree. Concurrent mutation is outside the supported contract, as for original SciPy inputs.
- The accelerated signature is scalar k=1, p=2, eps=0, infinite distance bound, finite FP32/FP64 NumPy input, and a whitelisted worker value. Unsupported signatures, invalid workers, shapes and trees forward their original arguments and exceptions. Root explicitly approved `cpu_workers=4` for supported small calls and uncertified rows. This is a declared execution schedule change; the unsupported-call worker argument remains unchanged.
- Original binary64 coordinates, actual tree depth and reviewed per-row outward bounds determine eligibility. Every raw flag, finite/order/nonnegative/index check and row-bound mask is required before certification. Ties and uncertain rows use the original tree. Flattening, chunk placement and reshaping retain row order, float64 distances and native index dtype.
- Certified distances use the previously reviewed pinned CPU operation sequence. Its justification is build-specific, not universal: SciPy's ordered XY/Z operations at 0x58add/0x58ae5 and 0x58af8-0x58b02, scalar square root at 0x59627, box update at 0x587d4/0x587d8, and NumPy SSE41 square roots at 0x37543f/0x375638. The old independent review contains binary hashes and disassembly evidence. The proxy retains those build/dispatch/RN/FTZ/DAZ checks and literal helper/compiler/PTX pins.
- The unchanged kernel's distinct-ID top-two merge and full-warp invalid propagation remain covered by the earlier review. The wrapper enforces 1D 128-thread launches, count/rounded-ID bounds, explicit buffers and the reviewed arithmetic kernel.

## Lifecycle and resources

One owner thread controls a lazy context/module and shared chunk buffers. Foreign-thread calls stay on CPU. Partial initialization and execution failures clean up; a failed accelerated call discards partial output and repeats the original whole call. A passing replay must have no disabled path, no cleanup errors, a destroyed context and zero retained tracked device bytes.

The CAD cache is limited to 4 MiB per CAD, 16 MiB total including entry allowance, and 32 entries. Device buffers are checked against 128 MiB before allocation. Chunks contain at most 262,144 rows; fewer than 16,384 eligible rows remain on CPU. The 1 GiB host-increment check observes process high-water RSS at guards, rather than preventing every possible transient allocation for arbitrary future calls. Approval is therefore limited to the fixed small nominal observation and requires the fresh host-reserve gate plus final measured peak. It does not establish a general memory guarantee for arbitrary inputs.

## Required replay evidence

The reviewed runner verifies this review and all reviewed-source hashes, then requires a geometry clearance no older than 120 seconds with at least the declared 2 GiB reserve plus 1 GiB allocation margin. It reconstructs the fixed cached key, verifies the original bank SHA, and runs all 432 hypotheses through a fresh local CAD object's wrapped tree. It cannot pass on an all-CPU run: GPU calls/rows and certified rows must be positive.

Every non-`seconds` key, ordered list, value type, float64 bit pattern and canonical JSON byte must match the saved complete bank. Frozen dependencies and scientific cache files must retain their hashes. Output is confined to a new technical directory; no required fit key or accuracy annotation is created/read. Historical recorded time is reported separately and is not treated as a paired speed control.

No remaining blocking source finding was identified for this one replay under those enforced conditions. Integration, larger workloads and restart decisions remain with root after the replay evidence and resource measurements exist.
