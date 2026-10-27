# Isolated preprocessing memo audit

The new adapter is **not integrated**. It was not imported by any live fitting
runner, installed as a native hook, or used to generate predictions. Frozen
scoring/evaluation files and inference dependency hashes remain unchanged.

Eight preprocessing-only synthetic tests passed with:

```text
python -m unittest -v test_preprocessing_memo.py
Ran 8 tests in 0.019s
OK
```

Tests cover exact first-output reuse, original CAD-sample voxelization, equal but
distinct array fallback, changed voxel size, scene/proposal-ID reuse, changed
plane, empty input, strict single-entry eviction/memory bound and no global hook.
They are not whole-bank parity evidence and do not authorize integration.

The fixed existing R00 noise2mm/proposal0 benchmark, without fitting or labels,
measured 5.59 seconds of reusable preprocessing per repeated CAD: 0.173% of the
54-minute reference dense pin fit. That small saving does not justify restarting
an in-flight fit. See [protocol.json](protocol.json) and [result.json](result.json).
Timings are from one observation under uncontrolled concurrent system load.

Retained cache arrays plus metadata allowance are capped at128MiB and one entry;
measured retention was83.54MiB. The benchmark applied an896MiB additional
address-space cap, with measured569.5MiB incremental peak RSS and633.25MiB total
peak RSS. It made zero full-fit/model calls and read no new accuracy labels.

The finalizer's existing broad Python source inventory will hash these audit
scripts at prediction lock. That source inventory is not a statement that they
were inference dependencies. Live runner import/reference search was empty, and
the archived inference/kernel dependency hashes do not include these scripts.
No prediction lock existed when this was checked; do not change these files
after the eventual source lock.
