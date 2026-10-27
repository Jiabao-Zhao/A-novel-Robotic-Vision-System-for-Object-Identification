# One cached full-bank parity replay

The independently reviewed query proxy passed the authorized R00 / Gear_Large / proposal 0 replay. All 432 hypotheses were recomputed in a new technical output directory, with no scientific-cache writes, new required fit keys, models or accuracy annotations. This is exactness evidence for this cached bank, not production integration.

The process exited normally with CUDA destroyed, zero retained tracked device bytes and no cleanup errors. Protected sources, inputs and cached scientific evidence retained their hashes.

## Output agreement

Output: `cached_replay/20261024T203452368531Z/`.

- 24,424 floating-point leaves have identical float64 bits.
- 6,965 remaining leaves have identical types and values.
- Every non-`seconds` dictionary key and ordered list element agrees, including all hypotheses, poses, histories, stop reasons, scores, ties and winners.
- Canonical non-timing JSON SHA-256 for both banks: `8faa45fbdb6ca666e75e8f304002ef904f6f7fa5ea7f36f074f7f4b478cc1194`.

The proxy performed 527 GPU queries over 63,833,670 rows. It certified 62,881,376 rows; 952,294 rows used the unchanged original SciPy query. No small-query rows occurred in this bank. The CAD was uploaded once and reused 526 times. Supported CPU work used the explicitly approved four-thread execution schedule; unsupported calls retain their original arguments and exceptions.

## Timing and resources

The complete replay took **37.914782 seconds** under the ongoing live workload, including lazy GPU setup, conversion, transfers, certificates, fallback and recomputation within the fit. NVRTC compilation/inspection took 0.169415 seconds and CUDA initialization 0.190615 seconds. This was not paired with a simultaneous CPU control. The original cache records 91.283923 seconds, and an earlier separate CPU4 technical replay records about 40.26 seconds; differences in load and execution schedule prevent treating either ratio as a measured full-fit speedup. Earlier paired dense query-only pilots remain the evidence for query throughput.

Peak process RSS was 557,461,504 bytes (531.64 MiB), an increment of 487,809,024 bytes (465.21 MiB). Explicit device allocation peaked at 7,467,844 bytes (7.122 MiB). Both were below the authorized limits. The fresh geometry clearance reported 6.2913 GiB available and a 5.2913 GiB reserve after the full 1 GiB allowance. No production job was paused or restarted.

## Frozen receipts

| Item | SHA-256 |
|---|---|
| Independent wrapper review | `6f4365e81f534e535c74b03d15e9fcb1d2c82524d00a13be29c36b9ddd29e196` |
| Fresh memory clearance | `46ad32ea80294510450140088af710e581c9ea843108357b79057fe0978a6525` |
| Proxy source | `c2ce6453bdf8c9baa0ca57366a47a051cf6154645707f6b3a83de993bd84b34a` |
| Replay source | `712022eb0b210f0330bd2ee1c95262a1a28d04f89be7cb92a2555d54c39ed7f2` |
| Replay completion | `3d50e5c7e87cbe35f1e5c7b123cd5eb001531b2848faffa97d1ca0e1061153b8` |
| Runtime completion | `7cdbb87d71fc607310221decc9da3fea4be074ffbba87a5966ee2c2c288fdac8` |
| New technical bank | `67a738972c245d69e1d4888fb077ac381b113ccf6eefed84d62fffa91781fdeb` |

The reviewed proof and binary/compiler/FP guards remain build-specific. They do not authorize another scientific fit or production integration. No live adapter or source was changed.
