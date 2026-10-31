# Published comparator for the current own-inference run

Verified 2026-10-30: [FreeZeV2 paper v1](https://arxiv.org/pdf/2506.09784v1), Table II, page 7; Sections III-A and IV-A–D, pages 3 and 6–8.

**Closest comparator: row 11, CNOS-only FreeZeV2 (71.1 mean pose AR).** Accurate reaches 82.1. Values are percentages; time is seconds/image. [Primary paper, Table II](https://arxiv.org/pdf/2506.09784v1#page=7).

| Table II row | Published variant | Mask sources | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | Mean AR | Time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | FreeZeV2 | CNOS | 69.4 | 56.3 | 93.5 | 54.5 | 60.0 | 76.9 | 87.4 | 71.1 | 1.5 |
| 15 | FreeZeV2 | SAM-6D | 73.3 | 59.7 | 95.2 | 60.9 | 63.3 | 80.1 | 87.8 | 74.3 | 1.4 |
| 18 | FreeZeV2 | CNOS + SAM-6D + NIDS + MUSE | 75.9 | 72.9 | 97.2 | 66.3 | 71.2 | 86.0 | 91.3 | 80.1 | 4.4 |
| 19 | FreeZeV2-Accurate | CNOS + SAM-6D + NIDS + MUSE | 77.1 | 75.5 | 97.6 | 69.7 | 74.2 | 89.2 | 91.5 | 82.1 | 24.8 |

Table II evaluates RGB-D, model-based 6D localization with object identities/counts known. Pose AR averages VSD, MSSD and MSPD recall. Base localization uses N+1 masks per source. Accurate adds SAR, up to 2N masks and rendered-pose visual rescoring; the paper identifies it as leaderboard entry FreeZeV2.1. Table III instead evaluates 6D detection AP and is not this run's metric. [Primary paper, Sections IV-A–D](https://arxiv.org/pdf/2506.09784v1#page=6).

## Scope of our comparison

Our fixed `FreeZeV2-Re-base-l30-public-v1` executes the pinned third-party base source with CNOS-FastSAM-public-BOP23 masks. It has no SAR, extra mask-source ensemble or rendered-pose visual rescoring. The exact configuration, original source revision, mask-file hashes and BOP19 localization target lists are recorded in `public_inference_v1.json` and each run identity. No configuration is selected using test GT.

The paper's row 11 labels its masks CNOS, without specifying the exact SAM/FastSAM variant or prediction archive used for that row. Exact mask parity remains unresolved. Our declared layer, depth tolerance, NMS distance and raw sampling assumptions also remain explicit; the paper does not supply a complete inference preset. Therefore row 11 is a published reference, not proof of identical experimental inputs or author-code equivalence.

Compare each measured own-inference dataset AR with the corresponding row-11 value and report differences in percentage points. Keep all official targets, including failures, in the denominator. Do not report a seven-dataset measured mean without compatible HB and ITODD metrics. Evaluating authors' Accurate CSVs is a separate evaluator/reference check and never counts as our own generated performance.

Published timings provide context only. Our hardware and the public reconstruction's NumPy CPU registration differ from the paper's execution; report our actual onboarding and inference runtimes separately. This document records comparison targets and does not assert any measured benchmark result.
