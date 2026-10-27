# First Qwen snapshot: unavailable outputs

**The exact reason for HTTP 400 is unknown.** The saved exception contains no
response body, and the readable standard Ollama server logs contain no current
run error. Neither a context overflow nor an image-size rejection is established.
No request was repeated and no prompt, image, parser or model output was changed.

The snapshot contains 8,742 description rows and 83 unavailable rows. The HTTP
failures account for 82 rows: 32 actual failed requests, two modes for each of 16
distinct image hashes, reused across 41 crop inputs. Every affected crop is
proposal 0 in a Gaussian-depth-noise condition: fourteen scenes at 1 mm,
fourteen at 2 mm, and thirteen at 5 mm. These are description-stage failures;
their target correctness has not been evaluated.

All sixteen rejected images are very large crops. Fourteen are 2,560 × 2,560
pixels; the other two are 2,560 × 2,129 and 2,560 × 2,142. Their boxes occupy
83.16%–100% of the original RGB frame. All 32 requests use the frozen 4,096
context setting and runtime 0.40.4; rejection takes 0.134–0.325 seconds. The
largest crop without these HTTP failures in this snapshot is 720 × 540. This
supports an association between oversized noisy-depth proposals and request
rejection. Image dimensions and the context setting alone do not reveal the
server's actual image token count or exact rejection reason.

Thirty-seven of the 41 affected scenes have no other description-available
crop, blocking CLIP description evidence for 444 CAD queries. Four scenes retain
other valid crops, so the failed crop is excluded while the remaining candidates
can proceed under the frozen rank-available rule. No correctness claim is made
about either the excluded or remaining candidates.

The remaining unavailable row is `cad__TLESS_000007__v01`, phrases mode. The
model finished normally (430 prompt tokens, 57 generated tokens), but wrote
`< MATERIAL>` with a space after `<`. The strict saved parser reports
`missing_empty_duplicate_or_nested_MATERIAL`. The original text is retained
verbatim. Since CLIP requires valid description evidence from every CAD view,
this blocks that CAD's twenty real-pilot queries. Depth-only and size-only
selection do not consume those descriptions.

The combined **description-stage** inventory is therefore 444 crop-blocked
queries plus twenty CAD-blocked queries. Later CLIP-vector validity checks may
cause additional unavailability. These counts are pipeline availability, not
classification accuracy, false negatives, or proof that objects are absent.
The deliberately withheld final scene is outside this first-snapshot inventory.

Evidence:

- `first_snapshot_measurements.json`: request/image dimensions, timing, exact
  malformed CAD output, and input mapping.
- `first_snapshot_unavailable_impact.json`: every affected crop/scene and every
  query blocked at the description stage.
- `log_availability.json`: relevant log-availability checks. No unrelated log
  text is copied; no current error excerpt was found.
- `diagnose_runtime.py`: reproducible, label-free inventory; no model calls.
